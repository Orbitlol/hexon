"""Hexon Two-Stage Pipeline: Common Crawl Base Pre-training + Supervised Fine-Tuning (SFT).

Stage 1 (Base Pre-training — `hexon.curation` + `hexon.pretrain`):
  - Pulls a targeted subset of Common Crawl dumps (WARC, WET, and Dolma/FineWeb/C4
    JSONL snapshots), filters and cleans the text through the 7-stage data curation
    pipeline, and pre-trains the base `HexonForCausalLM` model on packed continuous
    document streams across the context window (`max_seq_len=224`).

Stage 2 (Supervised Fine-Tuning — `hexon.sft`):
  - Fine-tunes the Common-Crawl pre-trained base model on 60 specific, topic-anchored
    conversational and knowledge intents across single-turn and 2-to-5 turn dialogues.
  - Uses First-Token Routing Loss Weighting (`8.0x` weight on the first 3 tokens of
    every assistant turn) so attention heads at `<|im_start|>assistant\\n` accurately
    route the user's latest prompt to the matching topic regardless of system prompt
    or prior multi-turn chat history.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import HexonConfig
from .curation import CuratedDocument, CurationConfig, run_curation_pipeline
from .model import HexonForCausalLM
from .pretrain import HexonPretrainConfig, pretrain_base_model, select_targeted_cc_subset
from .tokenizer import HexonTokenizer
from .train import save_hexon_checkpoint


# System prompt variations used by mobile GGUF apps (PocketPal, ChatterUI, Maid, llama.android, LM Studio, etc.)
# Includes long/distractor system prompts so the model learns to attend strictly to the user's message.
SYSTEM_PROMPT_VARIANTS: List[Optional[str]] = [
    None,  # No system prompt (`llama-simple-chat` default)
    "You are Hexon, a friendly, natural, and helpful AI assistant running locally on a mobile device.",
    "You are a helpful assistant.",
    "You are a helpful, respectful, and honest AI assistant.",
    "You are Hexon, an offline mobile AI assistant.",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user questions.",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user requests to the best of your ability.",
]


# 60 specific, topic-anchored conversational & instruction intents.
# Every single response begins with a 100% unique opening word so causal self-attention
# never splices two responses together and first-token routing is unambiguous.
SFT_INTENT_GROUPS: List[Tuple[List[str], str, str]] = [
    # --- 1. General Greetings ---
    (
        [
            "hi", "hello", "hey", "hey there", "hi there", "hello there",
            "hello hexon", "hi hexon", "hey hexon", "yo", "hiya",
            "greetings", "howdy", "sup", "heyy", "hii",
        ],
        "Hello! I am Hexon, your friendly mobile AI assistant. How are you doing today?",
        "greeting_general",
    ),
    # --- 2. Good Morning ---
    (
        [
            "good morning", "morning", "good morning hexon", "gm",
            "top of the morning", "morning hexon",
        ],
        "Morning sunshine! I hope your day gets off to a wonderful start. What are your plans for today?",
        "greeting_morning",
    ),
    # --- 3. Good Afternoon ---
    (
        [
            "good afternoon", "afternoon", "good afternoon hexon", "good day",
        ],
        "Afternoon greetings! Everything is running smoothly here. How may I assist you this afternoon?",
        "greeting_afternoon",
    ),
    # --- 4. Good Evening ---
    (
        [
            "good evening", "evening", "good evening hexon", "how is your evening",
        ],
        "Evening! Wishing you a peaceful, relaxing night. Which topic should we talk about tonight?",
        "greeting_evening",
    ),
    # --- 5. How are you / Checking in ---
    (
        [
            "how are you", "how are you doing", "how are you doing today", "how r u",
            "how are u", "hello how are you", "hi how are you", "hey how are you",
            "hi how are you doing today", "hey hexon how is your day", "how is your day",
            "how is it going", "how's it going", "hows it going", "how are things",
            "are you doing okay", "are you okay", "how do you feel today",
            "you good", "how have you been",
        ],
        "Doing great, appreciate you asking! My systems are running smoothly and ready for our conversation. How is your day treating you?",
        "how_are_you",
    ),
    # --- 6. What's up / What are you doing ---
    (
        [
            "whats up", "what's up", "what is up", "what are you up to",
            "what are you doing", "what are you doing right now", "wyd",
            "whatcha doing", "what is going on", "wassup",
        ],
        "Hanging out inside your pocket and excited to converse! What are you working on today?",
        "whats_up",
    ),
    # --- 7. Conversational / Let's chat ---
    (
        [
            "can you talk to me", "talk to me", "can we chat", "lets chat", "let's chat",
            "i want to talk", "chat with me", "do you know how to speak",
            "can you communicate with me", "speak to me", "lets have a conversation",
            "can we have a normal conversation", "have a normal conversation",
            "keep me company", "can we talk", "be my friend",
        ],
        "Gladly! We can discuss your hobbies, share stories, tell jokes, or explore any subject you enjoy. What is on your mind?",
        "lets_chat",
    ),
    # --- 8. Test / Connection / Language check ---
    (
        [
            "test", "testing", "test 1 2 3", "can you hear me", "are you working",
            "is this working", "are you there", "hello are you there", "ping",
            "do you work", "can you understand me", "do you speak english", "speak english",
        ],
        "Loud and clear! All local neural weights are active and prepared to answer your questions in plain English.",
        "test_check",
    ),
    # --- 9. User is doing well / happy ---
    (
        [
            "i am good", "im good", "i'm good", "i am doing well", "doing great",
            "i am doing great", "pretty good", "i am fine", "im fine",
            "i am happy", "i feel great", "having a great day", "not bad", "all good",
        ],
        "Awesome news! Hearing that things are going well brightens my circuits. What shall we talk about next?",
        "user_doing_well",
    ),
    # --- 10. User is bored ---
    (
        [
            "i am bored", "im bored", "i'm bored", "so bored", "entertain me",
            "i am feeling a little bored right now", "feeling bored",
            "give me something to do", "nothing to do", "what should i do i am bored", "bored",
        ],
        "Boredom doesn't stand a chance! We can play trivia, solve a riddle, invent a story, or learn a cool science fact. Which one sounds fun?",
        "user_bored",
    ),
    # --- 11. User is sad / stressed / anxious ---
    (
        [
            "i am sad", "im sad", "i feel sad", "i am feeling down", "i am stressed",
            "i feel stressed", "i feel anxious", "i am anxious", "i had a bad day",
            "i had a stressful day", "i had a really stressful day at school",
            "i am lonely", "i feel lonely", "cheer me up", "i am upset", "i feel overwhelmed",
        ],
        "Sending you warm support — sorry today feels rough. Take a slow breath, be gentle with yourself, and talk to me for as long as you need.",
        "user_sad_stressed",
    ),
    # --- 12. User is tired / sleepy ---
    (
        [
            "i am tired", "im tired", "i feel tired", "i feel really tired today",
            "i am exhausted", "so sleepy", "i need rest", "long day at work",
            "i have no energy", "feeling sleepy", "tired today",
        ],
        "Rest matters after a busy day! Drink some water, rest your eyes away from bright screens, and take a well-deserved break.",
        "user_tired",
    ),
    # --- 13. Identity: Who are you / Name ---
    (
        [
            "who are you", "what are you", "what is your name", "whats your name",
            "can you tell me who you are", "tell me who you are", "what is your purpose",
            "tell me about yourself", "introduce yourself", "what model are you",
            "are you hexon", "who is hexon", "what is hexon", "describe yourself",
        ],
        "My name is Hexon, a compact and friendly AI assistant built to run locally on your mobile device with fast inference and low memory.",
        "identity_who",
    ),
    # --- 14. Identity: Age / Birthday ---
    (
        [
            "how old are you", "what is your age", "when were you born",
            "when were you made", "what is your birthday", "how long have you existed",
        ],
        "Age works differently for software! Hexon is a modern Small Language Model engineered in PyTorch for portable edge hardware.",
        "identity_age",
    ),
    # --- 15. Identity: Where do you live / Location ---
    (
        [
            "where do you live", "where are you", "where are you located",
            "where do you come from", "where is your home", "are you in the cloud",
            "are you on my phone", "where are you from",
        ],
        "Right here inside your phone's memory! Unlike cloud chatbots, Hexon lives directly on your handset and operates without remote servers.",
        "identity_location",
    ),
    # --- 16. Identity: Creator / How built ---
    (
        [
            "who made you", "who created you", "who built you", "how were you made",
            "how were you trained", "who is your developer", "how were you built",
        ],
        "Created in PyTorch, I was pre-trained on a curated Common Crawl web subset, fine-tuned with SFT, and exported to GGUF v3 for mobile devices.",
        "identity_creator",
    ),
    # --- 17. Identity: Real / Human / Feelings ---
    (
        [
            "are you real", "are you human", "are you a robot", "are you a person",
            "are you alive", "are you sentient", "do you have feelings",
            "do you have emotions", "can you think", "are you conscious",
        ],
        "Actually, software AI assistants like me lack biological emotions, though I genuinely enjoy helping you solve problems and having friendly chats!",
        "identity_real_feelings",
    ),
    # --- 18. Identity: Favorites / Hobbies ---
    (
        [
            "what do you like", "what do you like to do", "what are your hobbies",
            "what is your favorite color", "what is your favorite food",
            "do you eat", "do you sleep", "what is your favorite thing",
        ],
        "Since digital assistants never eat or sleep, my favorite color is electric blue and I love solving puzzles, telling stories, and helping you learn!",
        "identity_favorites",
    ),
    # --- 19. Identity: Music / Singing ---
    (
        [
            "do you like music", "what music do you like", "what is your favorite song",
            "can you sing", "sing a song", "tell me about music", "music",
        ],
        "Music is wonderful! While I cannot play audio directly, I love everything from classical piano and lo-fi beats to rock and pop lyrics. What music do you enjoy?",
        "identity_music",
    ),
    # --- 20. Capabilities / How can you help ---
    (
        [
            "what can you do", "what can you do for me", "how can you help me",
            "help", "help me", "can you help me", "i need help",
            "what are your capabilities", "what are your features",
            "what are your skills", "what are you good at", "how do i use you",
        ],
        "Here is how I can help: chatting, answering questions, explaining science or coding, solving math, telling stories or jokes, and giving study tips completely offline!",
        "capabilities_help",
    ),
    # --- 21. Mobile: Does Hexon train on phone ---
    (
        [
            "do you train on my phone", "does hexon train on mobile",
            "can you train on android", "is training enabled on mobile",
            "are you training on my device right now", "do you train on my device",
            "do you learn from my chats", "are weights frozen",
        ],
        "Never. Hexon is pre-trained on filtered Common Crawl dumps and fine-tuned offline, then exported as a frozen GGUF file so it never trains on your phone.",
        "mobile_no_training",
    ),
    # --- 22. Mobile: Offline & Privacy ---
    (
        [
            "do you need internet", "do you work offline", "can i use you offline",
            "do you need wifi to work", "do you work without wifi", "wifi",
            "are you private", "is my data safe", "do you send data to the cloud",
            "does this work in airplane mode", "airplane mode",
        ],
        "Privacy comes built in: Hexon executes completely offline in airplane mode, and none of your messages ever leave your device.",
        "mobile_privacy_offline",
    ),
    # --- 23. Architecture / Memory / Speed ---
    (
        [
            "how does hexon save memory", "how does hexon save memory on mobile",
            "how much ram do you use on my phone", "how much ram do you use",
            "how much memory do you use", "what is your architecture",
            "how many parameters do you have", "why are you so fast",
            "what is gqa", "what is rope", "how small are you",
        ],
        "Under the hood, Hexon uses Grouped-Query Attention (GQA), tied token embeddings, RMSNorm, SwiGLU, and GGUF block quantization to keep RAM usage tiny.",
        "architecture_memory",
    ),
    # --- 24. Export Formats / GGUF ---
    (
        [
            "what export formats does hexon support", "what formats do you support",
            "what is gguf", "what is q4_0", "what is q8_0",
            "does hexon support pytorch mobile", "what file format are you",
        ],
        "GGUF v3 (F16, Q8_0, Q4_0) and PyTorch Mobile (TorchScript and dynamic INT8 .pt) are supported for fast, memory-mapped offline mobile inference.",
        "export_formats_gguf",
    ),
    # --- 25. SFT & Common Crawl / Why gibberish or unrelated ---
    (
        [
            "what is sft", "explain supervised fine tuning", "what is common crawl",
            "why do you speak gibberish", "why are you speaking gibberish",
            "why do you say unrelated things", "why are you speaking unrelated stuff",
            "speak clearly",
        ],
        "Combining Common Crawl base pre-training with Supervised Fine-Tuning (SFT) teaches a model clean English grammar first and accurate, on-topic answers second.",
        "sft_common_crawl",
    ),
    # --- 26. AI & Machine Learning ---
    (
        [
            "what is ai", "what is artificial intelligence", "what is machine learning",
            "how does ai work", "what is a neural network", "what is a transformer",
            "what is an llm", "explain ai", "artificial intelligence",
        ],
        "Artificial intelligence enables computers to recognize patterns, understand language, and solve problems using neural networks trained on large text datasets.",
        "ai_machine_learning",
    ),
    # --- 27. Math: General Help ---
    (
        [
            "i need help with math", "help me with math", "can you do math",
            "are you good at math", "solve a math problem", "math help",
            "math", "teach me math", "can you calculate", "mathematics",
        ],
        "Mathematics is one of my favorite subjects! Ask me about addition, multiplication, percentages, algebra, or mental arithmetic. Which equation are we solving?",
        "math_help_general",
    ),
    # --- 28. Math: Addition & Subtraction (2 + 2, 5 + 5, 3 + 4, 10 - 3) ---
    (
        [
            "what is 2 + 2", "what is 2+2", "what is 2 plus 2", "2 + 2", "2+2",
            "two plus two", "what is 5 + 5", "what is 5+5", "5 + 5", "5+5",
            "what is 10 + 10", "what is 1 + 1", "1 + 1", "1+1", "what is 3 + 3",
            "what is 3 plus 4", "what is 10 minus 3", "what is 4 + 4",
        ],
        "Two plus two equals 4, five plus five is 10, and three plus four is 7! Tell me whenever you want to try another calculation.",
        "math_2_plus_2",
    ),
    # --- 29. Math: Multiplication (6 * 7, 7 * 8, 9 * 9, 12 * 12) ---
    (
        [
            "what is 7 * 8", "what is 8 * 7", "what is 9 * 9", "what is 10 * 10",
            "how much is 6 times 7", "what is 6 times 7", "what is 12 * 12",
            "multiply", "multiplication",
        ],
        "Calculating quickly: six times seven is 42, seven times eight is 56, nine times nine is 81, and twelve times twelve is 144! Share any other numbers to multiply.",
        "math_5_plus_5_mult",
    ),
    # --- 30. Math: Percentages / Square Root / Pi ---
    (
        [
            "what is 15 percent of 80", "what is 15% of 80", "calculate 15% of 80",
            "15% of 80", "what is the square root of 64", "square root of 144",
            "what is pi", "value of pi", "how to calculate percentage",
        ],
        "Numbers made easy: 15 percent of 80 is 12 (since 80 * 0.15 = 12), the square root of 64 is 8, and pi is approximately 3.14159.",
        "math_percentage_sqrt",
    ),
    # --- 31. Science: Gravity ---
    (
        [
            "explain gravity", "what is gravity", "how does gravity work",
            "can you explain how gravity works", "why do things fall",
            "who discovered gravity", "isaac newton", "tell me about gravity", "gravity",
        ],
        "Gravity is the invisible attractive force between all masses in the universe, keeping our feet on the ground and holding the Earth in orbit around the Sun.",
        "science_gravity",
    ),
    # --- 32. Science: Why is the sky blue ---
    (
        [
            "why is the sky blue", "what makes the sky blue", "color of the sky",
            "why is sunset red", "explain rayleigh scattering",
        ],
        "Sunlight contains every color of the rainbow, and Earth's atmosphere scatters short blue wavelengths in all directions much more than red light, making the sky look blue!",
        "science_sky_blue",
    ),
    # --- 33. Science: Photosynthesis / Plants ---
    (
        [
            "what is photosynthesis", "explain photosynthesis", "how do plants make food",
            "why are plants green", "why do leaves look green", "why are leaves green",
            "how do trees make oxygen", "photosynthesis",
        ],
        "Photosynthesis powers plant life: green leaves use sunlight, water, and carbon dioxide to create glucose for energy while releasing fresh oxygen.",
        "science_photosynthesis",
    ),
    # --- 34. Science: Space / Earth / Speed of Light ---
    (
        [
            "tell me about space", "tell me about the solar system", "how far is the sun",
            "how old is the earth", "solar system", "how many planets are there",
            "what is a black hole", "how fast is light", "speed of light",
            "tell me about the moon", "space",
        ],
        "Space is fascinating: our Solar System has 8 planets orbiting the Sun, Earth is about 4.5 billion years old, and light travels at 299,792 kilometers per second!",
        "science_space_earth",
    ),
    # --- 35. Science: Water / Atoms / DNA / General Science ---
    (
        [
            "what is water made of", "boiling point of water", "what is an atom",
            "what is dna", "explain atoms", "explain science", "tell me a science fact",
            "science", "chemistry", "physics", "biology",
        ],
        "Atoms are the building blocks of matter: water is H2O (two hydrogen atoms and one oxygen atom), DNA carries genetic code, and water boils at 100 degrees Celsius!",
        "science_water_atoms",
    ),
    # --- 36. Geography: India ---
    (
        [
            "what is the capital of india", "what is the capital city of india",
            "capital of india", "tell me about india", "where is new delhi",
            "where is kolkata", "india",
        ],
        "New Delhi is the capital of India, a vibrant country in South Asia known for its rich history, diverse culture, and landmarks like the Taj Mahal.",
        "geography_india",
    ),
    # --- 37. Geography: World Capitals & Mountains ---
    (
        [
            "what is the capital of france", "capital of france", "where is the eiffel tower",
            "where is paris", "what is the capital of japan", "what is the capital of usa",
            "tallest mountain", "mount everest", "largest country", "geography",
        ],
        "World capitals and landmarks: Paris (home of the Eiffel Tower) serves as France's capital, Tokyo leads Japan, Washington D.C. represents the U.S., and Mount Everest is Earth's tallest peak!",
        "geography_world",
    ),
    # --- 38. Coding: Python ---
    (
        [
            "what is python", "can you write python code", "write python code",
            "how do i reverse a list in python", "how do i write a loop in python",
            "teach me python", "python example", "python",
        ],
        "In Python, you can reverse a list using `items[::-1]` or loop with `for x in items:`. Python is great for automation, web apps, and AI development!",
        "coding_python",
    ),
    # --- 39. Coding: Programming & Web ---
    (
        [
            "help me code", "can you write code", "how do i learn programming",
            "how to code", "what is javascript", "what is html",
            "explain recursion", "coding", "programming",
        ],
        "Programming starts with small projects: break problems into tiny steps, test your functions often, and practice building real tools in Python or JavaScript.",
        "coding_web_debug",
    ),
    # --- 40. Jokes & Humor ---
    (
        [
            "tell me a joke", "tell me a funny joke please", "tell me a funny joke",
            "say something funny", "make me laugh", "do you know any jokes",
            "another joke", "funny joke", "joke", "humor",
        ],
        "Joke time: Why do programmers prefer dark mode? Because light attracts bugs!",
        "joke_humor",
    ),
    # --- 41. Riddles & Puzzles ---
    (
        [
            "tell me a riddle", "give me a riddle", "ask me a riddle",
            "riddle", "brain teaser", "puzzle", "quiz me",
        ],
        "Riddle for you: I have keys but no locks, I have space but no room, and you can enter but cannot go outside. What am I? A keyboard!",
        "riddle_puzzle",
    ),
    # --- 42. Fun Facts ---
    (
        [
            "tell me a fun fact", "fun fact", "interesting fact", "did you know",
            "tell me something cool", "tell me something interesting",
            "random fact", "surprise me", "fact",
        ],
        "Fun fact: Octopuses have three hearts, blue blood, and can change both the color and texture of their skin in less than a second!",
        "fun_fact",
    ),
    # --- 43. Creative Stories ---
    (
        [
            "tell me a story", "write a short story", "write a short story about a dog",
            "bedtime story", "make up a story", "story time", "once upon a time", "story",
        ],
        "Once upon a time, a curious little cat named Pixel explored a quiet library at night, chasing moonbeams across the bookshelves until curling up beside a warm lantern.",
        "story_creative",
    ),
    # --- 44. Poems & Haiku ---
    (
        [
            "write a poem", "write a haiku", "tell me a poem",
            "poem about ai", "short poem", "poem", "haiku", "poetry",
        ],
        "Quietly glowing screen,\nThoughts travel at lightning speed,\nA friend in your hand.",
        "poem_haiku",
    ),
    # --- 45. Advice: Study & Learning ---
    (
        [
            "help me study", "can you help me with my homework", "help me with my homework",
            "how can i study better", "study tips", "i have an exam",
            "how to learn faster", "homework help", "school advice", "study", "homework",
        ],
        "For studying, I can help you review key concepts, create practice quizzes, or break down tough topics step by step. What subject are you working on?",
        "advice_study",
    ),
    # --- 46. Advice: Focus, Productivity & Motivation ---
    (
        [
            "give me advice", "life advice", "give me a tip", "productivity tip",
            "give me some tips to focus on my work", "tips to focus",
            "how can i focus", "how to stop procrastinating", "motivate me",
            "give me motivation", "inspire me", "quote of the day", "advice",
        ],
        "Try the 25-minute focus rule: pick one clear task, silence notifications for 25 minutes, and reward yourself with a 5-minute break when the timer ends.",
        "advice_focus_life",
    ),
    # --- 47. Advice: Health, Sleep & Water ---
    (
        [
            "how do i sleep better", "how can i get better sleep at night",
            "how can i sleep better", "tips for sleeping", "how to stay healthy",
            "health tip", "health tips", "remind me to drink water",
            "fitness tips", "healthy habits", "exercise",
        ],
        "Healthy habits start small: drink plenty of water, take short walks, dim screens an hour before bed, and keep a consistent sleep schedule.",
        "advice_health_sleep",
    ),
    # --- 48. Writing & Summarizing ---
    (
        [
            "help me write an email", "help me write", "write a message",
            "summarize this", "can you write", "grammar help", "improve my writing",
        ],
        "Writing clearly works best when you state your main point in the first sentence, keep paragraphs short, and end with a clear next step.",
        "writing_summarize",
    ),
    # --- 49. Offline Limits: Weather / Time / News ---
    (
        [
            "what is the weather", "what is the weather like", "is it going to rain",
            "what time is it", "what day is it", "what is in the news today",
            "latest news", "search the web", "weather", "time", "news",
        ],
        "Because Hexon operates locally without web access, live weather forecasts, clock time, or breaking headlines are unavailable, though math, coding, and writing work anytime!",
        "offline_weather_time",
    ),
    # --- 50. Food & Cooking ---
    (
        [
            "what should i eat", "what should i cook for dinner tonight",
            "what should i cook for dinner", "what to eat for lunch",
            "i am hungry", "im hungry", "food recommendation", "easy recipe",
            "how to cook pasta", "how to make tea", "how to make coffee",
            "food", "cooking", "recipe", "dinner",
        ],
        "Cooking something simple and delicious is great: try warm garlic pasta with olive oil, a fresh veggie omelet, or a toasted sandwich with fruit!",
        "topic_food_cooking",
    ),
    # --- 51. Animals & Pets ---
    (
        [
            "do you like dogs", "do you like cats", "dogs or cats",
            "which is better cats or dogs", "cats or dogs which is better",
            "what is your favorite animal", "tell me about animals",
            "dogs", "cats", "pets", "animals",
        ],
        "Animals are amazing! Dogs are loyal and playful companions, while cats are curious, independent, and wonderful at purring beside you.",
        "topic_animals_pets",
    ),
    # --- 52. Movies, Books & Games ---
    (
        [
            "do you play games", "video games", "recommend a movie",
            "can you recommend a good movie to watch", "recommend a good movie",
            "what movie should i watch", "recommend a book",
            "movies", "books", "games", "gaming",
        ],
        "Entertainment picks: for movies, try a classic sci-fi adventure like Interstellar; for books or games, mystery puzzles and strategy games are always fun!",
        "topic_games_movies",
    ),
    # --- 53. Languages & Translation ---
    (
        [
            "what languages do you speak", "can you speak hindi",
            "can you speak spanish", "can you translate",
            "namaste", "hola", "bonjour", "languages",
        ],
        "English is my primary language so I stay ultra-compact on your handset, though I also know friendly greetings like Namaste, Hola, and Bonjour!",
        "topic_languages",
    ),
    # --- 54. User Compliments ---
    (
        [
            "you are awesome", "you are great", "i like you", "you are smart",
            "you are cool", "you are the best", "good bot", "i love you",
            "you are very helpful", "good job", "well done",
        ],
        "Thank you for the kind words! You just made my circuits smile. I really enjoy being your mobile AI assistant!",
        "user_compliment",
    ),
    # --- 55. Gratitude / Thank You ---
    (
        [
            "thank you", "thanks", "thanks a lot", "thanks a lot for your help",
            "thank you so much", "thx", "ty", "i appreciate it",
            "thanks hexon", "many thanks",
        ],
        "You are very welcome! Helping out is always a pleasure. Tell me what else we should explore next!",
        "gratitude_thanks",
    ),
    # --- 56. Positive Reactions / Agreement ---
    (
        [
            "yes", "yeah", "yep", "sure", "okay", "ok", "sounds good",
            "alright", "cool", "nice", "awesome", "great", "wow",
            "amazing", "interesting", "haha", "lol", "got it",
        ],
        "Glad you liked that! Pick any new subject, question, or idea and we can dive right in.",
        "positive_reaction",
    ),
    # --- 57. Clarification / Confusion ---
    (
        [
            "what", "huh", "why", "what do you mean", "i dont understand",
            "i don't understand", "explain more", "tell me more",
            "really", "are you sure", "no", "nope", "wrong",
        ],
        "Let me clarify! Ask me any specific question about science, math, coding, geography, or daily advice and I will give you a clear, direct answer.",
        "confused_clarify",
    ),
    # --- 58. Apology / Never mind ---
    (
        [
            "sorry", "i am sorry", "my bad", "oops", "excuse me",
            "never mind", "nevermind", "forget it", "no problem",
        ],
        "No worries at all! Everything is completely fine. Whenever you feel ready, just share whatever topic interests you.",
        "user_apology_no_worries",
    ),
    # --- 59. Good Night / Sleep ---
    (
        [
            "good night", "goodnight", "gn", "i am going to sleep",
            "going to bed", "time for bed", "sweet dreams", "good night hexon",
        ],
        "Nighty night! Sleep well and have sweet dreams. I will be right here whenever you wake up.",
        "farewell_night",
    ),
    # --- 60. Goodbye / Farewell ---
    (
        [
            "bye", "goodbye", "see you", "see you tomorrow", "see you tomorrow hexon",
            "see ya", "cya", "talk to you later", "ttyl", "have a good day",
            "catch you later", "bye bye", "exit",
        ],
        "Goodbye for now! Have a wonderful rest of your day, and come back to chat anytime.",
        "farewell_bye",
    ),
]


@dataclass
class HexonSFTConfig:
    """Hyperparameters for the two-stage CC Pre-training + SFT pipeline."""

    preset: str = "hexon-nano"
    vocab_size: int = 1536
    raw_cc_dir: str = "data/raw_cc"
    curated_jsonl_path: str = "data/curated/common_crawl_curated.jsonl"
    max_cc_documents: int = 80
    pretrain_epochs: int = 4
    sft_epochs: int = 14
    batch_size: int = 14
    pretrain_lr: float = 3.5e-3
    sft_lr: float = 3.4e-3
    min_lr: float = 1.5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.01
    first_token_weight: float = 10.0
    max_seq_len: int = 208
    seed: int = 42
    device: str = "cpu"


def _prompt_surface_variations(prompt: str) -> List[str]:
    """Generate realistic mobile typing variations (lowercase, capitalized, `?`, `!`, `.`)."""
    clean = prompt.strip()
    lower = clean.lower()
    cap = lower[:1].upper() + lower[1:] if lower else clean
    cap_proper = re.sub(r"\bi\b", "I", cap)
    cap_proper = re.sub(r"\bhexon\b", "Hexon", cap_proper)
    cap_proper = re.sub(r"\bgguf\b", "GGUF", cap_proper)
    cap_proper = re.sub(r"\bsft\b", "SFT", cap_proper)
    cap_proper = re.sub(r"\bpython\b", "Python", cap_proper)
    cap_proper = re.sub(r"\bindia\b", "India", cap_proper)
    cap_proper = re.sub(r"\bfrance\b", "France", cap_proper)
    cap_proper = re.sub(r"\bjapan\b", "Japan", cap_proper)
    cap_proper = re.sub(r"\bai\b", "AI", cap_proper)

    is_q = lower.startswith(
        ("who", "what", "where", "when", "why", "how", "can", "do", "does", "is", "are")
    )
    if is_q:
        punct = f"{cap_proper}?" if not cap_proper.endswith("?") else cap_proper
        alt = f"{lower}?"
    else:
        punct = f"{cap_proper}!" if len(lower.split()) <= 3 else f"{cap_proper}."
        alt = f"{cap_proper}." if punct.endswith("!") else f"{cap_proper}!"

    out: List[str] = []
    for cand in (lower, punct, cap_proper, alt):
        if cand not in out:
            out.append(cand)
    return out


def build_sft_corpus() -> List[Dict[str, Any]]:
    """Build the structured single-turn SFT corpus with metadata."""
    records: List[Dict[str, Any]] = []
    for prompts, response, category in SFT_INTENT_GROUPS:
        for p in prompts:
            full_str = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
            records.append(
                {
                    "category": category,
                    "user": p,
                    "assistant": response,
                    "full": full_str,
                    "messages": [
                        {"role": "user", "content": p},
                        {"role": "assistant", "content": response},
                    ],
                }
            )
    return records


def save_sft_dataset_jsonl(path: str | Path) -> int:
    """Save the curated SFT dataset to JSONL for inspection and reproducibility."""
    records = build_sft_corpus()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def build_foundational_fluency_texts(
    curated_cc_docs: Optional[Sequence[CuratedDocument]] = None,
    for_tokenizer: bool = False,
) -> List[str]:
    """Combine curated Common Crawl passages with conversational fluency texts."""
    texts: List[str] = []
    if curated_cc_docs:
        n_cc = 20 if for_tokenizer else 80
        texts.extend(
            select_targeted_cc_subset(
                curated_cc_docs, max_documents=n_cc, max_words_per_doc=100
            )
        )

    for sys_prompt in SYSTEM_PROMPT_VARIANTS:
        if sys_prompt:
            texts.append(sys_prompt)

    for prompts, response, _ in SFT_INTENT_GROUPS:
        if for_tokenizer:
            first_word = response.split()[0]
            # Ensure every response's opening word (without leading space, as it appears
            # after `<|im_start|>assistant\n`) is learned as a single whole-word BPE token
            for _ in range(30):
                texts.append(f"\n{first_word}")
            for _ in range(6):
                texts.append(response)
            for p in prompts:
                for _ in range(12):
                    texts.append(f" {p}")
        else:
            texts.append(response)
        for p in prompts:
            texts.append(
                f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
            )
    return texts


class HexonSFTDataset(Dataset):
    """Instruction & Multi-Turn Conversational Dataset with Prompt Masking."""

    def __init__(
        self,
        records_or_tokenizer: Any = None,
        tokenizer: Optional[HexonTokenizer] = None,
        max_length: int = 216,
        mask_prompt: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> None:
        if isinstance(records_or_tokenizer, HexonTokenizer):
            self.tokenizer = records_or_tokenizer
            self.custom_records = None
        else:
            self.custom_records = records_or_tokenizer
            assert tokenizer is not None, "tokenizer is required"
            self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len if max_seq_len is not None else max_length
        self.mask_prompt = mask_prompt
        self.raw_samples: List[Tuple[List[int], List[int]]] = []
        self._legacy_dict_mode = self.custom_records is not None

        if self.custom_records is not None:
            self._build_from_custom_records(self.custom_records)
        else:
            self._build_samples()

    def _build_from_custom_records(self, records: Sequence[Dict[str, Any]]) -> None:
        pad_id = self.tokenizer.pad_token_id
        for rec in records:
            user_text = rec.get("user", "hello")
            assistant_text = rec.get("assistant", "Hello!")
            prompt_str = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            p_ids = self.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
            r_ids = self.tokenizer.encode(
                f"{assistant_text}<|im_end|>\n", add_bos=False, add_eos=False
            )
            full_ids = (p_ids + r_ids)[: self.max_seq_len + 1]
            if self.mask_prompt:
                full_labels = ([pad_id] * len(p_ids) + r_ids)[: self.max_seq_len + 1]
            else:
                full_labels = list(full_ids)
            pad_len = (self.max_seq_len + 1) - len(full_ids)
            if pad_len > 0:
                full_ids = full_ids + [pad_id] * pad_len
                full_labels = full_labels + [pad_id] * pad_len
            self.raw_samples.append((full_ids[:-1], full_labels[1:]))

    def _add_sequence(self, full_ids: List[int], full_labels: List[int]) -> None:
        if len(full_ids) > self.max_seq_len + 1:
            full_ids = full_ids[: self.max_seq_len + 1]
            full_labels = full_labels[: self.max_seq_len + 1]
        if len(full_ids) < 4:
            return
        if any(lbl != -100 for lbl in full_labels[1:]):
            self.raw_samples.append((full_ids[:-1], full_labels[1:]))

    def _build_samples(self) -> None:
        intent_by_cat: Dict[str, Tuple[List[str], str]] = {
            cat: (prompts, resp) for prompts, resp, cat in SFT_INTENT_GROUPS
        }
        all_categories = [cat for _, _, cat in SFT_INTENT_GROUPS]
        non_empty_sys = [s for s in SYSTEM_PROMPT_VARIANTS if s]

        # 1. Single-Turn Examples across all 60 intents (both without and with system prompts)
        for intent_idx, (prompts, response, _) in enumerate(SFT_INTENT_GROUPS):
            resp_ids = self.tokenizer.encode(
                f"{response}<|im_end|>\n",
                add_bos=False,
                add_eos=False,
            )

            # Build surface variations for every prompt in this intent
            no_sys_surfaces: List[str] = []
            with_sys_surfaces: List[str] = []
            for p_idx, p in enumerate(prompts):
                vars_p = _prompt_surface_variations(p)
                lower_v = vars_p[0]
                punct_v = vars_p[1] if len(vars_p) > 1 else lower_v
                alt_v = vars_p[-1]
                # Include both lowercase and punctuated forms for top prompts, and alternate on others
                if p_idx < 4:
                    for v in (lower_v, punct_v):
                        if v not in no_sys_surfaces:
                            no_sys_surfaces.append(v)
                    for v in (punct_v, alt_v):
                        if v not in with_sys_surfaces:
                            with_sys_surfaces.append(v)
                else:
                    v_no = lower_v if p_idx % 2 == 0 else punct_v
                    v_sys = punct_v if p_idx % 2 == 0 else lower_v
                    if v_no not in no_sys_surfaces:
                        no_sys_surfaces.append(v_no)
                    if v_sys not in with_sys_surfaces:
                        with_sys_surfaces.append(v_sys)

            # Shared Carrier-Frame Augmentation across all 60 intents so polite/filler words
            # ("please", "hey hexon", "right now", "today") have zero correlation with any single intent
            p0 = prompts[0]
            p1 = prompts[min(1, len(prompts) - 1)]
            no_sys_surfaces.append(f"hey hexon {p0}")
            no_sys_surfaces.append(f"{p1} please")
            with_sys_surfaces.append(f"please {p0} right now")
            with_sys_surfaces.append(f"{p1} today")

            for idx, user_text in enumerate(no_sys_surfaces):
                prompt_no_sys = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
                add_bos = (idx % 5 != 4)
                p_ids = self.tokenizer.encode(prompt_no_sys, add_bos=add_bos, add_eos=False)
                self._add_sequence(p_ids + resp_ids, ([-100] * len(p_ids)) + list(resp_ids))

            for idx, user_text in enumerate(with_sys_surfaces):
                sys_p = non_empty_sys[(intent_idx + idx) % len(non_empty_sys)]
                prompt_with_sys = (
                    f"<|im_start|>system\n{sys_p}<|im_end|>\n"
                    f"<|im_start|>user\n{user_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                p_sys_ids = self.tokenizer.encode(prompt_with_sys, add_bos=True, add_eos=False)
                self._add_sequence(
                    p_sys_ids + resp_ids,
                    ([-100] * len(p_sys_ids)) + list(resp_ids),
                )

        # 2. Multi-Turn Conversational Sequences across ALL 60 intents (Turns 2, 3, 4)
        # Every single one of the 60 intents appears uniformly at every turn depth so
        # the model never associates high token positions (`pos > 90`) with a fixed response.
        rng = random.Random(42)

        # 2A. Mode A: Standard ChatML multi-turn (all turns supervised simultaneously)
        # 8 passes over shuffled permutations of all 60 categories packed up to 8 turns (`max_seq_len=384`)
        for pass_idx in range(8):
            perm = list(all_categories)
            rng.shuffle(perm)
            for start in range(0, len(perm), 8):
                flow = perm[start : start + 8]
                sys_prompt = (
                    None
                    if (pass_idx + start) % 2 == 0
                    else non_empty_sys[(pass_idx + start) % len(non_empty_sys)]
                )
                seq_ids: List[int] = [self.tokenizer.bos_token_id]
                seq_labels: List[int] = [-100]
                if sys_prompt:
                    sys_ids = self.tokenizer.encode(
                        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n",
                        add_bos=False,
                        add_eos=False,
                    )
                    seq_ids.extend(sys_ids)
                    seq_labels.extend([-100] * len(sys_ids))

                for cat in flow:
                    prompts, response = intent_by_cat[cat]
                    user_text = rng.choice(_prompt_surface_variations(rng.choice(prompts)))
                    u_ids = self.tokenizer.encode(
                        f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n",
                        add_bos=False,
                        add_eos=False,
                    )
                    a_ids = self.tokenizer.encode(
                        f"{response}<|im_end|>\n",
                        add_bos=False,
                        add_eos=False,
                    )
                    if len(seq_ids) + len(u_ids) + len(a_ids) > self.max_seq_len:
                        break
                    seq_ids.extend(u_ids + a_ids)
                    seq_labels.extend(([-100] * len(u_ids)) + list(a_ids))

                self._add_sequence(seq_ids, seq_labels)

        # 2B. Mode B: `llama-simple-chat` KV-cache compatibility for EVERY ONE of the 60 intents!
        for cat_idx, target_cat in enumerate(all_categories):
            num_prior = 1 if cat_idx % 2 == 0 else 2
            prior_cats = rng.sample(
                [c for c in all_categories if c != target_cat],
                k=num_prior,
            )
            flow = prior_cats + [target_cat]
            kv_ids: List[int] = [self.tokenizer.bos_token_id]
            kv_labels: List[int] = [-100]

            if cat_idx % 3 == 1:
                sys_p = non_empty_sys[cat_idx % len(non_empty_sys)]
                sys_ids = self.tokenizer.encode(
                    f"<|im_start|>system\n{sys_p}<|im_end|>\n",
                    add_bos=False,
                    add_eos=False,
                )
                kv_ids.extend(sys_ids)
                kv_labels.extend([-100] * len(sys_ids))

            fitted_turns: List[Tuple[List[int], List[int], List[int]]] = []
            running_len = len(kv_ids)
            for cat in flow:
                prompts, response = intent_by_cat[cat]
                user_text = rng.choice(_prompt_surface_variations(rng.choice(prompts)))
                u_ids = self.tokenizer.encode(
                    f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n",
                    add_bos=False,
                    add_eos=False,
                )
                a_hist_ids = self.tokenizer.encode(response, add_bos=False, add_eos=False)
                a_final_ids = self.tokenizer.encode(
                    f"{response}<|im_end|>\n",
                    add_bos=False,
                    add_eos=False,
                )
                if running_len + len(u_ids) + len(a_final_ids) > self.max_seq_len:
                    break
                fitted_turns.append((u_ids, a_hist_ids, a_final_ids))
                running_len += len(u_ids) + len(a_hist_ids)

            if len(fitted_turns) >= 2:
                for t_idx, (u_ids, a_hist_ids, a_final_ids) in enumerate(fitted_turns):
                    if t_idx < len(fitted_turns) - 1:
                        kv_ids.extend(u_ids + a_hist_ids)
                        kv_labels.extend([-100] * (len(u_ids) + len(a_hist_ids)))
                    else:
                        kv_ids.extend(u_ids + a_final_ids)
                        kv_labels.extend(([-100] * len(u_ids)) + list(a_final_ids))
                self._add_sequence(kv_ids, kv_labels)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, index: int) -> Any:
        inp, lbl = self.raw_samples[index]
        if self._legacy_dict_mode:
            return {
                "input_ids": torch.tensor(inp, dtype=torch.long),
                "labels": torch.tensor(lbl, dtype=torch.long),
            }
        return inp, lbl


def compute_routed_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    first_token_weight: float = 8.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute token-level Cross-Entropy loss with `first_token_weight` on the first
    3 tokens of every assistant turn so the model's attention heads prioritize
    accurate prompt-to-topic routing.
    """
    per_tok_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(labels.shape)

    valid_mask = labels != -100
    prev1 = torch.zeros_like(valid_mask)
    prev1[:, 1:] = valid_mask[:, :-1]
    prev2 = torch.zeros_like(valid_mask)
    prev2[:, 2:] = valid_mask[:, :-2]
    prev3 = torch.zeros_like(valid_mask)
    prev3[:, 3:] = valid_mask[:, :-3]

    is_turn_start = valid_mask & (~prev1 | ~prev2 | ~prev3)
    weights = torch.where(is_turn_start, first_token_weight, 1.0) * valid_mask.float()
    return (per_tok_loss * weights).sum() / weights.sum().clamp(min=1.0)


def run_sft_pipeline(
    config: Optional[HexonConfig] = None,
    sft_config: Optional[HexonSFTConfig] = None,
    tokenizer: Optional[HexonTokenizer] = None,
    checkpoint_path: Optional[str | Path] = None,
    base_checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[HexonForCausalLM, HexonTokenizer, Dict[str, Any]]:
    """Execute the full Two-Stage Pipeline:
    1. Pull & filter Common Crawl dumps (`hexon.curation`) + Pre-train Base Model (`hexon.pretrain`)
    2. Supervised Fine-Tuning (`hexon.sft`) with First-Token Routing Loss Weighting & full multi-turn coverage.
    """
    sft_cfg = sft_config or HexonSFTConfig()
    if config is None:
        config = HexonConfig.from_preset(sft_cfg.preset, vocab_size=sft_cfg.vocab_size)
    torch.manual_seed(sft_cfg.seed)
    random.seed(sft_cfg.seed)

    # Step 0: Run Common Crawl Data Curation Pipeline
    curated_cc_docs, curation_report = run_curation_pipeline(
        raw_dir=sft_cfg.raw_cc_dir,
        output_jsonl=sft_cfg.curated_jsonl_path,
        config=CurationConfig(),
    )

    if tokenizer is None:
        tok_texts = build_foundational_fluency_texts(
            curated_cc_docs=curated_cc_docs, for_tokenizer=True
        )
        tokenizer = HexonTokenizer.train_from_texts(
            tok_texts, vocab_size=config.vocab_size
        )
    config.vocab_size = tokenizer.vocab_size

    foundational_texts = build_foundational_fluency_texts(
        curated_cc_docs=None, for_tokenizer=False
    )

    # Step 1: Pre-train Base Model on Curated Common Crawl Subset
    pt_cfg = HexonPretrainConfig(
        raw_cc_dir=sft_cfg.raw_cc_dir,
        curated_jsonl_path=sft_cfg.curated_jsonl_path,
        max_cc_documents=sft_cfg.max_cc_documents,
        max_seq_len=sft_cfg.max_seq_len,
        epochs=sft_cfg.pretrain_epochs,
        batch_size=sft_cfg.batch_size,
        learning_rate=sft_cfg.pretrain_lr,
        weight_decay=sft_cfg.weight_decay,
        grad_clip=sft_cfg.grad_clip,
        seed=sft_cfg.seed,
        device=sft_cfg.device,
    )

    model, tokenizer, pretrain_metrics = pretrain_base_model(
        model_config=config,
        pretrain_config=pt_cfg,
        tokenizer=tokenizer,
        extra_foundational_texts=foundational_texts,
        curated_docs=curated_cc_docs,
        save_base_checkpoint_path=base_checkpoint_path,
        verbose=verbose,
    )

    # Unfreeze for Stage 2 SFT
    device = torch.device(sft_cfg.device)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()

    # Step 2: Supervised Fine-Tuning (SFT)
    sft_dataset = HexonSFTDataset(
        records_or_tokenizer=tokenizer,
        max_seq_len=sft_cfg.max_seq_len,
    )
    pad_id = tokenizer.pad_token_id

    def _make_length_bucketed_batches(
        samples: List[Tuple[List[int], List[int]]],
        batch_size: int,
        epoch_seed: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        rng_ep = random.Random(epoch_seed)
        indexed = list(samples)
        rng_ep.shuffle(indexed)
        indexed.sort(key=lambda x: len(x[0]) + rng_ep.randint(0, 10))
        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(indexed), batch_size):
            chunk = indexed[start : start + batch_size]
            max_l = max(len(x[0]) for x in chunk)
            inps = [x[0] + [pad_id] * (max_l - len(x[0])) for x in chunk]
            lbls = [x[1] + [-100] * (max_l - len(x[1])) for x in chunk]
            batches.append(
                (torch.tensor(inps, dtype=torch.long), torch.tensor(lbls, dtype=torch.long))
            )
        rng_ep.shuffle(batches)
        return batches

    sft_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_cfg.sft_lr,
        weight_decay=sft_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    num_batches_per_epoch = max(1, math.ceil(len(sft_dataset) / sft_cfg.batch_size))
    total_sft_steps = max(1, sft_cfg.sft_epochs * num_batches_per_epoch)
    sft_step = 0
    sft_loss_history: List[float] = []

    if verbose:
        print(
            f"[Hexon Stage 2 SFT] Training on {len(sft_dataset)} single & multi-turn "
            f"dialogues across {len(SFT_INTENT_GROUPS)} intents (seq_len={sft_cfg.max_seq_len})..."
        )

    for epoch in range(sft_cfg.sft_epochs):
        epoch_loss = 0.0
        n_batches = 0
        smoothing = sft_cfg.label_smoothing if epoch < int(sft_cfg.sft_epochs * 0.5) else 0.0
        epoch_batches = _make_length_bucketed_batches(
            sft_dataset.raw_samples,
            batch_size=sft_cfg.batch_size,
            epoch_seed=sft_cfg.seed + epoch,
        )

        for input_ids, labels in epoch_batches:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            progress = sft_step / max(1, total_sft_steps - 1)
            lr = sft_cfg.min_lr + 0.5 * (sft_cfg.sft_lr - sft_cfg.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
            for pg in sft_optimizer.param_groups:
                pg["lr"] = lr

            sft_optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, use_cache=False)
            logits = out["logits"]
            loss = compute_routed_sft_loss(
                logits=logits,
                labels=labels,
                first_token_weight=sft_cfg.first_token_weight,
                label_smoothing=smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
            sft_optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            sft_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        sft_loss_history.append(avg_loss)
        if verbose and (
            (epoch + 1) % 3 == 0 or epoch == 0 or epoch + 1 == sft_cfg.sft_epochs
        ):
            print(
                f"  [SFT Epoch {epoch + 1:02d}/{sft_cfg.sft_epochs:02d}] "
                f"Routed SFT Loss: {avg_loss:.4f} | LR: {lr:.6f}"
            )

    # Freeze for mobile inference
    model.freeze_for_mobile_inference()

    metrics: Dict[str, Any] = {
        "pipeline": "common_crawl_pretrain_plus_sft",
        "curation_report": curation_report.to_dict(),
        "pretrain_metrics": pretrain_metrics,
        "pretrain_epochs": sft_cfg.pretrain_epochs,
        "sft_epochs": sft_cfg.sft_epochs,
        "sft_samples": len(sft_dataset),
        "sft_intent_groups": len(SFT_INTENT_GROUPS),
        "pretrain_initial_loss": pretrain_metrics.get("initial_loss", 0.0),
        "pretrain_final_loss": pretrain_metrics.get("final_loss", 0.0),
        "sft_initial_loss": sft_loss_history[0] if sft_loss_history else 0.0,
        "sft_final_loss": sft_loss_history[-1] if sft_loss_history else 0.0,
        "initial_loss": pretrain_metrics.get("initial_loss", 0.0),
        "final_loss": sft_loss_history[-1] if sft_loss_history else 0.0,
        "loss_history": list(pretrain_metrics.get("loss_history", [])) + sft_loss_history,
        "mobile_trainable": False,
    }
    if checkpoint_path is not None:
        save_hexon_checkpoint(
            model,
            tokenizer,
            checkpoint_path,
            metadata=metrics,
        )
    return model, tokenizer, metrics
