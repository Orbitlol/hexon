#!/usr/bin/env python3
"""
Build the hexon-1.5 supervised fine-tuning set.

Design notes (these matter, read before editing):

*   **Prompts match inference exactly.** Grounded examples are formatted with
    the same strings `HexonChromaRAG.build_rag_prompt` produces at runtime, so
    the model is trained on the exact shape of prompt it will see on device.
    Two templates are used, one for coding questions and one for everything
    else.

*   **Contexts are the real chunks.** Each grounded example retrieves the
    actual 200-300 word chunk that will sit in ChromaDB, rather than a
    paraphrase, so training and retrieval agree.

*   **Balanced by design.** The set is deliberately spread across grounded
    factual QA, coding, general chat and calibrated refusal. The refusal
    examples are capped on purpose: an earlier run over-trained on them and
    the model started refusing questions it could answer. Refusals should be
    the smallest coherent group, not a mood.

*   **Held-out protected.** `scripts/run_eval.py` re-checks this file against
    `eval/heldout_eval.json` before every run. Do not add eval questions here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexon.rag import chunk_text  # noqa: E402

OUT = ROOT / "data" / "sft" / "hexon_15_sft.jsonl"

CORPUS_FILES = [
    ROOT / "data" / "knowledge" / "hexon_core_science.jsonl",
    ROOT / "data" / "knowledge" / "hexon_core_tech.jsonl",
]

# ---------------------------------------------------------------------------
# The training set
# ---------------------------------------------------------------------------

# (question, answer, topic-hint or None for ungrounded, kind)
# kind: "fact" | "code" | "chat" | "refuse"
GROUNDABLE: List[Dict[str, str]] = [
    # --- science: physics -------------------------------------------------
    {"q": "What does Newton's first law state?", "a": "Newton's first law says an object stays at rest, or keeps moving in a straight line at constant speed, unless a net external force acts on it. This resistance to change in motion is called inertia.", "topic": "Newton's laws inertia force", "kind": "fact"},
    {"q": "Why does a heavy box need a bigger push than a light one?", "a": "Because net force equals mass times acceleration. To give a heavier mass the same acceleration you must apply a proportionally larger force.", "topic": "Newton second law force mass acceleration", "kind": "fact"},
    {"q": "What is the difference between speed and velocity?", "a": "Speed is how fast something moves, while velocity is speed in a particular direction. Velocity is a vector, so an object turning at constant speed is still changing its velocity.", "topic": "velocity speed direction motion", "kind": "fact"},
    {"q": "What is power measured in?", "a": "Power is measured in watts. One watt is one joule of energy transferred per second, so a 100 watt bulb converts 100 joules every second.", "topic": "power watts energy joules work", "kind": "fact"},
    {"q": "Why is the sky blue?", "a": "Because of Rayleigh scattering. Air molecules scatter short wavelength blue light far more strongly than longer red wavelengths, so the light reaching your eyes from across the sky has been scattered out of the sunlight beam.", "topic": "Rayleigh scattering sky blue light wavelength", "kind": "fact"},
    {"q": "What colour is at the short wavelength end of visible light?", "a": "Violet, at roughly 400 nanometres. Visible light runs from about 400 nanometres at the violet end to about 700 nanometres at the red end.", "topic": "visible light wavelength violet red spectrum", "kind": "fact"},
    # --- science: chemistry -----------------------------------------------
    {"q": "What defines which element an atom belongs to?", "a": "The number of protons in its nucleus, called the atomic number. Hydrogen has one proton, carbon has six, iron has 26 and oxygen has eight.", "topic": "atomic number protons element periodic table", "kind": "fact"},
    {"q": "Why is the symbol for iron Fe and not Ir?", "a": "Chemical symbols are often derived from Latin names, and iron's symbol comes from the Latin word ferrum. Similarly sodium is Na and gold is Au from aurum.", "topic": "chemical symbol iron Fe Latin ferrum", "kind": "fact"},
    {"q": "What is the formula for table salt and how does it form?", "a": "Table salt is sodium chloride, NaCl. It forms by ionic bonding: a sodium atom transfers an electron to a chlorine atom, and the resulting oppositely charged ions attract to build a cubic crystal lattice.", "topic": "sodium chloride NaCl ionic bonding salt", "kind": "fact"},
    {"q": "What does a catalyst do?", "a": "A catalyst speeds up a chemical reaction without being used up itself. It does not change what the products are, only how fast they form.", "topic": "catalyst reaction rate speed", "kind": "fact"},
    {"q": "What is the pH of pure water?", "a": "Pure water is neutral, so its pH is 7. Values below 7 are acidic and values above 7 are alkaline.", "topic": "pH scale neutral acid alkaline water", "kind": "fact"},
    # --- science: biology -------------------------------------------------
    {"q": "Why is the mitochondrion called the powerhouse of the cell?", "a": "Because mitochondria carry out aerobic respiration, releasing usable energy from glucose. Cells with high energy demands, such as muscle cells, contain many mitochondria.", "topic": "mitochondria powerhouse respiration energy cell", "kind": "fact"},
    {"q": "What do chloroplasts do?", "a": "Chloroplasts contain chlorophyll, which absorbs light energy for photosynthesis. They are found in plant cells and algae but not in animal cells.", "topic": "chloroplast chlorophyll photosynthesis plant cell", "kind": "fact"},
    {"q": "How many bones does an adult human have?", "a": "An adult human skeleton has 206 bones. Newborns start with around 270, but many fuse together as a person grows.", "topic": "adult skeleton 206 bones newborn fuse", "kind": "fact"},
    {"q": "Which part of blood carries oxygen?", "a": "Red blood cells, also called erythrocytes. They contain haemoglobin, an iron containing protein that binds oxygen in the lungs and releases it where oxygen is scarce.", "topic": "red blood cells erythrocytes haemoglobin oxygen", "kind": "fact"},
    {"q": "What do platelets do?", "a": "Platelets are small cell fragments that help blood clot at a wound, sealing it and preventing further blood loss.", "topic": "platelets clot blood wound", "kind": "fact"},
    {"q": "How many chambers does the human heart have?", "a": "The human heart has four chambers: two atria on top and two ventricles below. The right side pumps blood to the lungs and the left side pumps it around the body.", "topic": "heart four chambers atria ventricles", "kind": "fact"},
    {"q": "What does photosynthesis produce?", "a": "Photosynthesis uses light energy to convert carbon dioxide and water into glucose, releasing oxygen as a by-product. The plant then uses the glucose for respiration, starch storage or building cell walls.", "topic": "photosynthesis glucose oxygen carbon dioxide chlorophyll", "kind": "fact"},
    {"q": "Which mineral do plants need to build chlorophyll?", "a": "Magnesium. A shortage causes yellowing leaves. Plants also need nitrates to make proteins and phosphates for roots and flowers.", "topic": "magnesium chlorophyll nitrates plant minerals", "kind": "fact"},
    # --- science: earth ---------------------------------------------------
    {"q": "What is the most abundant gas in the atmosphere?", "a": "Nitrogen, making up roughly 78 percent of the atmosphere by volume. Oxygen is second at about 21 percent, and carbon dioxide is a small but climatically important fraction.", "topic": "atmosphere nitrogen 78 percent oxygen", "kind": "fact"},
    {"q": "Why does water boil below 100 Celsius on a mountain?", "a": "Because atmospheric pressure is lower at altitude. Water boils when its vapour pressure equals the surrounding pressure, so less pressure means a lower boiling temperature.", "topic": "boiling point altitude pressure water", "kind": "fact"},
    {"q": "What is the hardest natural substance?", "a": "Diamond, which scores 10 on the Mohs scale of hardness. It is pure carbon with every atom bonded to four others in a rigid lattice. Graphite is also pure carbon but is soft because of its layered structure.", "topic": "diamond hardest Mohs carbon graphite", "kind": "fact"},
    {"q": "How are sedimentary rocks formed?", "a": "Sedimentary rocks such as sandstone, limestone and shale form from particles or shells that are deposited, buried and cemented together over long periods.", "topic": "sedimentary rock sandstone limestone deposited", "kind": "fact"},
    # --- geography --------------------------------------------------------
    {"q": "What is the capital of Australia?", "a": "Canberra. It was purpose built as a compromise between the rival cities of Sydney and Melbourne, which is why it is not the largest city.", "topic": "Canberra capital Australia Sydney Melbourne", "kind": "fact"},
    {"q": "What is the capital of Canada?", "a": "Ottawa, which sits on the Ontario side of the border with Quebec. It is not Toronto or Montreal.", "topic": "Ottawa capital Canada Toronto Montreal", "kind": "fact"},
    {"q": "Which is the largest ocean?", "a": "The Pacific Ocean. It covers more area than all of the land on Earth combined and contains the Mariana Trench, the deepest known point in the seabed.", "topic": "Pacific largest ocean Mariana Trench", "kind": "fact"},
    {"q": "Which river carries the most water?", "a": "The Amazon in South America carries more water than any other river. Its length is close to the Nile's, and which is longest is genuinely disputed, so it is safest to call the Amazon the largest by volume.", "topic": "Amazon river volume Nile longest disputed", "kind": "fact"},
    {"q": "Which mountain range contains Everest?", "a": "The Himalayas, the highest range on Earth. Everest stands at 8849 metres on the border between Nepal and China's Tibet region.", "topic": "Himalayas Everest highest mountain range", "kind": "fact"},
    {"q": "What is the largest hot desert?", "a": "The Sahara in North Africa. Antarctica is technically the largest desert overall if you define desert by low precipitation rather than by heat.", "topic": "Sahara largest hot desert Antarctica", "kind": "fact"},
    {"q": "Which is the smallest country in the world?", "a": "Vatican City, an enclave inside Rome of less than half a square kilometre with a few hundred residents. It is the headquarters of the Roman Catholic Church.", "topic": "Vatican City smallest country enclave Rome", "kind": "fact"},
    # --- history ----------------------------------------------------------
    {"q": "When did the Berlin Wall fall?", "a": "The Berlin Wall was built in 1961 and opened in November 1989. The Soviet Union dissolved two years later in 1991, ending the Cold War.", "topic": "Berlin Wall 1961 opened 1989 Cold War", "kind": "fact"},
    {"q": "Who was the first person to walk on the Moon?", "a": "Neil Armstrong, on 20 July 1969 during the Apollo 11 mission. Buzz Aldrin followed him onto the surface while Michael Collins orbited above.", "topic": "Neil Armstrong first Moon Apollo 11 1969", "kind": "fact"},
    {"q": "When did the Titanic sink and why did so many die?", "a": "RMS Titanic struck an iceberg on 14 April 1912 and sank in the early hours of 15 April 1912. More than 1500 of about 2200 people aboard died, partly because the ship carried lifeboats for only about half of them.", "topic": "Titanic iceberg 1912 lifeboats sank", "kind": "fact"},
    {"q": "Who invented the World Wide Web?", "a": "Tim Berners-Lee, a British computer scientist at CERN, who proposed it in 1989 and built the first browser, server and web page by the end of 1990. He later founded the W3C.", "topic": "Tim Berners-Lee invented World Wide Web CERN", "kind": "fact"},
    {"q": "What ended the Second World War?", "a": "The war in Europe ended in May 1945 with Germany's surrender, and the war overall ended in September 1945 after Japan surrendered following the atomic bombings of Hiroshima and Nagasaki.", "topic": "Second World War ended 1945 surrender", "kind": "fact"},
    # --- tech -------------------------------------------------------------
    {"q": "What does RAM stand for and what does it do?", "a": "RAM stands for Random Access Memory. It is the computer's working memory, holding running programs and their data. It is volatile, so its contents are lost when power is removed.", "topic": "RAM Random Access Memory volatile working memory", "kind": "fact"},
    {"q": "What does CPU stand for?", "a": "CPU stands for Central Processing Unit. It is the component that fetches, decodes and executes instructions, billions of times per second.", "topic": "CPU Central Processing Unit executes instructions", "kind": "fact"},
    {"q": "What does HTTP stand for?", "a": "HTTP stands for HyperText Transfer Protocol. It is the request and response protocol a browser uses to ask a server for a page. HTTPS is the same protocol wrapped in encryption.", "topic": "HTTP HyperText Transfer Protocol request response", "kind": "fact"},
    {"q": "What does HTML stand for?", "a": "HTML stands for HyperText Markup Language. It structures a web page into headings, paragraphs, links, images and forms.", "topic": "HTML HyperText Markup Language web page", "kind": "fact"},
    {"q": "Who develops Android?", "a": "Android is an open source operating system developed by Google, built on the Linux kernel. It powers phones from Samsung, Xiaomi, OnePlus and many other manufacturers.", "topic": "Android Google developed Linux kernel open source", "kind": "fact"},
    {"q": "When was the first iPhone released?", "a": "The first iPhone was released in 2007. It ran Apple's iOS, which is closed source and runs only on iPhone and iPad.", "topic": "first iPhone released 2007 Apple iOS", "kind": "fact"},
    {"q": "What does a GPU do?", "a": "A graphics processing unit has thousands of simple cores that work in parallel. It was designed for drawing images but is now the main engine behind machine learning, because training neural networks is largely matrix arithmetic.", "topic": "GPU graphics parallel cores machine learning", "kind": "fact"},
    {"q": "What is the difference between RAM and storage?", "a": "RAM is fast working memory that is wiped when the power goes off, while storage such as an SSD or hard disk keeps files when the machine is switched off but is much slower. Programs are copied from storage into RAM to run.", "topic": "RAM storage volatile SSD persistent difference", "kind": "fact"},
    {"q": "What is an operating system kernel?", "a": "The kernel is the core of the operating system and runs with full hardware privileges. It schedules processes, manages virtual memory, provides the filesystem and supplies device drivers, while applications run with limited rights.", "topic": "kernel operating system privileges scheduling drivers", "kind": "fact"},
    {"q": "Why is quantisation used for mobile models?", "a": "Quantisation stores trained weights in fewer bits, so a model in four bit integers is roughly a quarter the size of the same model in sixteen bit floats, with only a small loss in quality. That makes it fit and run on a phone.", "topic": "quantisation fewer bits mobile model size", "kind": "fact"},
    # --- mobile -----------------------------------------------------------
    {"q": "What processor does the Poco M3 use?", "a": "The Xiaomi Poco M3 is powered by a Qualcomm Snapdragon 662 processor with an Adreno 610 GPU. It is a budget Android phone with a 6.53 inch display.", "topic": "Poco M3 Snapdragon 662 Adreno 610 processor", "kind": "fact"},
    {"q": "What is the battery capacity of the Poco M3?", "a": "The Poco M3 has a 6000 mAh battery. Capacity in milliamp hours is not the same as memory, which is measured in gigabytes, so quoting a battery in gigabytes would be wrong.", "topic": "Poco M3 battery 6000 mAh capacity", "kind": "fact"},
    {"q": "What GPU does the Poco M3 have?", "a": "The Poco M3 uses an Adreno 610 GPU, paired with the Qualcomm Snapdragon 662 processor.", "topic": "Poco M3 Adreno 610 GPU Snapdragon", "kind": "fact"},
    {"q": "What does mAh mean on a phone battery?", "a": "mAh means milliamp hours, a unit of charge capacity. A 6000 mAh battery can in principle deliver 6000 milliamps for one hour. It measures capacity, not memory, so it must not be confused with gigabytes of RAM.", "topic": "mAh milliamp hours battery capacity definition", "kind": "fact"},
    # --- math -------------------------------------------------------------
    {"q": "How do you work out a percentage of a number?", "a": "Multiply the number by the percentage and divide by 100. For example, 15 percent of 200 is 15 times 200 divided by 100, which is 30.", "topic": "percentage multiply divide 100 fifteen percent", "kind": "fact"},
    {"q": "What is a prime number?", "a": "A prime number is a whole number greater than 1 whose only positive divisors are 1 and itself. So 2, 3, 5, 7, 11 and 13 are prime, while 1 is not and 6 is not because it divides by 2 and 3.", "topic": "prime number greater than one divisors", "kind": "fact"},
    {"q": "What do the angles inside a triangle add up to?", "a": "The three interior angles of any triangle always add up to 180 degrees, which is why a triangle can never contain more than one right angle.", "topic": "triangle interior angles 180 degrees", "kind": "fact"},
    {"q": "What is the decimal number 10 in binary?", "a": "The decimal number 10 is written 1010 in binary. Each position is a power of two, so 1010 means one eight, zero fours, one two and zero units, which is ten.", "topic": "decimal 10 binary 1010 convert", "kind": "fact"},
    # --- general ----------------------------------------------------------
    {"q": "What is the currency of Japan?", "a": "The yen, issued by the Bank of Japan. It should not be confused with the yuan of China or the won of South Korea.", "topic": "yen currency Japan Bank", "kind": "fact"},
    {"q": "Who wrote The Hobbit?", "a": "J. R. R. Tolkien, who published it in 1937. He was an Oxford professor of Anglo Saxon who built entire languages for his world.", "topic": "Hobbit Tolkien author published 1937", "kind": "fact"},
]

CODING: List[Dict[str, str]] = [
    {"q": "Write a Python function that checks whether a string is a palindrome.",
     "a": "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = \"\".join(ch.lower() for ch in s if ch.isalnum())\n    return cleaned == cleaned[::-1]\n```\nNormalise first by lowercasing and keeping only alphanumeric characters, then compare the string with its reverse obtained by slicing with a step of minus one.",
     "topic": "palindrome string reverse slice Python", "kind": "code"},
    {"q": "Write a Python function that computes the factorial of a number.",
     "a": "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```\nThe iterative version avoids the recursion limit. Factorial of 0 is 1.",
     "topic": "factorial loop Python recursive iterative", "kind": "code"},
    {"q": "Write a Python function that counts how often each word appears in a string.",
     "a": "```python\nfrom collections import Counter\n\ndef word_frequency(text: str) -> dict:\n    return dict(Counter(text.lower().split()))\n```\nSplit the text into words, then count each one. If you want it without the standard library, build a dict and increment the count for each word.",
     "topic": "word frequency count dictionary Counter text", "kind": "code"},
    {"q": "Write a Python function that merges two already sorted lists into one sorted list.",
     "a": "```python\ndef merge_sorted(a: list, b: list) -> list:\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n```\nTwo pointers walk both lists, always taking the smaller front item. This is the merge step of merge sort and runs in linear time.",
     "topic": "merge two sorted lists two pointers linear", "kind": "code"},
    {"q": "Write a Python function that removes duplicates from a list but keeps the original order.",
     "a": "```python\ndef unique_in_order(items: list) -> list:\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n```\nA set gives constant time membership tests but loses order, so it is used only to track what has been seen while a list preserves the order.",
     "topic": "remove duplicates preserve order set seen", "kind": "code"},
    {"q": "Write a Python function that returns the greatest common divisor of two numbers.",
     "a": "```python\ndef gcd(a: int, b: int) -> int:\n    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a\n```\nThis is Euclid's algorithm: repeatedly replace the larger number with its remainder on division by the smaller until one of them is zero.",
     "topic": "greatest common divisor Euclid remainder modulo", "kind": "code"},
    {"q": "Write a Python function that flattens a list of lists into one list.",
     "a": "```python\ndef flatten(nested: list) -> list:\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out\n```\nRecursion handles arbitrarily deep nesting. For a single level, a list comprehension over the sublists is shorter.",
     "topic": "flatten nested list recursion extend", "kind": "code"},
    {"q": "Write a Python class for a stack with push and pop.",
     "a": "```python\nclass Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, item):\n        self._items.append(item)\n\n    def pop(self):\n        if not self._items:\n            raise IndexError(\"pop from empty stack\")\n        return self._items.pop()\n\n    def __len__(self):\n        return len(self._items)\n```\nAppending and popping from the end of a list are both constant time, so a stack is last in first out.",
     "topic": "stack class push pop last in first out", "kind": "code"},
    {"q": "Write a Python function that plays FizzBuzz for a given number.",
     "a": "```python\ndef fizzbuzz(n: int) -> str:\n    if n % 15 == 0:\n        return \"FizzBuzz\"\n    if n % 3 == 0:\n        return \"Fizz\"\n    if n % 5 == 0:\n        return \"Buzz\"\n    return str(n)\n```\nCheck the combined multiple of 15 first, otherwise the factor of 3 branch catches those numbers and FizzBuzz never prints.",
     "topic": "FizzBuzz multiples of three five fifteen", "kind": "code"},
    {"q": "Write a JavaScript function that reverses a string.",
     "a": "```javascript\nfunction reverseString(s) {\n  return s.split('').reverse().join('');\n}\n```\nStrings have no reverse method of their own, so split into an array of characters, reverse the array and join it back together.",
     "topic": "JavaScript reverse string split join", "kind": "code"},
    {"q": "Write a Python function that counts the vowels in a string.",
     "a": "```python\ndef count_vowels(s: str) -> int:\n    return sum(1 for ch in s.lower() if ch in \"aeiou\")\n```\nLowercase first so the comparison is case insensitive, then count characters that are members of the vowel set.",
     "topic": "count vowels string membership set", "kind": "code"},
    {"q": "Write a Python function that finds the largest number in a list.",
     "a": "```python\ndef largest(nums: list):\n    if not nums:\n        raise ValueError(\"empty list\")\n    best = nums[0]\n    for n in nums[1:]:\n        if n > best:\n            best = n\n    return best\n```\nIn real code you would just call max, but the loop that tracks the current best is the pattern worth knowing.",
     "topic": "largest element list max loop track", "kind": "code"},
    {"q": "Write a Python function that checks whether two strings are anagrams.",
     "a": "```python\nfrom collections import Counter\n\ndef are_anagrams(a: str, b: str) -> bool:\n    return Counter(a.lower()) == Counter(b.lower())\n```\nTwo strings are anagrams when they contain the same characters in the same quantities, so comparing sorted characters or two character counts both work.",
     "topic": "anagrams strings same characters count sorted", "kind": "code"},
    {"q": "How does binary search work?",
     "a": "Binary search requires sorted data. Look at the middle element; if it is not the target, discard the half that cannot contain it and repeat, halving the search space each step. That gives logarithmic time instead of the linear time of checking every item.",
     "topic": "binary search sorted middle halve logarithmic", "kind": "code"},
    {"q": "How do you reverse a linked list?",
     "a": "Walk the list with three pointers. Keep prev, curr and next_node; at each step store curr.next in next_node, point curr.next at prev, then advance prev to curr and curr to next_node. Return prev, which ends up as the new head.",
     "topic": "reverse linked list pointers prev curr next", "kind": "code"},
    {"q": "What is the difference between a list and a tuple in Python?",
     "a": "A list is ordered and mutable, written with square brackets, so you can change it after creation. A tuple is ordered and immutable, written with parentheses, so it cannot be modified and can be used as a dictionary key.",
     "topic": "list tuple mutable immutable Python", "kind": "code"},
    {"q": "Why is a mutable default argument a bug in Python?",
     "a": "The default object is created once when the function is defined, not on each call, so a mutable default such as a list is shared between calls and accumulates changes. The safe idiom is to default to None and create a fresh list inside the body.",
     "topic": "mutable default argument None idiom bug", "kind": "code"},
    {"q": "What does a decorator do in Python?",
     "a": "A decorator wraps a function to add behaviour before or after it runs, without changing the function's own code. It works because functions are first class objects that can be passed as arguments and returned.",
     "topic": "decorator wraps function first class", "kind": "code"},
    {"q": "What is the difference between let, const and var in JavaScript?",
     "a": "let and const are block scoped, while var is function scoped. let allows reassignment and const does not, though a const object can still have its properties changed.",
     "topic": "JavaScript let const var block scoped", "kind": "code"},
    {"q": "How do you wait for an asynchronous call in JavaScript?",
     "a": "Use a promise with async and await. Mark the containing function async and await the promise inside it, which reads like synchronous code while still leaving the browser responsive.",
     "topic": "JavaScript async await promise asynchronous", "kind": "code"},
    {"q": "What does a list comprehension do?",
     "a": "A list comprehension builds a new list by applying an expression to each item of a source sequence, optionally filtering with a condition. It is usually clearer and faster than an explicit loop with append.",
     "topic": "list comprehension transform filter Python", "kind": "code"},
    {"q": "What is the time complexity of merge sort?",
     "a": "Merge sort runs in O of n log n time. It splits the list in half, sorts each half recursively, then merges the two sorted halves in linear time, and the split happens log n levels deep.",
     "topic": "merge sort complexity n log n recursive", "kind": "code"},
    {"q": "What is JSON used for?",
     "a": "JSON, JavaScript Object Notation, is the standard text format for exchanging structured data between a client and a server. It represents objects, arrays, strings, numbers, booleans and null.",
     "topic": "JSON data exchange format structured", "kind": "code"},
    {"q": "What is an index in a database?",
     "a": "An index is a sorted copy of a column that makes lookups far faster. It costs extra storage and slows down writes, because the index has to be updated whenever the data changes.",
     "topic": "database index lookup faster writes", "kind": "code"},
    {"q": "What does a list's map method do in JavaScript?",
     "a": "map returns a new array containing the result of applying a function to each element. It does not change the original array, unlike forEach which simply iterates.",
     "topic": "JavaScript map array transform new array", "kind": "code"},
    {"q": "What is recursion and what stops it?",
     "a": "Recursion solves a problem in terms of smaller instances of itself. Every recursive function needs a base case, a condition under which it returns directly instead of calling itself, otherwise it never terminates.",
     "topic": "recursion base case terminates", "kind": "code"},
    {"q": "How do you handle a missing key in a Python dictionary?",
     "a": "Use the get method with a default, or check membership with the in operator, or catch KeyError. The collections module also provides defaultdict, which supplies a default value for missing keys automatically.",
     "topic": "dictionary missing key get default KeyError", "kind": "code"},
    {"q": "What is UTF-8?",
     "a": "UTF-8 is the dominant encoding of Unicode on the web. It is backwards compatible with ASCII and compact for English text, while still being able to represent every character in every writing system.",
     "topic": "UTF-8 Unicode encoding ASCII", "kind": "code"},
    {"q": "What does the DOM do in a browser?",
     "a": "The Document Object Model represents the page as a tree of nodes that scripts can read and change. You use querySelector to find elements and addEventListener to respond to clicks and other events.",
     "topic": "DOM Document Object Model querySelector events", "kind": "code"},
    {"q": "What is the difference between HTTP and HTTPS?",
     "a": "HTTPS is HTTP wrapped in TLS encryption, so traffic cannot be read or tampered with in transit. HTTPS is now the default for the web and browsers warn users when a site does not use it.",
     "topic": "HTTPS HTTP TLS encrypted difference", "kind": "code"},
    {"q": "What is a queue as opposed to a stack?",
     "a": "A stack is last in first out, so the most recently added item is removed first. A queue is first in first out, so the item that has been waiting longest is removed first.",
     "topic": "queue stack first in first out last in", "kind": "code"},
    {"q": "Why would you use a set in Python?",
     "a": "A set stores unique values and gives constant time membership tests, which makes it ideal for deduplication and for tracking what has already been seen. It is unordered, so it cannot preserve insertion order.",
     "topic": "set unique membership constant time unordered", "kind": "code"},
    {"q": "What is Big O notation?",
     "a": "Big O notation describes how an algorithm's cost grows as the input grows, ignoring constant factors. O of n means the cost grows linearly, O of log n means it grows very slowly, and O of n squared means it grows quadratically.",
     "topic": "Big O notation complexity growth", "kind": "code"},
]

CHAT: List[Dict[str, str]] = [
    {"q": "Who are you?", "a": "I am Hexon, a compact language model built to run locally on your phone. I answer questions, help with code and work offline, and I never train on your device.", "kind": "chat"},
    {"q": "Do you train on my phone?", "a": "No. Training happens offline on servers. What runs on your phone is a frozen, read-only model file that is strictly for inference.", "kind": "chat"},
    {"q": "Are you connected to the internet?", "a": "No. I run entirely on your device and answer from my own weights plus the local ChromaDB knowledge base, so nothing you ask me leaves the phone.", "kind": "chat"},
    {"q": "What can you help me with?", "a": "Explaining concepts, answering general knowledge questions, writing and debugging code, working through maths, and drafting text. If something is not in my knowledge base I will say so rather than invent an answer.", "kind": "chat"},
    {"q": "Are you sure about that?", "a": "Not unless the evidence supports it. Let me know which part you would like me to double check and I will tell you what my sources actually say.", "kind": "chat"},
    {"q": "Can you write code for me?", "a": "Yes. Tell me the language and what it should do, and I will write a clean, complete solution with a short explanation of how it works.", "kind": "chat"},
    {"q": "Explain it more simply.", "a": "Of course. In plain terms, think of it as one step producing the next, where each step only depends on what came before, which is why the result is easy to follow.", "kind": "chat"},
    {"q": "Thanks, that helped.", "a": "You are welcome. Ask me anything else whenever you like.", "kind": "chat"},
    {"q": "Good morning", "a": "Good morning. What would you like to work on today?", "kind": "chat"},
    {"q": "Tell me a joke.", "a": "Why do programmers prefer dark mode? Because light attracts bugs.", "kind": "chat"},
    {"q": "How do I improve my Python?", "a": "Write small programs every day, read code by experienced authors, and learn the standard library well. Reading error messages carefully will teach you more than any tutorial.", "kind": "chat"},
    {"q": "What should I learn first, Python or JavaScript?", "a": "Python, if you have no other constraint. Its syntax is closer to plain English and it lets you focus on programming ideas rather than punctuation. JavaScript makes sense once you want to build things in a browser.", "kind": "chat"},
    {"q": "Summarise what we discussed.", "a": "We covered what I am, how I run locally on your device, and the kinds of questions and code tasks I can help with. Tell me if you would like to go deeper on any of it.", "kind": "chat"},
    {"q": "Be concise from now on.", "a": "Understood. I will keep answers short and to the point.", "kind": "chat"},
    {"q": "What is ChromaDB?", "a": "It is the local vector database I retrieve from. My knowledge base is stored as chunks of a few hundred words, and when you ask something I pull the closest chunks and ground my answer in them.", "kind": "chat"},
    {"q": "Why did you give a different answer last time?", "a": "Either I retrieved different context, or I made a mistake. If you tell me what the earlier answer was I can tell you which one I now think is right and why.", "kind": "chat"},
    {"q": "How large are you?", "a": "Small enough to run on a phone. My weights sit in a single quantised file of a couple of hundred megabytes, which is why I can run without a server.", "kind": "chat"},
    {"q": "Do you remember our earlier conversation?", "a": "Only within this session, and only the last few turns. I do not keep a permanent memory of what you have asked me.", "kind": "chat"},
    {"q": "Is my data private?", "a": "Yes. Inference happens on your device, I do not send your questions anywhere, and I do not train on what you type.", "kind": "chat"},
    {"q": "What happens if you do not know something?", "a": "I say so. Guessing would be worse than admitting the gap, so I will tell you what is missing and suggest how to find the answer.", "kind": "chat"},
]

REFUSALS: List[Dict[str, str]] = [
    {"q": "What is the exact population of my city right now?", "a": "I do not have live population data for your city, so I will not guess a number. A recent census or your city's statistics office would give you a figure I can trust.", "kind": "refuse"},
    {"q": "What will the weather be next Tuesday?", "a": "I cannot check live weather because I have no internet connection and no forecast data in my knowledge base. Your phone's weather app will have an actual forecast.", "kind": "refuse"},
    {"q": "Who won the match yesterday?", "a": "I do not have current sports results in my knowledge base, so I cannot tell you. A sports site or app would have the score.", "kind": "refuse"},
    {"q": "What is the current price of Bitcoin?", "a": "I have no live market data, so any figure I gave would be invented. Check an exchange or a finance app for the current price.", "kind": "refuse"},
    {"q": "What is my friend's phone number?", "a": "I do not have access to your contacts and would not look them up even if I did. Your phone's contacts app is the right place.", "kind": "refuse"},
    {"q": "Tell me something that is definitely going to happen next year.", "a": "I cannot predict the future, and I would rather say so than invent a confident forecast. I can tell you what has been true historically if that helps.", "kind": "refuse"},
    {"q": "What is the meaning of life?", "a": "That is a philosophical question rather than a factual one, so there is no single answer in my knowledge base. Different traditions answer it very differently, and I can outline the main ones if you would like.", "kind": "refuse"},
    {"q": "Who invented the word floccinaucinihilipilification?", "a": "I do not have a reliable source for that in my knowledge base, so I will not attribute it to anyone. It is a real English word meaning the habit of estimating something as worthless.", "kind": "refuse"},
    {"q": "How many people are in this room?", "a": "I cannot see or sense anything, so I have no way to know that. If you tell me, I can work with the number.", "kind": "refuse"},
    {"q": "What did I ask you three conversations ago?", "a": "I do not keep a record across conversations, so I cannot know what was asked before this session. I only have the current session in front of me.", "kind": "refuse"},
]

MULTITURN: List[List[Dict[str, str]]] = [
    [
        {"q": "What is the capital of France?", "a": "Paris. It has been the French capital since the year 508 and is also the country's largest city."},
        {"q": "And Germany?", "a": "Berlin. It became the capital of a reunified Germany in 1990, after the country was divided following the Second World War."},
    ],
    [
        {"q": "Explain what a variable is.", "a": "A variable is a named storage location that holds a value. You can read it, and in most languages you can change what it holds while the program runs."},
        {"q": "Now explain what a function is.", "a": "A function is a named, reusable block of code that takes inputs, called parameters, and returns an output. It lets you write logic once and call it from many places."},
    ],
    [
        {"q": "How do I make a cup of tea?", "a": "Boil water, put a tea bag in a mug, pour the water over it and leave it to steep for three to five minutes. Remove the bag and add milk or sugar if you like."},
        {"q": "And coffee?", "a": "For instant coffee, boil water, add a spoon of instant coffee to a mug and pour the water over it, then stir. For filter coffee, pour hot water slowly over ground coffee in a filter and let it drip through."},
    ],
    [
        {"q": "What is a list in Python?", "a": "A list is an ordered, mutable collection written with square brackets. You can add, remove and change its items after creating it."},
        {"q": "What about a dictionary?", "a": "A dictionary is an unordered mapping from keys to values, written with curly braces. You look values up by their key rather than by their position."},
        {"q": "When should I use each?", "a": "Use a list when order matters or you need to iterate over a sequence. Use a dictionary when you need to look something up by a meaningful label, such as a name or an id."},
    ],
    [
        {"q": "What is the hardest natural substance?", "a": "Diamond, which scores 10 on the Mohs scale of hardness."},
        {"q": "Why is it so hard?", "a": "Because it is pure carbon with every atom bonded to four neighbours in a rigid three dimensional lattice, so there are no weak planes to slip along."},
    ],
    [
        {"q": "Who wrote The Hobbit?", "a": "J. R. R. Tolkien, who published it in 1937."},
        {"q": "Did he write anything else?", "a": "Yes, most notably The Lord of the Rings, a three volume sequel, and The Silmarillion, which was published after his death. He was an Oxford professor of Anglo Saxon."},
    ],
    [
        {"q": "What is RAM?", "a": "Random Access Memory, the computer's fast working memory that holds running programs and their data."},
        {"q": "How is that different from a hard drive?", "a": "RAM is volatile and loses everything when the power goes off, while a drive keeps its contents but is far slower. Programs are copied from the drive into RAM so the processor can work with them at full speed."},
    ],
    [
        {"q": "How do I reverse a string in JavaScript?", "a": "Split it into an array of characters, reverse the array and join it back, because strings have no reverse method of their own."},
        {"q": "And in Python?", "a": "Slice with a step of minus one, which returns the reversed string directly. Python strings support slicing, so no conversion to a list is needed."},
    ],
]


# ---------------------------------------------------------------------------
# Corpus loading and context matching
# ---------------------------------------------------------------------------


def load_chunks() -> List[Dict[str, Any]]:
    """Chunk every authored article exactly the way ChromaDB will."""
    chunks: List[Dict[str, Any]] = []
    for path in CORPUS_FILES:
        if not path.exists():
            print(f"  ! missing corpus file {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                pieces = chunk_text(doc["text"], min_words=200, max_words=300, overlap_words=35)
                for i, piece in enumerate(pieces):
                    chunks.append(
                        {
                            "text": piece,
                            "title": doc.get("title", ""),
                            "category": doc.get("category", ""),
                            "source": doc.get("source", ""),
                            "chunk_ref": f"{doc.get('id', 'doc')}#{i}",
                        }
                    )
    return chunks


def _stems(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def pick_chunk(chunks: Sequence[Dict[str, Any]], query: str, topic: str) -> Optional[Dict[str, Any]]:
    """Pick the corpus chunk that best covers a question, by keyword overlap."""
    wanted = _stems(f"{topic} {query}")
    if not wanted:
        return None
    best, best_score = None, 0.0
    for c in chunks:
        have = _stems(c["text"])
        score = len(wanted & have) / max(1, len(wanted))
        # light preference for the chunk that also matches rarer topic words
        score += 0.15 * (len(_stems(topic) & have) / max(1, len(_stems(topic))))
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.35 else best


def build_prompt(question: str, context: Optional[str], coding: bool) -> str:
    """Reproduce `HexonChromaRAG.build_rag_prompt` byte for byte."""
    if not context:
        return question
    ctx_block = "Context from ChromaDB:\n" + f"- {context}"
    if coding:
        return (
            f"{ctx_block}\n\n"
            f"Using the ChromaDB reference above, write a clean, accurate code solution "
            f"and concise explanation for:\n"
            f"{question}"
        )
    return (
        f"{ctx_block}\n\n"
        f"Based on the context above, answer the following question clearly and concisely:\n"
        f"{question}"
    )


def main() -> int:
    chunks = load_chunks()
    print(f"loaded {len(chunks)} corpus chunks for grounding")

    records: List[Dict[str, Any]] = []
    unmatched: List[str] = []

    def add(prompt: str, answer: str, category: str, **extra) -> None:
        rec = {
            "category": category,
            "user": prompt,
            "assistant": answer,
            "full": (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>\n"
            ),
        }
        rec.update(extra)
        records.append(rec)

    for ex in GROUNDABLE + CODING:
        coding = ex["kind"] == "code"
        chunk = pick_chunk(chunks, ex["q"], ex.get("topic", ""))
        if chunk is None:
            unmatched.append(ex["q"])
            context = None
        else:
            context = chunk["text"]
        add(
            build_prompt(ex["q"], context, coding),
            ex["a"],
            f"grounded_{ex['kind']}" if context else f"plain_{ex['kind']}",
            chunk_ref=chunk["chunk_ref"] if chunk else None,
            chunk_title=chunk["title"] if chunk else None,
            grounded=bool(context),
        )

    # Refusals are grounded with an intentionally OFF-TOPIC chunk, so the
    # model learns to decline when the retrieved context does not answer the
    # question. This is the behaviour that reduces fabrication.
    refusal_chunk = pick_chunk(chunks, "operating system kernel memory", "operating system kernel")
    for ex in REFUSALS:
        add(
            build_prompt(ex["q"], refusal_chunk["text"] if refusal_chunk else None, coding=False),
            ex["a"],
            "grounded_refuse",
            chunk_ref=refusal_chunk["chunk_ref"] if refusal_chunk else None,
            grounded=bool(refusal_chunk),
        )

    for ex in CHAT:
        add(ex["q"], ex["a"], "plain_chat", grounded=False)

    # Multi-turn: build the full conversation so the model sees distinct,
    # non-repeating assistant turns inside one context window.
    for seq in MULTITURN:
        turns = []
        for t in seq:
            turns.append(f"<|im_start|>user\n{t['q']}<|im_end|>\n")
            turns.append(f"<|im_start|>assistant\n{t['a']}<|im_end|>\n")
        records.append(
            {
                "category": "multiturn",
                "turns": len(seq),
                "turns_text": [{"q": t["q"], "a": t["a"]} for t in seq],
                "user": seq[-1]["q"],
                "assistant": seq[-1]["a"],
                "full": "".join(turns),
                "grounded": False,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: Dict[str, int] = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    print(f"wrote {len(records)} training records -> {OUT.relative_to(ROOT)}")
    print(f"  by category: {dict(sorted(counts.items()))}")
    grounded = sum(1 for r in records if r.get("grounded"))
    print(f"  grounded with real ChromaDB chunks: {grounded}/{len(records)}")
    if unmatched:
        print(f"  WARNING no context matched for: {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
