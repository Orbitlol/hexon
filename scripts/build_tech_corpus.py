#!/usr/bin/env python3
"""
Build the hand-authored Hexon technology, mobile and programming corpus.

Companion to `build_core_corpus.py`. Covers computing fundamentals, the
internet, mobile hardware, Python and JavaScript programming, algorithms and
machine learning. Articles are written at 400-800 words so the ChromaDB
chunker splits them into clean 200-300 word pieces; `main()` verifies that
every emitted chunk lands inside the 200-300 word schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "knowledge" / "hexon_core_tech.jsonl"

ARTICLES = [
    {
        "id": "core-tech-001",
        "title": "How Computers Work: CPU, Memory and Storage",
        "category": "tech",
        "text": """Every general purpose computer is built from the same small set of parts. The central processing unit, or CPU, is the part that actually executes instructions. It fetches an instruction from memory, decodes it, executes it and then moves to the next one, billions of times per second. A CPU contains several cores, each of which can run its own stream of instructions, and each core contains arithmetic units for maths and logic plus a small amount of extremely fast on chip memory called cache. RAM stands for Random Access Memory. It is the computer's working memory: it holds the programs that are currently running and the data they are working on. RAM is volatile, meaning everything in it is lost when the power goes off, but it is fast enough to keep up with the processor. Storage, by contrast, is persistent. A solid state drive or a hard disk keeps your files, applications and the operating system safely when the machine is switched off, but is far slower than RAM. The motherboard connects every component together with buses that carry data between them. The power supply converts mains electricity into the low voltages the electronics need, and cooling removes the heat that all this switching produces. When you open an application, the computer copies it from storage into RAM, and the CPU then executes it from there. This hierarchy exists because of a basic trade off: the faster a memory technology is, the more expensive it is per gigabyte and the smaller it has to be, so machines use a small amount of very fast memory close to the processor and large amounts of slow, cheap storage further away. A GPU, or graphics processing unit, is a specialised processor with thousands of simple cores that work in parallel. It was designed to draw images but is now also the main engine behind machine learning, because training neural networks is mostly the same kind of matrix arithmetic that graphics requires.""",
    },
    {
        "id": "core-tech-002",
        "title": "Operating Systems and Mobile Platforms",
        "category": "tech",
        "text": """An operating system is the software layer that sits between applications and the hardware. Its job is to hide the messy details of disks, screens, network cards and processors behind clean abstractions, and to share those resources safely between many running programs. The kernel is the core of the operating system and runs with full hardware privileges; everything else runs as a process with limited rights. The operating system schedules which process gets to use each CPU core and for how long, gives every process the illusion of having its own private memory through virtual memory, provides a filesystem so programs can read and write named files instead of raw disk blocks, and supplies device drivers so that a single program can talk to any printer, camera or keyboard. Desktop operating systems include Microsoft Windows, Apple macOS and the many distributions of Linux such as Ubuntu and Fedora. On mobile there are two dominant platforms. Android is an open source operating system developed by Google, built on top of the Linux kernel, and it powers phones from Samsung, Xiaomi, OnePlus, Motorola and many other manufacturers. Apple's iOS is closed source and runs only on the iPhone and iPad. Because Android is based on Linux it inherits a familiar process and permission model, and Android apps are usually written in Kotlin or Java and distributed through the Google Play Store, while iOS apps are written in Swift or Objective C and distributed through the App Store. A mobile operating system has extra responsibilities a desktop one does not: it must manage radios for cellular, Wi-Fi and Bluetooth, track and limit battery use, suspend background apps aggressively so the battery lasts a full day, and provide biometric authentication such as fingerprint and face unlock.""",
    },
    {
        "id": "core-tech-003",
        "title": "The Internet and the World Wide Web",
        "category": "tech",
        "text": """The Internet and the World Wide Web are related but not the same thing. The Internet is the global network of physical cables, fibre, routers and radio links that carries data between computers. The Web is one application that runs on top of it, a system of interlinked documents identified by addresses and fetched over a protocol. The Web was invented by Tim Berners-Lee, a British computer scientist working at CERN in Switzerland, who proposed it in 1989 and built the first web browser, web server and web page by the end of 1990. He went on to found the W3C, the body that still stewards web standards. Web addresses are called URLs, and they begin with a scheme such as http or https. HTTP stands for HyperText Transfer Protocol and is the request and response protocol a browser uses to ask a server for a page and receive it back. HTTPS is the same protocol wrapped in TLS encryption, and it is now the default. Every response carries a status code: 200 means success, 404 means not found and 500 means the server failed. HTML stands for HyperText Markup Language and is the language used to structure a web page into headings, paragraphs, links, images and forms. CSS controls how the page looks, and JavaScript is the programming language that runs in the browser to make pages interactive. A DNS server translates a human readable name such as example.com into the numeric IP address that routers actually need. Underneath, data is chopped into packets that are routed independently across many hops and reassembled at the destination.""",
    },
    {
        "id": "core-tech-004",
        "title": "Data Representation: Binary, Text and Numbers",
        "category": "tech",
        "text": """Computers store and process everything as binary, that is, as sequences of the two digits 0 and 1. A single binary digit is a bit, and eight bits make a byte, which can represent 256 distinct values. Binary is used because it maps cleanly onto physical hardware: a transistor that is off represents 0 and one that is on represents 1, and that is far easier to build reliably than hardware with ten voltage levels. To convert a decimal number to binary you repeatedly divide by two and record the remainders, reading them bottom to top. Working through the decimal number 10: ten divided by two is five remainder zero, five divided by two is two remainder one, two divided by two is one remainder zero, and one divided by two is zero remainder one, so reading the remainders upwards gives 1010. Therefore the decimal number 10 is written 1010 in binary. Each position in a binary number represents a power of two, so 1010 means one eight, zero fours, one two and zero units, which is eight plus two, or ten. Text is stored by giving every character a number and then storing those numbers. ASCII was the early standard using seven bits, and Unicode has superseded it, assigning a unique code point to essentially every character in every writing system in the world. UTF-8 is the dominant encoding of Unicode on the web because it is backwards compatible with ASCII and compact for English text. Images, audio and video are all stored the same way, as numbers that follow a particular format such as PNG, MP3 or MP4.""",
    },
    {
        "id": "core-tech-005",
        "title": "Python Fundamentals: Values, Collections and Control Flow",
        "category": "coding",
        "text": """Python is a high level, dynamically typed, interpreted language that emphasises readable code. Values have types rather than variables, so the same name can hold an integer, then later a string. The core numeric types are int for whole numbers of unlimited size and float for double precision decimals. Text is held in str objects, which are immutable sequences of Unicode characters and support slicing, so s reversed is written s with a step of minus one. Booleans are True and False, and None represents the absence of a value. The workhorse collections are the list, an ordered mutable sequence written with square brackets; the tuple, an ordered immutable sequence written with parentheses; the dict, an unordered mapping from keys to values written with curly braces; and the set, an unordered collection of unique values. Control flow uses if, elif and else for branching, and for and while for looping, with break to leave a loop early and continue to skip to the next iteration. Python uses indentation rather than braces to delimit blocks, which is why consistent indentation is mandatory. Truthiness means that empty collections, zero and None all behave as false in a condition. A list comprehension such as building a new list by applying an expression to each item of a source sequence is the idiomatic way to transform data, and it is usually clearer and faster than an explicit loop with append. Strings offer methods such as upper, lower, strip, split and join, and the in operator tests membership in any collection.""",
    },
    {
        "id": "core-tech-006",
        "title": "Python Functions, Arguments and Returns",
        "category": "coding",
        "text": """A function is a named, reusable block of code that takes inputs, called arguments or parameters, and produces an output with the return statement. Functions are defined with the def keyword followed by the name, a parameter list in parentheses and a colon, with the body indented underneath. Calling the function runs its body. If a function reaches the end without executing return, or executes a bare return, it returns None. Returning early is a common way to handle a special case before the main logic. Parameters can be positional, or passed by keyword using the name equals value form, which makes call sites self documenting. Default parameter values let a caller omit an argument, but mutable defaults such as a list are a classic bug because the default object is created once and shared between calls, so the safe idiom is to default to None and create a fresh list inside. A function can accept a variable number of positional arguments with a starred parameter and a variable number of keyword arguments with a double starred parameter. Python functions are first class objects, so they can be stored in variables, passed as arguments to other functions and returned from them, which is what makes decorators and callbacks work. A lambda expression defines a small anonymous function in a single expression, useful as a sort key. Names defined inside a function are local to it; assigning to a name that also exists outside creates a new local unless the name is declared global or nonlocal. Docstrings, a string literal as the first statement, document what the function does.""",
    },
    {
        "id": "core-tech-007",
        "title": "Python Strings, Lists, Dictionaries and Common Tasks",
        "category": "coding",
        "text": """Many everyday Python tasks come down to manipulating strings, lists and dictionaries. Testing whether a word reads the same forwards and backwards is a palindrome check, and the shortest way to do it is to compare the string with its reverse, obtained by slicing with a step of minus one; for real input you usually normalise first by lowercasing and stripping out anything that is not alphanumeric. Counting how often each word occurs in a piece of text is a dictionary task: split the text into words, then for each word either add a new key with a count of one or increment the existing count. The collections module provides a Counter that does exactly this. Removing duplicate entries from a list while keeping the original order is a common interview question, and the clean solution keeps a set of items already seen and appends each item to the result only the first time it appears, because a set gives constant time membership tests but loses order. Counting vowels in a string means iterating the characters and testing membership in a vowel set. Reversing or sorting a list is available as a method, and sorted returns a new list while the sort method sorts in place. Flattening a nested list of lists can be done with a loop that extends the result when it finds a list, or recursively for deeper nesting. Finding the largest value in a list is what the built in max function does, though an explicit loop that tracks the current best is the pattern worth knowing. Checking whether two strings are anagrams means verifying they contain the same characters in the same quantities, which is easiest by comparing sorted characters or by comparing two character counts.""",
    },
    {
        "id": "core-tech-008",
        "title": "Python Classes and Object Oriented Programming",
        "category": "coding",
        "text": """A class bundles data and the functions that operate on it into one object. Classes are declared with the class keyword, and calling the class constructs an instance. The initialisation method runs automatically when an object is created and is where attributes are set up; the first parameter of every method is the instance itself, called self by convention. Attributes store state and methods define behaviour. A stack is the textbook example of a small class, and it is a last in first out structure: push adds an item to the top and pop removes and returns the most recently added item, with a backing list where appending and popping from the end are both constant time operations. A queue is the first in first out counterpart. Inheritance lets one class derive from another and reuse or override its methods, which is written by putting the parent class name in parentheses after the class name. The three pillars usually named are encapsulation, inheritance and polymorphism. Special methods surrounded by double underscores let your objects work with built in syntax, so defining the length method makes the len function work, defining equality makes the double equals operator work, and defining the string method controls how the object prints. A property turns a method into something that looks like an attribute, which lets you add validation later without changing call sites. Class attributes are shared by all instances while instance attributes belong to one object, and confusing the two is a common source of bugs.""",
    },
    {
        "id": "core-tech-009",
        "title": "Algorithms: Searching, Sorting, Recursion and Complexity",
        "category": "coding",
        "text": """An algorithm is a step by step procedure for solving a problem, and complexity analysis describes how its cost grows as the input grows. Big O notation expresses that growth, ignoring constant factors. Linear search walks through a list checking each item in turn and takes time proportional to the length of the list, written O of n. Binary search is much faster, but it requires the data to be sorted first: you look at the middle element, and if it is not the target you discard the half that cannot contain it and repeat, halving the search space each step, which gives O of log n. Sorting puts data in order. Simple sorts such as bubble and insertion sort are O of n squared and are fine only for small inputs, while merge sort and quick sort are O of n log n. Merge sort works by splitting the list in half, sorting each half recursively and then merging the two sorted halves by repeatedly taking the smaller of the two front items, and merging two already sorted lists with two pointers is O of n. Recursion solves a problem in terms of smaller instances of itself and needs a base case to stop; the factorial of n is n times the factorial of n minus one with factorial of zero equal to one, and the Fibonacci sequence is defined so that each term is the sum of the previous two, though the naive recursive version repeats work exponentially and an iterative loop is far better. Euclid's algorithm computes the greatest common divisor of two numbers by repeatedly replacing the larger with its remainder on division by the smaller until one of them is zero. FizzBuzz prints Fizz for multiples of three, Buzz for multiples of five and FizzBuzz for multiples of both, and is a standard screening exercise.""",
    },
    {
        "id": "core-tech-010",
        "title": "JavaScript and Web Development Essentials",
        "category": "coding",
        "text": """JavaScript is the programming language that runs in every web browser and, thanks to Node.js, on servers too. It is dynamically typed and uses prototype based objects, and modern code is written with let and const rather than the older var, because var is function scoped while let and const are block scoped. Functions are declared with the function keyword or written as arrow functions, which are shorter and do not rebind the this keyword. A function returns a value with the return statement. JavaScript has one number type, a double precision float, and strings are immutable sequences with methods such as slice, split, toLowerCase and includes. To reverse a string you split it into an array of characters, reverse the array and join it back together, because strings themselves have no reverse method. Arrays carry a rich set of methods including map to transform each element, filter to keep those matching a test, reduce to combine them into a single value, find to locate one, and forEach to simply iterate. Objects are unordered collections of key value pairs and are the closest thing to a dictionary. The DOM, or Document Object Model, represents the page as a tree of nodes that scripts can read and change, using querySelector to find elements and addEventListener to respond to clicks and other events. Because network requests and timers are asynchronous, JavaScript relies on promises and the async and await syntax to wait for results without blocking the browser. JSON, JavaScript Object Notation, is the standard text format for exchanging structured data with a server.""",
    },
    {
        "id": "core-tech-011",
        "title": "Smartphone Hardware and Battery Technology",
        "category": "mobile",
        "text": """A modern smartphone packs the components of a small computer into a device a few millimetres thick. The system on a chip, or SoC, combines the CPU cores, the GPU, a neural processing unit, an image signal processor and the cellular modem onto one piece of silicon, and it is the single part that most determines performance. Qualcomm's Snapdragon series and MediaTek's Helio and Dimensity series power the majority of Android phones, while Apple designs its own A series chips. The display is usually an LCD or an OLED panel measured diagonally in inches, with a resolution such as 1080 pixels wide and a refresh rate in hertz. Battery capacity is measured in milliamp hours, written mAh, and a figure like 6000 mAh means the battery can in principle deliver 6000 milliamps for one hour. Capacity in mAh must not be confused with RAM, which is measured in gigabytes: a phone with 4 GB of memory and a 6000 mAh battery is a normal combination, and quoting a battery capacity in gigabytes is simply wrong. Lithium ion and lithium polymer cells are used because they store a lot of energy for their weight, and a charge cycle is one full discharge and recharge. Fast charging raises the current or voltage to refill the cell quickly, and charge controllers slow down as the battery fills to protect it. The Xiaomi Poco M3 is a budget Android phone that illustrates the typical specification sheet: a Qualcomm Snapdragon 662 processor, an Adreno 610 GPU, a 6000 mAh battery, a 6.53 inch display and 4 GB or 6 GB of RAM.""",
    },
    {
        "id": "core-tech-012",
        "title": "Databases, Machine Learning and Security Basics",
        "category": "tech",
        "text": """A database stores structured data so it can be queried reliably. Relational databases such as PostgreSQL, MySQL and SQLite organise data into tables with fixed columns and use SQL to query them, joining tables together on shared keys; SQLite is embedded inside almost every phone and browser because it needs no server. An index is a sorted copy of a column that makes lookups far faster, at the cost of slower writes and extra storage. Document databases such as MongoDB store JSON like records instead of rows and are more flexible but give up some guarantees. Machine learning builds models that find patterns in data rather than following hand written rules. A neural network is made of layers of artificial neurons, each computing a weighted sum of its inputs followed by a non linear activation function; adjusting the weights to reduce a loss function is called training, and it is done with backpropagation and an optimiser such as stochastic gradient descent. A transformer is the architecture behind today's large language models: it replaces recurrence with attention, a mechanism that lets every position in a sequence weigh how much to attend to every other position, which is expensive because it scales with the square of the sequence length. Quantisation shrinks a trained model by storing its weights in fewer bits, so a model stored as four bit integers is roughly a quarter the size of the same model in sixteen bit floats, with only a small loss in quality. Security rests on a few core ideas: authenticate who someone is, authorise what they may do, encrypt data in transit with TLS and at rest, hash passwords with a slow salted algorithm rather than storing them, and validate every input that crosses a trust boundary.""",
    },
]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from hexon.rag import chunk_text

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for a in ARTICLES:
            row = dict(a)
            row["source"] = "hexon_core_authored"
            row["word_count"] = len(a["text"].split())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    chunk_words: List[int] = []
    bad: List[tuple] = []
    for a in ARTICLES:
        pieces = chunk_text(a["text"], min_words=200, max_words=300, overlap_words=35)
        for p in pieces:
            w = len(p.split())
            chunk_words.append(w)
            if not (200 <= w <= 300):
                bad.append((a["id"], w))

    counts: Dict[str, int] = {}
    for a in ARTICLES:
        counts[a["category"]] = counts.get(a["category"], 0) + 1

    aw = [len(a["text"].split()) for a in ARTICLES]
    print(f"wrote {len(ARTICLES)} articles -> {OUT.relative_to(ROOT)}")
    print(f"  categories  : {counts}")
    print(f"  article len : min={min(aw)} max={max(aw)} avg={sum(aw)/len(aw):.1f}")
    print(f"  total chunks: {len(chunk_words)} (min={min(chunk_words)} "
          f"max={max(chunk_words)} avg={sum(chunk_words)/len(chunk_words):.1f})")
    print(f"  chunks outside 200-300 words: {bad if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
