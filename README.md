# Introduction
My attempt at transforming all those theory and equations from a book into reports and code!

Note on AI: I have kept the usage of AI to a minimum as much as possible to genuinely pick up on the skills I attempted to program. For disclosure, it was used in three major ways throughout the programming aspects
- building a logical and sensible pipeline. This was a largely one-sided usage as the AI guided me what kind of files I would probably need to make a logical pipeline. I usually did not follow the AI word-for-word because it sometimes recommended arbitrarily complex workflows. Instead, it was a mix of intuition and the AI output.
- making matplotlib outputs prettier. Honestly, I wish I could say I had a flair for this stuff, but I do not. Most of the time, I would end up with actually workable data and a... "graph", but it was unbearably ugly. I usually just ask AI to then spruce it up with colours and whatever magic it does. That is another reason you may notice the matplotlib and plotting section of the code to be far less... in my style.
- debugging. self-explanatory.

https://www.linkedin.com/in/shashwat-sarawagi/

# Quant
The Quant folder focuses, as the name suggests, on quantitative finance projects, steering a bit into the area of algorithmic trading. Importantly, any mini-project that lies in a different directory from another will not crossreference the other. For instance, I don't use my BSM function's implied volatility calculator in the projects outside the impVolSurface.

Also, I highly rely on Bloomberg API to extract data for testing my code thanks to Imperial College London's subscription to Bloomberg. Due to their TnCs, naturally, I cannot have the data or results displayed. The former is a clear requirement of the API usage, but the latter is just me being extra careful (cause no risks).

The books that created this part of the repository are as follows
1. Paul Wilmott on Quantitative Finance: A three-volume set of engaging and approachable breakdowns of the subject area. Some VBasic Code that did provide a good foundation for some of these, but largely a process of converting the formulae into a pipeline for the code, instead of just "translating" the VBasic, which tended to be less comprehensive.
2. Python for Finance Book by Yves Hilpisch: A big fan. It has a solid walkthrough of the Python logic from scratch, which is included through files like dataScience.py where I cemented my Python skills.

# Algo
## Books
The books that created this part of the repository are as follows
1. Python for Algorithmic Trading by Yves Hilpisch: The code from this book is very straightforward, hence not included in this repository as the replication would be almost identical. That would, naturally, be a violation of its copyright. Nonetheless, an important founding brick for my understanding.
2. Algorithmic Trading: Winning Strategies and Their Rationale by Ernest Chan: This book pushed my understanding a lot more than the first book. I was forced to dig deeper into each function I implemented since the book relied heavily on MATLAB, while I programmed in Python. A lot of times, Chan also spoke of concepts without direct MATLAB implementation making me write my own code from scratch. This was also done over a much longer time due to limited access to this book (owing to irregular visits to the British Library). While I only programmed a few of his discussions, I did read the others in equal depth but did not find as much value in programming them.

## Papers (Read descriptions in respective READMEs
1. ACT: Anti-Crosstalk Learning for Cross-Sectional Stock Ranking Implementation of the ACT framework proposed by Juntao Li and Liang Zhang.
