'''
Once the environment for the camera is built, perspective transforming must be applied if needed to flatten the image.
(Currently failed) Identify the mana symbols to filter the database (goal, sub second identification, etc).

Store all card data in a database for easy consultation. The database must be able to store the following information:
- Card name
- Card price
- Card type
- Card color
- Card mana cost
- Card text
- Card set
- Card rarity
- Card image
'''

'''
TODO:
1. Use Efficient OpenCV Functions (Vectorized Operations)
What it means:
OpenCV and NumPy are optimized for operations on whole arrays (images) at once, rather than looping pixel-by-pixel in Python.
Why:
Python loops are slow; vectorized operations are much faster because they use compiled C code under the hood.

Example: Instead of:

Use:

In your code:
Most of your image processing already uses OpenCV functions (cv2.cvtColor, cv2.Canny, etc.), which are efficient.
Tip:
Avoid writing custom loops for pixel-wise operations; always look for a NumPy or OpenCV function.

2. Efficient Search: Annoy or FAISS for Large Hash Databases
What it means:
If your hash database grows large (thousands+ cards), brute-force search (comparing every hash) becomes slow.
Annoy (Spotify) and FAISS (Facebook) are libraries for fast nearest-neighbor search in high-dimensional spaces.

How to use:

Convert your hashes to vectors (e.g., 64-bit phash → 64-dim vector of 0/1).
Build an Annoy or FAISS index with your database.
Query the index for the nearest neighbor(s) to your input hash.
Annoy Example:

Benefit:
Search time drops from O(N) to O(log N) or better.

3. Parallelize Hash Comparisons
What it means:
If you must brute-force search, use all CPU cores to compare hashes in parallel.

How to use:

Benefit:
On multi-core CPUs, this can make brute-force search 2–8× faster, depending on your hardware.

Summary:

Use OpenCV/NumPy functions, not Python loops, for image operations.
For large hash DBs, use Annoy/FAISS for fast nearest-neighbor search.
For brute-force, use concurrent.futures or multiprocessing to parallelize hash comparisons.
Let me know if you want a concrete code example for any of these in your context!
'''