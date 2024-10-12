# Calculating the size of the poem corpus
import json

import nltk
from nltk.tokenize import word_tokenize

# Step 1: Read the file and create a corpus with UTF-8 encoding
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Join lines into a single string for processing
text = ' '.join([entry["poem"].replace('\n', ' ') for entry in data])

# Step 2: Tokenize the text into individual words using NLTK
tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase

# Step 3: Calculate the number of tokens and types
num_tokens = len(tokens)          # Total number of tokens (including repetitions)
num_types = len(set(tokens))      # Total number of unique tokens (types)

# Step 4: Extract the unique poem IDs
unique_poem_ids = {entry['id'] for entry in data}  # Using set comprehension to ensure uniqueness

# Step 5: Count the number of unique poems
num_unique_poems = len(unique_poem_ids)

# Step 6: Print the results
print(f"Number of tokens in the corpus: {num_tokens}")
print(f"Number of types (unique words) in the corpus: {num_types}")
print(f"Number of unique poems in the corpus: {num_unique_poems}")