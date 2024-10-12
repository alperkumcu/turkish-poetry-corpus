import nltk
# Download required NLTK data (run this once if not already done)
nltk.download('punkt')

import json
from nltk.tokenize import word_tokenize
from math import log
from collections import defaultdict
import re
import csv
import string  # For handling punctuation

# Step 1: Import the 'train.json' file and read it using UTF-8 encoding
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 2: Create a corpus from the "poem" field and clean up newline codes (\n)
corpus = ' '.join([entry["poem"].replace('\n', ' ') for entry in data])

# Step 3: Tokenize the text into individual words using NLTK
tokens = word_tokenize(corpus.lower())  # Tokenize and convert to lowercase

# Step 4: Define a function to match any form of 'zaman' using regex
def is_zaman_form(word):
    return bool(re.match(r'^zaman.*', word))

# Step 5: Function to check if a token is a valid word (i.e., not punctuation)
def is_valid_word(word):
    return word not in string.punctuation and not re.match(r'^[^\w]+$', word)

# Initialize dictionaries to store left and right collocations
collocations_left = defaultdict(int)
collocations_right = defaultdict(int)

# Step 6: Analyze left and right collocations of 'zaman' and its affixed forms
for i, word in enumerate(tokens):
    if is_zaman_form(word):
        # Check if there's a valid word to the left
        if i > 0 and is_valid_word(tokens[i - 1]):
            collocations_left[tokens[i - 1]] += 1
        # Check if there's a valid word to the right
        if i < len(tokens) - 1 and is_valid_word(tokens[i + 1]):
            collocations_right[tokens[i + 1]] += 1

# Step 7: Calculate raw frequency, MI, and Log-Likelihood

# Function to calculate Mutual Information (MI)
def calculate_mi(word, collocation_count, word_freq, total_tokens, collocation_word_freq):
    p_word_collocate = collocation_count / total_tokens
    p_word = word_freq[word] / total_tokens
    p_collocate = collocation_word_freq / total_tokens

    if p_word_collocate > 0 and p_word > 0 and p_collocate > 0:
        return log(p_word_collocate / (p_word * p_collocate), 2)  # log base 2
    else:
        return 0  # Avoid negative infinity or undefined cases

# Function to calculate Log-Likelihood
def calculate_log_likelihood(word, collocation_count, word_freq, total_tokens, collocation_word_freq):
    E11 = (word_freq[word] * collocation_word_freq) / total_tokens
    O11 = collocation_count

    if E11 > 0:
        return 2 * (O11 * log(O11 / E11) if O11 > 0 else 0)
    else:
        return 0

# Count the frequency of all words in the corpus
word_freq = defaultdict(int)
for token in tokens:
    if is_valid_word(token):  # Only count words, not punctuation
        word_freq[token] += 1

# Total number of tokens in the corpus (excluding punctuation)
total_tokens = sum(word_freq.values())

# Collocation statistics storage
collocation_stats = []

# Calculate statistics for left collocations
for word, count in collocations_left.items():
    if is_valid_word(word):  # Ensure the collocation is a valid word
        mi_score = calculate_mi('zaman', count, word_freq, total_tokens, word_freq[word])
        log_likelihood = calculate_log_likelihood('zaman', count, word_freq, total_tokens, word_freq[word])
        collocation_stats.append((word, 'left', count, mi_score, log_likelihood))

# Calculate statistics for right collocations
for word, count in collocations_right.items():
    if is_valid_word(word):  # Ensure the collocation is a valid word
        mi_score = calculate_mi('zaman', count, word_freq, total_tokens, word_freq[word])
        log_likelihood = calculate_log_likelihood('zaman', count, word_freq, total_tokens, word_freq[word])
        collocation_stats.append((word, 'right', count, mi_score, log_likelihood))

# Step 8: Sort the collocations by Log-Likelihood in descending order
collocation_stats_sorted = sorted(collocation_stats, key=lambda item: item[4], reverse=True)

# Step 9: Print the sorted results
print("Collocation Statistics for 'zaman' and its affixed forms (sorted by Log-Likelihood):")
for stat in collocation_stats_sorted:
    print(f"Word: {stat[0]} ({stat[1]}), Frequency: {stat[2]}, MI: {stat[3]:.4f}, Log-Likelihood: {stat[4]:.4f}")

# Step 10: Export the output to a .csv file
with open('zaman_collocations_large.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Word', 'Position', 'Frequency', 'MI', 'Log-Likelihood'])
    for stat in collocation_stats_sorted:
        writer.writerow([stat[0], stat[1], stat[2], f"{stat[3]:.4f}", f"{stat[4]:.4f}"])