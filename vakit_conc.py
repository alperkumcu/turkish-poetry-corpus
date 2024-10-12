import json
import re
import csv
import nltk
from nltk.tokenize import word_tokenize
import string

# Step 1: Import the 'train.json' file and read it using UTF-8 encoding
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 2: Create a corpus from the "poem" field and clean up newline codes (\n)
corpus = ' '.join([entry["poem"].replace('\n', ' ') for entry in data])

# Step 3: Tokenize the text into individual words using NLTK
tokens = word_tokenize(corpus.lower())  # Tokenize and convert to lowercase


# Step 4: Define a function to match any form of 'vakit' using regex
def is_vakit_form(word):
    return bool(re.match(r'^vakit.*', word))


# Step 5: Function to check if a token is a valid word (i.e., not punctuation)
def is_valid_word(word):
    return word not in string.punctuation and not re.match(r'^[^\w]+$', word)


# Initialize a list to store concordance results
concordance_list = []

# Step 6: Analyze the corpus and find concordances for 'vakit' and its forms
window_size = 5  # Define the window size: 5 words to the left and right
for i, word in enumerate(tokens):
    if is_vakit_form(word) and is_valid_word(word):
        # Get 5 words to the left, making sure not to go out of bounds
        left_context = [w for w in tokens[max(0, i - window_size):i] if is_valid_word(w)]

        # Get 5 words to the right, making sure not to go out of bounds
        right_context = [w for w in tokens[i + 1:i + 1 + window_size] if is_valid_word(w)]

        # Join the context into strings
        left_context_str = ' '.join(left_context[-window_size:])  # Get the last 5 words on the left
        right_context_str = ' '.join(right_context[:window_size])  # Get the first 5 words on the right

        # Append the concordance entry
        concordance_list.append((left_context_str, word, right_context_str))

# Step 7: Print the concordance results
print(f"{'Left Context':<30} {'Node':<10} {'Right Context'}")
print("-" * 70)
for left, node, right in concordance_list:
    print(f"{left:<30} {node:<10} {right}")

# Step 8: Export the concordance output to a .csv file
with open('vakit_concordance.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Left Context', 'Node', 'Right Context'])
    for left, node, right in concordance_list:
        writer.writerow([left, node, right])
