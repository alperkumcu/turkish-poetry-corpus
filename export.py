import json
from nltk.tokenize import word_tokenize
import csv

# Step 1: Read the file and create a corpus with UTF-8 encoding
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 2: Create the Turkish Poetry Corpus as a single text string
corpus_text = ' '.join([entry["poem"].replace('\n', ' ') for entry in data])

# Step 3: Tokenize the text into individual words
tokens = word_tokenize(corpus_text.lower())  # Tokenize and convert to lowercase
print(f"Number of tokens in the corpus: {len(tokens)}")

# Step 4: Export the corpus as a .txt file
with open('Turkish_Poetry_Corpus.txt', 'w', encoding='utf-8') as file:
    file.write(corpus_text)
print("Corpus has been exported as 'Turkish_Poetry_Corpus.txt'.")

# Step 5: Export the corpus with metadata as a .csv file
with open('Turkish_Poetry_Corpus.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Poem'])  # Write the header
    for entry in data:
        writer.writerow([entry['id'], entry['poem'].replace('\n', ' ')])
print("Corpus has been exported as 'Turkish_Poetry_Corpus.csv'.")

# Step 6: Export the corpus with metadata as an .xml file
with open('Turkish_Poetry_Corpus.xml', 'w', encoding='utf-8') as file:
    file.write('<Corpus>\n')
    for entry in data:
        file.write(f'  <Poem id="{entry["id"]}">\n')
        file.write(f'    <Text>{entry["poem"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")}</Text>\n')
        file.write('  </Poem>\n')
    file.write('</Corpus>')
print("Corpus has been exported as 'Turkish_Poetry_Corpus.xml'.")

# Step 7: Export the tokenized words into a .csv file (one word per cell)
with open('Turkish_Poetry_Corpus_Tokenized.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Token'])  # Header for tokenized words
    for token in tokens:
        writer.writerow([token])  # Write each token as a single row
print("Tokenized words have been exported as 'Turkish_Poetry_Corpus_Tokenized.csv'.")