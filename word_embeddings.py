import json
import nltk
from nltk.tokenize import RegexpTokenizer
from gensim.models import FastText
from stop_words import get_stop_words
from snowballstemmer import TurkishStemmer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd

# Download NLTK tokenizer
nltk.download('punkt')

# Step 1: Data Preparation

# Load data
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract 'poem' fields
corpus_raw = ' '.join([entry["poem"].replace('\n', ' ') for entry in data])

# Tokenize and clean text
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(corpus_raw.lower())

# Remove stopwords
turkish_stopwords = get_stop_words('turkish')
tokens = [word for word in tokens if word not in turkish_stopwords]

# Remove non-alphabetic tokens and short words
tokens = [word for word in tokens if word.isalpha() and len(word) > 2]

# Stemming
stemmer = TurkishStemmer()
tokens_stemmed = [stemmer.stemWord(word) for word in tokens]

# Prepare sentences
sentences = []
current_sentence = []

for word in tokens_stemmed:
    current_sentence.append(word)
    if word == 'nokta':  # Assuming 'nokta' indicates sentence end
        sentences.append(current_sentence)
        current_sentence = []

if current_sentence:
    sentences.append(current_sentence)

# Step 2: Train FastText Model
model_fasttext = FastText(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

# Step 3: Analyze Time Metaphors

# Find words similar to 'zaman'
zaman_stemmed = stemmer.stemWord('zaman')
similar_words = model_fasttext.wv.most_similar(zaman_stemmed, topn=20)

print("Most similar words to 'zaman':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

# Visualization
words = [zaman_stemmed] + [word for word, _ in similar_words]
word_vectors = model_fasttext.wv[words]

tsne = TSNE(n_components=2, random_state=42)
components = tsne.fit_transform(word_vectors)

plt.figure(figsize=(10, 7))
plt.scatter(components[:, 0], components[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(components[i, 0], components[i, 1]))

plt.title("t-SNE Visualization of 'zaman' and Similar Words")
plt.show()

# Clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(word_vectors)
labels = kmeans.labels_

df = pd.DataFrame({'word': words, 'cluster': labels})

for cluster in range(num_clusters):
    cluster_words = df[df['cluster'] == cluster]['word'].values
    print(f"\nCluster {cluster}:")
    print(', '.join(cluster_words))

