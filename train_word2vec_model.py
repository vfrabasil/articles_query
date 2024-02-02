from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

# Load the list of article filenames from articles_list.txt
with open('articles_list.txt', 'r') as file:
    article_files = file.read().splitlines()

# Initialize an empty list to store tokenized text
tokenized_corpus = []

# Track statistics
total_words = 0
total_lines = 0

# Read and preprocess text from each article file
for article_file in article_files:
    with open(article_file, 'r', encoding='utf-8') as file:
        article_text = file.read()

        # Update statistics
        total_words += len(simple_preprocess(article_text))
        total_lines += article_text.count('\n')

        # Tokenize and preprocess the text
        tokens = simple_preprocess(article_text)
        tokenized_corpus.append(tokens)

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
model.save("word2vec_model.bin")

# Print summary
print(f"Processed {len(article_files)} articles.")
print(f"Total words processed: {total_words}")
print(f"Total lines processed: {total_lines}")
print(f"Word2Vec model trained and saved to word2vec_model.bin.")
