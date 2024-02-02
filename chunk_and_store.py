import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
import faiss

# Load the list of article filenames from articles_list.txt
with open('articles_list.txt', 'r') as file:
    article_files = file.read().splitlines()

# Initialize Word2Vec model for text embedding
embedding_model = Word2Vec.load("word2vec_model.bin")  # Replace with the path to your pre-trained Word2Vec model

# Initialize an empty list to store article embeddings
article_embeddings = []

# Function to preprocess and embed text chunks
def embed_text(text):
    tokens = simple_preprocess(text)
    return np.mean([embedding_model.wv[token] for token in tokens if token in embedding_model.wv], axis=0)

# Iterate through each article file
for article_file in article_files:
    with open(article_file, 'r', encoding='utf-8') as file:
        article_text = file.read()

        # Split the article into chunks (adjust chunk_size as needed)
        chunk_size = 4000  # Approximate average words per chunk
        chunks = [article_text[i:i+chunk_size] for i in range(0, len(article_text), chunk_size)]

        # Embed each chunk and store the embeddings
        chunk_embeddings = [embed_text(chunk) for chunk in chunks]
        article_embeddings.extend(chunk_embeddings)

# Convert the list of embeddings to a numpy array
article_embeddings = np.array(article_embeddings)

# Initialize the Faiss index for vector search
index = faiss.IndexFlatL2(article_embeddings.shape[1])
index.add(article_embeddings)

# Save the embeddings and Faiss index to files for querying
np.save('article_embeddings.npy', article_embeddings)
faiss.write_index(index, 'faiss_index.index')

# Print summary
print(f"Processed {len(article_files)} articles.")
