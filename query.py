import os
import numpy as np
import faiss
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

# Load the list of article filenames from articles_list.txt
with open('articles_list.txt', 'r') as file:
    article_files = file.read().splitlines()

# Load the saved embeddings and Faiss index
article_embeddings = np.load('article_embeddings.npy')
index = faiss.read_index('faiss_index.index')

# Load the Word2Vec model
embedding_model = Word2Vec.load("word2vec_model.bin")

# User input for the search query
user_query = input("Enter your search query: ")

# Function to preprocess and embed the user query
def embed_text(text):
    tokens = simple_preprocess(text)
    return np.mean([embedding_model.wv[token] for token in tokens if token in embedding_model.wv], axis=0)

# Embed the user query
query_embedding = embed_text(user_query)

# Perform vector search using Faiss
k_results = 5  # Adjust the number of results as needed
_, result_indices = index.search(np.array([query_embedding]), k_results)

# Calculate cosine similarity between the query and each result
query_result_similarity = cosine_similarity(np.array([query_embedding]), article_embeddings[result_indices[0]])

# Check if the number of retrieved results is less than expected
if len(result_indices[0]) < k_results:
    print(f"Warning: Only {len(result_indices[0])} results retrieved, which is less than the specified k_results.")

# Retrieve and print the top K results with relevance percentages and contents
for i, idx in enumerate(result_indices[0]):
    # Check if the index is within the range of article_files
    if idx < len(article_files):
        relevance_percentage = query_result_similarity[0, i] * 100
        print(f"Result {i+1}: {article_files[idx]} - Relevance: {relevance_percentage:.2f}%")
        
        # Load and print the content of the article
        with open(article_files[idx], 'r', encoding='utf-8') as article_file:
            article_content = article_file.read()
            print(f"Content: {article_content[:200]}...")  # Displaying the first 200 characters for brevity
        print()
    else:
        print(f"Result {i+1}: Index {idx} out of range (len(article_files): {len(article_files)})")

# Print summary
print(f"Query processed: {user_query}")
