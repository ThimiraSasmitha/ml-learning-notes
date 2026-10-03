!pip install faiss-cpu
import numpy as np
import faiss

# 1. Generate dummy embedding vectors (e.g., 1000 documents, 384 dimensions matching our model)
dimension = 384
num_documents = 1000
np.random.seed(42)
database_embeddings = np.random.random((num_documents, dimension)).astype('float32')

# 2. Normalize vectors for cosine similarity search using Inner Product
faiss.normalize_L2(database_embeddings)

# 3. Create FAISS Index for Inner Product (cosine similarity on normalized vectors)
index = faiss.IndexFlatIP(dimension)
index.add(database_embeddings)
print(f"Total vectors indexed in FAISS: {index.ntotal}")

# 4. Simulate a user query embedding (shape: 1, 384)
query_embedding = np.random.random((1, dimension)).astype('float32')
faiss.normalize_L2(query_embedding)

# 5. Search for the top 3 closest matches
k = 3
distances, indices = index.search(query_embedding, k)

print("\nTop Search Results:")
for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. Document ID: {idx} | Similarity Score: {score:.4f}")
