from sentence_transformers import SentenceTransformer, util
import torch

# 1. Initialize the embedding model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define a custom knowledge base
kb_documents = [
    "Python is a popular programming language for data science, web development, and AI.",
    "Deep learning relies heavily on neural networks, backpropagation, and large datasets.",
    "FAISS is a high-performance library for efficient similarity search of dense vectors.",
    "Vector databases help scale retrieval-augmented generation (RAG) pipelines for LLMs.",
    "Chocolate chip cookies are best served warm with a cold glass of milk."
]

# 3. Encode the knowledge base into dense embeddings
print("Encoding knowledge base documents...")
kb_embeddings = model.encode(kb_documents, convert_to_tensor=True)

# 4. Define a user search query
user_query = "How do I search through millions of vectors quickly?"
print(f"\nUser Query: '{user_query}'\n")

# 5. Encode the query and compute cosine similarity against all documents
query_embedding = model.encode(user_query, convert_to_tensor=True)
cos_scores = util.cos_sim(query_embedding, kb_embeddings)[0]

# 6. Retrieve the top match
top_result_idx = torch.argmax(cos_scores).item()

# 7. Retrieve the bottom match 
bottom_result_idx = torch.argmin(cos_scores).item()

print("Best Matching Knowledge Base Article:")
print(f"-> {kb_documents[top_result_idx]}")
print(f"Confidence Similarity Score: {cos_scores[top_result_idx]:.4f}")

print("\nWorst Matching Knowledge Base Article:")
print(f"-> {kb_documents[bottom_result_idx]}")
print(f"Confidence Similarity Score: {cos_scores[bottom_result_idx]:.4f}")
