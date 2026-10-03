!pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [
    "Machine learning models require clean data.",
    "Artificial intelligence algorithms process large amounts of information.",
    "I love baking chocolate chip cookies on weekends."
]
embeddings = model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Sample vector values:\n{embeddings[0][:5]}...")
