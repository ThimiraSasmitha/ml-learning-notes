from sentence_transformers import SentenceTransformer, util
import torch

# 1. Initialize the embedding model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Expanded knowledge base, grouped by topic for readability
kb_documents = [
    # --- Programming & software ---
    "Python is a popular programming language for data science, web development, and AI.",
    "JavaScript runs in every web browser and, with Node.js, on the server as well.",
    "Rust provides memory safety without a garbage collector through its ownership model.",
    "Git is a distributed version control system that tracks changes to source code.",
    "Docker packages applications and their dependencies into portable containers.",
    "Kubernetes automates deployment, scaling, and management of containerized applications.",
    "REST APIs use HTTP methods like GET, POST, PUT, and DELETE to expose resources.",
    "SQL is the standard language for querying and managing relational databases.",
    "Unit tests verify that individual functions behave correctly and catch regressions early.",

    # --- Machine learning & AI ---
    "Deep learning relies heavily on neural networks, backpropagation, and large datasets.",
    "Transformers use self-attention to model relationships between all tokens in a sequence.",
    "Gradient descent iteratively adjusts model parameters to minimize a loss function.",
    "Overfitting happens when a model memorizes training data and fails to generalize.",
    "Fine-tuning adapts a pretrained model to a specific task using a smaller dataset.",
    "Reinforcement learning trains agents to maximize cumulative reward through trial and error.",
    "Convolutional neural networks are widely used for image classification and detection.",
    "Tokenization splits text into smaller units such as words or subwords for language models.",

    # --- Retrieval & vector search ---
    "FAISS is a high-performance library for efficient similarity search of dense vectors.",
    "Vector databases help scale retrieval-augmented generation (RAG) pipelines for LLMs.",
    "Approximate nearest neighbor algorithms like HNSW trade a little accuracy for large speedups.",
    "Embeddings map text into dense numeric vectors where semantic similarity becomes geometric distance.",
    "Cosine similarity measures the angle between two vectors, ignoring their magnitude.",
    "BM25 is a classic keyword-based ranking function used in traditional search engines.",
    "Hybrid search combines keyword matching with semantic vector search for better recall.",
    "Chunking splits long documents into smaller passages before embedding them for retrieval.",

    # --- Data & infrastructure ---
    "Apache Spark processes large datasets in parallel across a cluster of machines.",
    "Data pipelines extract, transform, and load information between systems on a schedule.",
    "Caching stores frequently accessed data in memory to reduce latency.",
    "Load balancers distribute incoming traffic across multiple servers to improve reliability.",

    # --- Science & nature ---
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "DNA carries genetic instructions in a double-helix structure of paired nucleotides.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "Plate tectonics explains the movement of Earth's lithosphere and the formation of mountains.",
    "Vaccines train the immune system to recognize pathogens without causing the disease.",

    # --- Food & cooking ---
    "Chocolate chip cookies are best served warm with a cold glass of milk.",
    "Sourdough bread relies on wild yeast and bacteria for its tangy flavor and rise.",
    "Al dente pasta is cooked until firm to the bite, usually 8 to 10 minutes in salted water.",
    "Searing meat at high heat creates flavor through the Maillard reaction.",
    "Fresh basil, tomatoes, and mozzarella are the classic ingredients of a Caprese salad.",

    # --- Travel, sports & everyday life ---
    "Sri Lanka is known for its tea plantations, ancient temples, and palm-lined beaches.",
    "Marathon runners typically train for months, gradually increasing their weekly mileage.",
    "Regular sleep of seven to nine hours supports memory, mood, and overall health.",
    "Compound interest allows savings to grow exponentially as earnings are reinvested.",
    "Index funds offer low-cost, diversified exposure to the stock market.",
]

# 3. Encode the knowledge base into dense embeddings
print(f"Encoding {len(kb_documents)} knowledge base documents...")
kb_embeddings = model.encode(kb_documents, convert_to_tensor=True, show_progress_bar=True)


def search(query: str, top_k: int = 3):
    """Return the top_k most similar documents for a query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, kb_embeddings)[0]
    top_k = min(top_k, len(kb_documents))
    top_results = torch.topk(cos_scores, k=top_k)

    print(f"\nUser Query: '{query}'")
    print("-" * 70)
    for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
        print(f"{rank}. [{score:.4f}] {kb_documents[idx]}")

# 4. Try several queries across different topics
test_queries = [
    "How do I search through millions of vectors quickly?",
    "What's the best way to avoid my model memorizing the training data?",
    "How do plants make energy?",
    "Tips for baking something sweet",
    "How should I invest for the long term?",
    "Scaling and deploying apps across many servers",
]

for q in test_queries:
    search(q, top_k=3)
# 5. Define a user search query
user_query = "How do I search through millions of vectors quickly?"
print(f"\nUser Query: '{user_query}'\n")

# 6. Encode the query and compute cosine similarity against all documents
query_embedding = model.encode(user_query, convert_to_tensor=True)
cos_scores = util.cos_sim(query_embedding, kb_embeddings)[0]

# 7. Retrieve the top match
top_result_idx = torch.argmax(cos_scores).item()

print("Best Matching Knowledge Base Article:")
print(f"-> {kb_documents[top_result_idx]}")
print(f"Confidence Similarity Score: {cos_scores[top_result_idx]:.4f}")
