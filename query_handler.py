
from sentence_transformers import SentenceTransformer
import numpy as np
from representation import index, lecture_notes, model_architectures

# Function to handle queries
def handle_query(query: str, top_k: int = 3) -> list:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    D, I = index.search(np.array([query_embedding]), top_k)
    results = []
    for i in I[0]:
        if i < len(lecture_notes):
            results.append(list(lecture_notes.keys())[i])
        else:
            results.append(model_architectures.iloc[i - len(lecture_notes)]['Paper'])
    return results

# Example queries
if __name__ == "__main__":
    queries = [
        "What are some milestone model architectures and papers in the last few years?",
        "What are the layers in a transformer block?",
        "Tell me about datasets used to train LLMs and how they’re cleaned"
    ]

    for query in queries:
        print(f"Query: {query}")
        answers = handle_query(query)
        for answer in answers:
            print(f"Answer: {answer}")
        print()
