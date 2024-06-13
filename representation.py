
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from data_ingestion import lecture_notes, model_architectures

# Load a pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for lecture notes
lecture_embeddings = {title: model.encode(content) for title, content in lecture_notes.items()}

# Generate embeddings for model architectures
architecture_embeddings = model.encode(model_architectures['Paper'].tolist())

# Create a FAISS index
dimension = 384  # Dimensions of the embeddings
index = faiss.IndexFlatL2(dimension)

# Add lecture notes embeddings to the index
for embedding in lecture_embeddings.values():
    index.add(np.array([embedding]))

# Add architecture embeddings to the index
index.add(np.array(architecture_embeddings))
