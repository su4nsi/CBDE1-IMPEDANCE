from transformers import AutoModel, AutoTokenizer
import chromadb
import torch
import numpy as np
import time
import os

client = chromadb.PersistentClient(path=os.getcwd())

try:
    collection = client.get_collection(name="bookcorpus")
except:
    raise ValueError("La colección bookcorpus no existe o no se puede acceder a ella.")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

def generate_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

documents = collection.get()["documents"]
print(f"Primeros 5 documentos: {documents[:5]}")

insertion_times = []

for idx, sentence in enumerate(documents):
    embedding = generate_embedding(sentence)
    start_time = time.time()

    collection.update(
        ids=[f"sentence_{idx}"],
        embeddings=[embedding.tolist()]
    )

    end_time = time.time()
    insertion_time = end_time - start_time
    insertion_times.append(insertion_time)

    print(f"Embedding para sentence_{idx} guardado en {insertion_time:.4f} segundos.")

min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = sum(insertion_times) / len(insertion_times)
std_dev_time = (sum((x - avg_time) ** 2 for x in insertion_times) / len(insertion_times)) ** 0.5

print(f"Insertion Times (Embeddings): Min: {min_time:.4f}s, Max: {max_time:.4f}s, Avg: {avg_time:.4f}s, Stddev: {std_dev_time:.4f}s")
print("Embeddings generados y almacenados en Chroma.")
