from transformers import AutoModel, AutoTokenizer
import chromadb
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
import time
import random

client = chromadb.PersistentClient(path=os.getcwd())
collection = client.get_collection(name="bookcorpus")

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
ids = collection.get()["ids"]

print(f"Primeros 5 documentos: {documents[:5]}")
print(f"Primeros 5 IDs: {ids[:5]}")

embeddings = []
valid_sentences = []
for idx, sentence in enumerate(documents):
    if sentence is None or not sentence.strip():
        print(f"Oración vacía o nula en el índice {idx}, se omite.")
        continue
    embedding = generate_embedding(sentence)
    embeddings.append(embedding)
    valid_sentences.append(sentence)
    print(f"Embedding generado para {ids[idx]}")

if embeddings:
    embeddings = np.array(embeddings)
    random_indices = random.sample(range(len(valid_sentences)), 10)
    selected_sentence_texts = [valid_sentences[i] for i in random_indices]
    selected_embeddings = np.array([embeddings[i] for i in random_indices])

    print("\nOraciones seleccionadas:")
    for text in selected_sentence_texts:
        print(text)

    best_cosine_pair = None
    best_euclidean_pair = None
    best_cosine_score = -1
    best_euclidean_score = float('inf')

    time_cosine = []
    time_euclidean = []

    for i in range(len(selected_embeddings)):
        for j in range(len(selected_embeddings)):
            if i != j:
                start_cosine = time.time()
                cosine_sim = cosine_similarity(selected_embeddings[i].reshape(1, -1), selected_embeddings[j].reshape(1, -1))[0][0]
                end_cosine = time.time()
                time_cosine.append(end_cosine - start_cosine)

                start_euclidean = time.time()
                euclidean_dist = euclidean_distances(selected_embeddings[i].reshape(1, -1), selected_embeddings[j].reshape(1, -1))[0][0]
                end_euclidean = time.time()
                time_euclidean.append(end_euclidean - start_euclidean)

                if cosine_sim > best_cosine_score:
                    best_cosine_score = cosine_sim
                    best_cosine_pair = (selected_sentence_texts[i], selected_sentence_texts[j])

                if euclidean_dist < best_euclidean_score:
                    best_euclidean_score = euclidean_dist
                    best_euclidean_pair = (selected_sentence_texts[i], selected_sentence_texts[j])

    print("\nOraciones más similares (Cosine Similarity):")
    print(f"1. {best_cosine_pair[0]}")
    print(f"2. {best_cosine_pair[1]} (Score: {best_cosine_score:.4f})")

    print("\nOraciones más similares (Euclidean Distance):")
    print(f"1. {best_euclidean_pair[0]}")
    print(f"2. {best_euclidean_pair[1]} (Distance: {best_euclidean_score:.4f})")

    if time_cosine:
        print("\nEstadísticas de tiempo (Cosine Similarity):")
        print(f"Mínimo: {min(time_cosine):.6f} s")
        print(f"Máximo: {max(time_cosine):.6f} s")
        print(f"Promedio: {np.mean(time_cosine):.6f} s")
        print(f"Desviación estándar: {np.std(time_cosine):.6f} s")

    if time_euclidean:
        print("\nEstadísticas de tiempo (Euclidean Distance):")
        print(f"Mínimo: {min(time_euclidean):.6f} s")
        print(f"Máximo: {max(time_euclidean):.6f} s")
        print(f"Promedio: {np.mean(time_euclidean):.6f} s")
        print(f"Desviación estándar: {np.std(time_euclidean):.6f} s")

else:
    print("No se generaron embeddings.")
