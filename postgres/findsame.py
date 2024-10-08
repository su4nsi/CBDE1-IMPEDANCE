import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import time

conn = psycopg2.connect(
    dbname="bookcorpus_db", user="myuser", password="mypassword", host="localhost")
cursor = conn.cursor()

cursor.execute("SELECT id, text, embedding FROM sentences ORDER BY RANDOM() LIMIT 10")
rows = cursor.fetchall()

sentence_ids = []
sentence_texts = []
embeddings = []

for row in rows:
    sentence_ids.append(row[0])
    sentence_texts.append(row[1])
    embeddings.append(np.array(row[2]))

print("Sentences:")
for text in sentence_texts:
    print(text)

time_cosine = []
time_euclidean = []

best_cosine_pair = None
best_euclidean_pair = None
best_cosine_score = -1
best_euclidean_score = float('inf')

start_time = time.time()
for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        if i != j:  
            cosine_sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            time_cosine.append(time.time() - start_time)
            
            euclidean_dist = euclidean_distances(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            time_euclidean.append(time.time() - start_time)
            
            if cosine_sim > best_cosine_score:
                best_cosine_score = cosine_sim
                best_cosine_pair = (sentence_texts[i], sentence_texts[j])

            if euclidean_dist < best_euclidean_score:
                best_euclidean_score = euclidean_dist
                best_euclidean_pair = (sentence_texts[i], sentence_texts[j])

print("\nMas similares:")
print(f"Cosine:")
print(f"1. {best_cosine_pair[0]}")
print(f"2. {best_cosine_pair[1]} (Score: {best_cosine_score:.4f})")
print("\nEuclidean:")
print(f"1. {best_euclidean_pair[0]}")
print(f"2. {best_euclidean_pair[1]} (Distance: {best_euclidean_score:.4f})")

if time_cosine:
    print("\nTime Stats cosine:")
    print(f"Min: {min(time_cosine):.6f} s")
    print(f"Max: {max(time_cosine):.6f} s")
    print(f"Avg: {np.mean(time_cosine):.6f} s")
    print(f"Std Dev: {np.std(time_cosine):.6f} s")

if time_euclidean:
    print("\nTime Stats euclidean:")
    print(f"Min: {min(time_euclidean):.6f} s")
    print(f"Max: {max(time_euclidean):.6f} s")
    print(f"Avg: {np.mean(time_euclidean):.6f} s")
    print(f"Std Dev: {np.std(time_euclidean):.6f} s")

cursor.close()
conn.close()
