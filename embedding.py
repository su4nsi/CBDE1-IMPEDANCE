from transformers import AutoTokenizer, AutoModel
import torch
import psycopg2
import time
import numpy as np

conn = psycopg2.connect(
    dbname="bookcorpus_db", user="myuser", password="mypassword", host="localhost")
cursor = conn.cursor()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
cursor.execute("SELECT id, text FROM sentences")
rows = cursor.fetchall()
insertion_times = []
for row in rows:
    sentence_id, sentence_text = row
    inputs = tokenizer(sentence_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, 1).squeeze().numpy()
    start_time = time.time()
    cursor.execute("UPDATE sentences SET embedding = %s WHERE id = %s", (embedding.tolist(), sentence_id))
    conn.commit()
    end_time = time.time()
    insertion_times.append(end_time - start_time)
min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = sum(insertion_times) / len(insertion_times)
std_dev_time = (sum((x - avg_time) ** 2 for x in insertion_times) / len(insertion_times)) ** 0.5
print(f"Insertion Times (Embeddings): Min: {min_time:.4f}s, Max: {max_time:.4f}s, Avg: {avg_time:.4f}s, Std Dev: {std_dev_time:.4f}s")
cursor.close()
conn.close()
