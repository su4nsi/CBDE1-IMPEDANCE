import psycopg2
import csv
import os
import time

conn = psycopg2.connect(
    dbname="bookcorpus_db", user="myuser", password="mypassword", host="localhost")
cursor = conn.cursor()
insertion_times = []

chunks_dir = os.getcwd()  

for filename in os.listdir(chunks_dir):
    if filename.startswith("chunk_") and filename.endswith(".csv"):
        print(f"Cargando {filename} en PostgreSQL...")
        with open(os.path.join(chunks_dir, filename), mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  
            start_time = time.time()
            for row in reader:
                cursor.execute("INSERT INTO sentences (text) VALUES (%s)", (row[0],))
        conn.commit()
        end_time = time.time()
        insertion_times.append(end_time - start_time)
cursor.close()
conn.close()
min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = sum(insertion_times) / len(insertion_times)
std_dev_time = (sum((x - avg_time) ** 2 for x in insertion_times) / len(insertion_times)) ** 0.5
print(f"Insertion Times (Text): min: {min_time:.4f}s, max: {max_time:.4f}s, avg: {avg_time:.4f}s, stddev: {std_dev_time:.4f}s")
print("Complete")