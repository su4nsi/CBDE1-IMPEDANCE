import chromadb
import csv
import os
import time

client = chromadb.PersistentClient(path=os.getcwd())
collection = client.create_collection(name="bookcorpus")

insertion_times = []
chunks_dir = os.getcwd()

for filename in os.listdir(chunks_dir):
    if filename.startswith("chunk_") and filename.endswith(".csv"):
        print(f"Cargando {filename} en Chroma...")
        with open(os.path.join(chunks_dir, filename), mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar el encabezado del CSV

            start_time = time.time()

            for idx, row in enumerate(reader):
                sentence = row[0]
                # Usar un ID único basado en el nombre del archivo y el índice
                collection.add(
                    ids=[f"{filename}_sentence_{idx}"],  # Asegúrate de que los IDs son únicos
                    documents=[sentence]
                )
            
            end_time = time.time()
            insertion_times.append(end_time - start_time)

# Después de la carga en ChromaDB
documents = collection.get()["documents"]
print(f"Total de documentos almacenados: {len(documents)}")

min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = sum(insertion_times) / len(insertion_times)
std_dev_time = (sum((x - avg_time) ** 2 for x in insertion_times) / len(insertion_times)) ** 0.5

print(f"Insertion Times (Chroma - Text): min: {min_time:.4f}s, max: {max_time:.4f}s, avg: {avg_time:.4f}s, stddev: {std_dev_time:.4f}s")
print("Carga completada en Chroma.")
