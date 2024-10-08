from datasets import load_dataset
import csv
import os

output_dir = os.getcwd()

print("Downloading BookCorpus dataset")
dataset = load_dataset("bookcorpus")

print("Extracting sentences from dataset")
sentences = [sentence['text'] for sentence in dataset['train']]

max_sentences = 10000
if len(sentences) > max_sentences:
    sentences = sentences[:max_sentences]

chunk_size = 1000  

print("Splitting the corpus into chunks")
chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

for i, chunk in enumerate(chunks):
    with open(os.path.join(output_dir, f'chunk_{i}.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])
        for sentence in chunk:
            writer.writerow([sentence])

    print(f"Chunk {i} saved to {output_dir}/chunk_{i}.csv")

print(f"Corpus divided into {len(chunks)} chunks, with a total of {len(sentences)} sentences.")