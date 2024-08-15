import json
import csv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "Food"
DIMENSION = 384 #vector dimension for sentence-transformers from hugging face
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
QDRANT_CLIENT = QdrantClient(url="http://qdrant:6333")

def generate_embedding(text: str):
    embeddings = EMBEDDING_MODEL.encode(text)
    return embeddings

def check_if_collection_exists() -> bool:
    return QDRANT_CLIENT.collection_exists(collection_name=COLLECTION_NAME)

def read_csv_and_upload_data_to_qdrant() -> None:
    count = 0
    payloads = []
    embds = []

    with open('./food.csv', mode='r', encoding="utf-8") as file:
        csv_reader = csv.DictReader(file) #dictReader for key/value pairs in rows

        next(csv_reader) # to skip first/header line in csv

        for row in csv_reader:
            payloads.append({
                'name': row['Name'],
                'calories': row['Calories'],
                'fat': row['Fat'],
                'protein': row['Protein'],
                'carbohydrate': row['Carbohydrate'],
                'sugar': row['Sugar'],
                'fiber': row['Fiber']
            })
            embds.append(generate_embedding(row['Name']))

            count+=1
            if count%10==0 and count >0:
                print(f'Embeddings generated for {count} foods')

    ids = []
    for x in range(len(payloads)):
        ids.append(x)

    QDRANT_CLIENT.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=ids,
            payloads=payloads,
            vectors=embds,
        ),
    )

def main() -> None:

    if check_if_collection_exists():
        QDRANT_CLIENT.delete_collection(collection_name=COLLECTION_NAME)

    QDRANT_CLIENT.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
    )

    read_csv_and_upload_data_to_qdrant()

    while True:
        search_term = input("What do you like to eat?: ")
        search_vec = generate_embedding(search_term)

        search_result = QDRANT_CLIENT.search(
            collection_name=COLLECTION_NAME,
            query_vector=search_vec,
            limit=3
        )
        print()

        for result in search_result:
            payload_string = json.dumps(result.payload, indent=4)
            payload_json = json.loads(payload_string)
            name = payload_json['name']
            calories = payload_json['calories']
            print(f"{name}, contains {calories}kCal/100g, with similarity score: {result.score}")

        keep_asking = input("Do you have another question?[Y/n]: ")

        if not keep_asking == 'Y' :
            print("See you again!")
            break


if __name__ == '__main__':
    main()
