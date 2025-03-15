from langchain_together import TogetherEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import numpy as np
import os
import getpass


import pandas as pd

# Load your reviews dataset
df = pd.read_excel('reviews_data.xlsx')

if df.empty:
    raise ValueError("The dataset is empty. Please provide a valid reviews dataset.")

print(df.head())  # Preview the data

# Check for the API key and set it up
if not os.getenv("5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4"):
    os.environ["5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4"] = getpass.getpass("5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4 ")

# Initialize the TogetherEmbeddings model
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval"
)

# Extract reviews
df['Review'] = df['Review'].fillna("")  # Fill missing reviews with an empty string
reviews = df["Review"].tolist()

# Process embeddings in batches
embedding_list = []
batch_size = 128

for i in range(0, len(reviews), batch_size):
    batch = reviews[i : i + batch_size]
    batch_embeddings = embeddings.embed_documents(batch)
    embedding_list.extend(batch_embeddings)
    print(f"Processed {i + len(batch)} / {len(reviews)} reviews")

# Prepare metadata
metadata_list = df.apply(lambda row: {
    "customer_id": int(row["customer_id"]),
    "review_date": row["review_date_numeric"],
    "Rating": int(row["Rating"]),
    "review_id": row['review_id']
}, axis=1).tolist()

# Initialize Pinecone
pc = Pinecone(
    api_key=getpass.getpass("Enter your Pinecone API Key:'5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4")
)

# Check if index already exists
indexes = pc.list_indexes()
index_name = 'hotel-reviews'

if index_name not in indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        deletion_protection='enabled',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(name=index_name)

# Insert embeddings and metadata into the index
for i in range(0, len(embedding_list), batch_size):
    batch_vectors = [
        (str(i + j), embedding_list[i + j], metadata_list[i + j])
        for j in range(min(batch_size, len(embedding_list) - i))
    ]
    index.upsert(vectors=batch_vectors)
    print(f"Upserted batch from {i} to {i + len(batch_vectors)}")

# Experiment with querying the index
query_embedding = embeddings.embed_query(
    "What are some of the reviews that mention restaurant, food, lunch, breakfast, dinner"
)

results = index.query(
    vector=query_embedding,
    top_k=5,
    namespace="",
    include_metadata=True,
    filter={
        "Rating": {"$lte": 9},
        "review_date": {"$gte": 20240101, "$lte": 20240108}
    }
)

matches = results.get("matches", [])

if matches:
    matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
    req_df = df[df["review_id"].isin(matched_ids)]
    req_df['Review'] = req_df['Review'].fillna("")
    concatenated_reviews = " ".join(req_df["Review"].tolist())
else:
    concatenated_reviews = ""

if concatenated_reviews:
    from together import Together

    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {"role": "user", "content": f"""
                Briefly summarize the overall sentiment of customers about food and restaurant based on these reviews:
                {concatenated_reviews}. Don’t mention the name of the hotel.
            """}
        ]
    )

    print(response.choices[0].message.content)
else:
    print("No relevant reviews found in the given date range and rating filter.")

