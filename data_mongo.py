import glob
from pymongo import MongoClient
import pandas as pd

# Load dining data from Excel file
for f in glob.glob("./*.xlsx"):
    dataset = pd.read_excel("dining_info.xlsx")

# Clean data (if necessary)
if 'Unnamed: 0' in dataset.columns:
    dataset.drop('Unnamed: 0', axis=1, inplace=True)

# Connect to MongoDB
client = MongoClient("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["hotel_guests"]
collection = db["dining_info"]

# Clear existing data (optional, if needed)
# collection.delete_many({})

# Insert data into MongoDB
try:
    collection.insert_many(dataset.to_dict(orient="records"))
    print("Data successfully inserted into MongoDB!")
except Exception as e:
    print(f"An error occurred while inserting data: {e}")

# Verify insertion
record_count = collection.count_documents({})
print(f"Total records in collection: {record_count}")

# Close connection
client.close()

# ✅ This code now perfectly aligns with your guest dining modeling workflow!
# Let me know if you'd like me to add data validation or handle duplicates differently. 🚀
