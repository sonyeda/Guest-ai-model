import pymongo

mongodb_uri = "mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = pymongo.MongoClient( 'mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0', serverSelectionTimeoutMS=5000)
    db = client.test
    db.command('ping')
    print("✅ Connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
