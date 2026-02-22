import pymongo
import json

# ==== MongoDB Configuration ====
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "HYDRA"

# ==== Connect to MongoDB ====
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
hydration_col = db["hydration_data"]

print("✅ Connected to MongoDB")

# ==== Input Subject ID ====
try:
    subject_id = int(input("\nEnter Subject ID to retrieve: "))

    # ==== Retrieve Data ====
    record = hydration_col.find_one({"Subject_ID": subject_id}, {"_id": 0})  # Exclude _id

    if record:
        print("\n📋 Retrieved Data:")
        print(json.dumps(record, indent=4))  # Pretty print JSON
    else:
        print(f"\n⚠️ No record found for Subject_ID = {subject_id}")

except ValueError:
    print("⚠️ Invalid input. Please enter a valid integer Subject_ID.")
except Exception as e:
    print(f"❌ Error retrieving data: {e}")

finally:
    client.close()
