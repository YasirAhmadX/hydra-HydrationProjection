import csv
import pymongo

# ==== CONFIGURATION ====
CSV_PATH = r"E:\22MIA\4th year(2025-26)\Fall semester\CSE3086 NoSQL\Project\nosql_code\data\formatted_hydration_data.csv"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "HYDRA"

# ==== CONNECT TO MONGODB ====
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]

hydration_col = db["hydration_data"]
metadata_col = db["metadata"]

print("✅ Connected to MongoDB")

# ==== READ & PROCESS CSV FILE ====
subjects = []
with open(CSV_PATH, mode="r", encoding="utf-8-sig") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # Convert numeric fields to float where possible
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value  # keep as string if not a number

        # Create nested structure for each subject
        subject = {
            "Subject_ID": int(row["Subject_ID"]),
            "Gender": row["Gender"],
            "Age": row["Age"],
            "data": {
                "Initial_Weight_kg": row["Initial_Weight_kg"],
                "Final_Weight_kg": row["Final_Weight_kg"],
                "Total_Water_Consumed_ml": row["Total_Water_Consumed_ml"],
                "final_readings": {
                    "Gear s2": {
                        "Sweat_kg": row["Final_Gear1_Sweat_kg"],
                        "Salt_Lost": row["Final_Salt_Lost_1"]
                    },
                    "Gear fit 2": {
                        "Sweat_kg": row["Final_Gear2_Sweat_kg"],
                        "Salt_Lost": row["Final_Salt_Lost_2"]
                    }
                },
                "TARGET_True_Water_Loss_kg": row["TARGET_True_Water_Loss_kg"]
            }
        }
        subjects.append(subject)

# ==== CREATE METADATA ====
metadata = {
    "description": "Hydration and sweat loss data for study subjects",
    "total_subjects": len(subjects),
    "units": {
        "weight": "kg",
        "water_consumed": "ml",
        "salt_lost": "g (assumed)",
        "water_loss": "kg"
    }
}

# ==== INSERT INTO MONGODB ====
# Optional: clear previous data
metadata_col.delete_many({})
hydration_col.delete_many({})

# Insert new data
if metadata:
    metadata_col.insert_one(metadata)
    print("✅ Metadata inserted successfully.")

if subjects:
    hydration_col.insert_many(subjects)
    print(f"✅ Inserted {len(subjects)} subject records into 'hydration_data' collection.")
else:
    print("⚠️ No subject data found in the CSV file.")

# ==== VERIFY ====
print("\n📋 Collections in 'HYDRA' database:")
print(db.list_collection_names())

if "hydration_data" in db.list_collection_names():
    print("✅ 'hydration_data' collection exists.")
if "metadata" in db.list_collection_names():
    print("✅ 'metadata' collection exists.")
