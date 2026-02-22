"""
data_ingestion.py
-----------------
Modular MongoDB ingestion script for Hydration Dataset.
Collects metadata and subject data interactively and stores them in MongoDB.

Author: Yasir Ahmad
"""

import pymongo
from pymongo import MongoClient

# ==== CONFIGURATION ====
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "HYDRA"
COLL_HYDRATION = "hydration_data"
COLL_METADATA = "metadata"


# ==== CONNECT FUNCTION ====
def connect_mongo(uri=MONGO_URI, db_name=DB_NAME):
    """Establish MongoDB connection."""
    client = MongoClient(uri)
    db = client[db_name]
    print(f"✅ Connected to MongoDB database '{db_name}'\n")
    return client, db


# ==== METADATA INGESTION ====
def collect_metadata(db):
    """Collect metadata information interactively and insert into MongoDB."""
    print("--- Enter Metadata Information ---")
    description = input("Description of dataset (default: Hydration and sweat loss data): ") or "Hydration and sweat loss data"
    weight_unit = input("Weight unit (default: kg): ") or "kg"
    water_unit = input("Water consumed unit (default: ml): ") or "ml"
    salt_unit = input("Salt lost unit (default: g (assumed)): ") or "g (assumed)"
    water_loss_unit = input("Water loss unit (default: kg): ") or "kg"

    metadata = {
        "description": description,
        "units": {
            "weight": weight_unit,
            "water_consumed": water_unit,
            "salt_lost": salt_unit,
            "water_loss": water_loss_unit,
        },
    }

    meta_col = db[COLL_METADATA]
    meta_col.delete_many({})  # overwrite existing
    meta_col.insert_one(metadata)
    print("✅ Metadata inserted successfully.\n")
    return metadata


# ==== SUBJECT INGESTION ====
def collect_subject_data():
    """Collect subject-level hydration data interactively."""
    subjects = []

    print("--- Enter Subject Data ---")
    while True:
        try:
            subject_id = int(input("Subject ID: "))
            gender = input("Gender (male/female/trans): ").strip().lower()
            age = float(input("Age: "))

            initial_weight = float(input("Initial Weight (kg): "))
            final_weight = float(input("Final Weight (kg): "))
            total_water_consumed = float(input("Total Water Consumed (ml): "))

            gear1_sweat = float(input("Final Gear1 Sweat (kg): "))
            gear1_salt = float(input("Final Salt Lost (Gear1): "))
            gear2_sweat = float(input("Final Gear2 Sweat (kg): "))
            gear2_salt = float(input("Final Salt Lost (Gear2): "))

            target_loss = float(input("TARGET True Water Loss (kg): "))

            # Create structured document
            subject_doc = {
                "Subject_ID": subject_id,
                "Gender": gender,
                "Age": age,
                "measurements": {
                    "Initial_Weight_kg": initial_weight,
                    "Final_Weight_kg": final_weight,
                    "Total_Water_Consumed_ml": total_water_consumed,
                    "final_readings": {
                        "Gear1": {"Sweat_kg": gear1_sweat, "Salt_Lost": gear1_salt},
                        "Gear2": {"Sweat_kg": gear2_sweat, "Salt_Lost": gear2_salt},
                    },
                    "TARGET_True_Water_Loss_kg": target_loss,
                },
            }

            subjects.append(subject_doc)
            print(f"✅ Added subject {subject_id} to buffer.")

            another = input("\nAdd another subject? (y/n): ").strip().lower()
            if another != "y":
                break

        except ValueError as e:
            print(f"⚠️ Invalid input: {e}. Please try again.\n")
            continue

    return subjects


# ==== INSERT FUNCTION ====
def insert_subjects(db, subjects):
    """Insert subject records into MongoDB."""
    if not subjects:
        print("\n⚠️ No subject data entered. Nothing inserted.")
        return 0

    hydration_col = db[COLL_HYDRATION]
    hydration_col.insert_many(subjects)
    print(f"\n✅ Successfully inserted {len(subjects)} subject record(s) into '{COLL_HYDRATION}'.")
    return len(subjects)


# ==== VERIFY FUNCTION ====
def verify_collections(db):
    """Verify collections exist in MongoDB."""
    print("\n📋 Collections in database:")
    collections = db.list_collection_names()
    print(collections)

    if COLL_HYDRATION in collections:
        print(f"✅ '{COLL_HYDRATION}' collection exists.")
    if COLL_METADATA in collections:
        print(f"✅ '{COLL_METADATA}' collection exists.")
    print()


# ==== MAIN ====
def main():
    """Main entrypoint for data ingestion."""
    client, db = connect_mongo()
    try:
        metadata = collect_metadata(db)
        subjects = collect_subject_data()
        insert_subjects(db, subjects)
        verify_collections(db)
    finally:
        client.close()
        print("🔒 MongoDB connection closed.")


# ==== ENTRYPOINT ====
if __name__ == "__main__":
    main()
