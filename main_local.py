# scripts/mongo_ml_pipeline.py
import pymongo
import pandas as pd
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Import inference utilities from your existing script
from scripts.model_inference import (
    load_model_and_preproc,
    preprocess_and_predict,
    FEATURES,
)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "HYDRA"

def connect_to_mongo():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    hydration_col = db["hydration_data"]
    print("✅ Connected to MongoDB database 'HYDRA'")
    return client, hydration_col

def retrieve_subject_data(hydration_col, subject_id: int) -> Dict[str, Any]:
    record = hydration_col.find_one({"Subject_ID": subject_id}, {"_id": 0})
    if not record:
        raise ValueError(f"No record found for Subject_ID={subject_id}")
    
    display_record = record.copy()
    if "data" in display_record and "TARGET_True_Water_Loss_kg" in display_record["data"]:
        del display_record["data"]["TARGET_True_Water_Loss_kg"]

    print(f"📋 Retrieved record for Subject_ID={subject_id}")
    print(json.dumps(display_record, indent=4))
    """
        print(f"📋 Retrieved record for Subject_ID={subject_id}")
        print(json.dumps(record, indent=4))
        return record
    """
    return display_record
def _safe_get(d: Dict[str, Any], *keys, default=None):
    """Try sequence of keys on dict d, return first found or default."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

def _extract_gear(final_readings: Dict[str, Any], gear_index: int):
    """
    Pick gear entry by index (0 or 1) from final_readings.
    Returns (sweat, salt) or (None, None) if not found.
    """
    if not isinstance(final_readings, dict) or len(final_readings) == 0:
        return None, None

    # Deterministic order: sort keys so behavior is consistent
    keys = sorted(final_readings.keys())
    if gear_index >= len(keys):
        return None, None

    gear_key = keys[gear_index]
    gear_obj = final_readings.get(gear_key, {})
    # possible sweat keys
    sweat = _safe_get(gear_obj, "Sweat_kg", "Sweat", "sweat_kg", "sweat")
    salt = _safe_get(gear_obj, "Salt_Lost", "Salt_Lost_1", "Salt_Lost_2", "Salt", "salt_lost", "Salt_Lost(g)")
    return _to_float(sweat), _to_float(salt)

def parse_record_to_features(record: Dict[str, Any]) -> pd.DataFrame:
    """
    Map MongoDB record to the FEATURES expected by the model.
    Handles multiple schemata (data vs measurements) and variable gear names.
    """
    # top-level gender/age may be present at root
    gender = record.get("Gender", "") or record.get("gender", "") or ""
    age = _to_float(_safe_get(record, "Age", "age"))

    # two common places for measurement block: 'data' or 'measurements'
    data_block = _safe_get(record, "data", "measurements", default={}) or {}

    initial_w = _to_float(_safe_get(data_block, "Initial_Weight_kg", "Initial_Weight", "initial_weight", default=None))
    final_w = _to_float(_safe_get(data_block, "Final_Weight_kg", "Final_Weight", "final_weight", default=None))
    total_water = _to_float(_safe_get(data_block, "Total_Water_Consumed_ml", "Total_Water_Consumed", "Total_Water_Consumed_ml", default=None))

    # final_readings might be under data_block["final_readings"] or data_block["final_readings "]
    final_readings = _safe_get(data_block, "final_readings", "final readings", default={}) or {}

    # If final_readings empty but gear keys are directly under data_block, try to detect
    if (not final_readings) and isinstance(data_block, dict):
        # heuristic: keys that contain 'Gear' or 'gear' or 'fit' are gear entries
        candidate = {k: v for k, v in data_block.items() if isinstance(v, dict) and ("gear" in k.lower() or "fit" in k.lower() or "s2" in k.lower())}
        if candidate:
            final_readings = candidate

    # Extract gear1 & gear2 values (first two gear entries sorted by key)
    g1_sweat, g1_salt = _extract_gear(final_readings, 0)
    g2_sweat, g2_salt = _extract_gear(final_readings, 1)

    # If any value is still None, try alternate names directly at top-level measurement block
    if g1_sweat is None:
        g1_sweat = _to_float(_safe_get(data_block, "Final_Gear1_Sweat_kg", "Gear1_Sweat", default=None))
    if g1_salt is None:
        g1_salt = _to_float(_safe_get(data_block, "Final_Salt_Lost_1", "Gear1_Salt", default=None))
    if g2_sweat is None:
        g2_sweat = _to_float(_safe_get(data_block, "Final_Gear2_Sweat_kg", "Gear2_Sweat", default=None))
    if g2_salt is None:
        g2_salt = _to_float(_safe_get(data_block, "Final_Salt_Lost_2", "Gear2_Salt", default=None))

    # Build mapping in the FEATURES order expected by model_inference
    mapped = {
        "Gender": gender,
        "Age": age,
        "Initial_Weight_kg": initial_w,
        "Total_Water_Consumed_ml": total_water,
        "Final_Gear1_Sweat_kg": g1_sweat,
        "Final_Salt_Lost_1": g1_salt,
        "Final_Gear2_Sweat_kg": g2_sweat,
        "Final_Salt_Lost_2": g2_salt,
    }

    # Print helpful warnings for missing keys (so you can inspect/clean DB)
    for k, v in mapped.items():
        if v is None or (isinstance(v, str) and v == ""):
            print(f"⚠️ Warning: extracted feature '{k}' is missing or empty (value={v})")

    df = pd.DataFrame([mapped], columns=FEATURES)
    return df

def main():
    client, hydration_col = connect_to_mongo()
    try:
        subject_id = int(input("\nEnter Subject_ID to run inference: "))
        record = retrieve_subject_data(hydration_col, subject_id)

        df_input = parse_record_to_features(record)
        print("\n🔍 Extracted Features for Inference (DataFrame):")
        print(df_input)

        print("\n⚙️ Loading DeepRNN model and preprocessor...")
        model, preproc, feat_per_step, seq_len, device = load_model_and_preproc()
        print(f"✅ Model loaded (feat_per_step={feat_per_step}, seq_len={seq_len}) on device={device}")

        prediction = preprocess_and_predict(df_input, model, preproc, feat_per_step, seq_len, device)
        print(f"\n🎯 Predicted TARGET_True_Water_Loss_kg = {prediction:.6f}")

    except Exception as e:
        print("❌ Error:", e)
    finally:
        client.close()
        print("\n🔒 MongoDB connection closed.")

if __name__ == "__main__":
    main()
