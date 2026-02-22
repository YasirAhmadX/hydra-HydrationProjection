# app.py
import streamlit as st
import pandas as pd
# Add this import at the top of app.py
from scripts.visualization_utils import make_water_loss_viz
from scripts.data_ingestion import connect_mongo, insert_subjects
from scripts.mongo_ml_pipeline import (
    connect_to_mongo,
    retrieve_subject_data,
    parse_record_to_features,
)
from scripts.model_inference import load_model_and_preproc, preprocess_and_predict

st.set_page_config(page_title="HYDRA ML Dashboard", page_icon="💧", layout="centered")
st.title("💧 HYDRA - Hydration Loss Prediction System")

# MongoDB connection
client, db = connect_mongo()
hydration_col = db["hydration_data"]

# Tabs
tabs = st.tabs(["➕ Insert Subject", "📋 Retrieve Subject", "🤖 AI Prediction"])

# --- TAB 1: INSERT SUBJECT ---
with tabs[0]:
    st.subheader("➕ Insert New Subject Data")

    with st.form("insert_form"):
        subject_id = st.number_input("Subject ID", min_value=1, step=1)
        gender = st.selectbox("Gender", ["male", "female", "trans"])
        age = st.number_input("Age", min_value=0, step=1)

        st.write("### Measurements")
        initial_weight = st.number_input("Initial Weight (kg)")
        final_weight = st.number_input("Final Weight (kg)")
        total_water = st.number_input("Total Water Consumed (ml)")

        st.write("### Gear 1 Readings")
        g1_sweat = st.number_input("Final Gear1 Sweat (kg)")
        g1_salt = st.number_input("Final Salt Lost (Gear1)")

        st.write("### Gear 2 Readings")
        g2_sweat = st.number_input("Final Gear2 Sweat (kg)")
        g2_salt = st.number_input("Final Salt Lost (Gear2)")

        target_loss = st.number_input("TARGET True Water Loss (kg)")

        submitted = st.form_submit_button("Insert Record")

        if submitted:
            subject_doc = {
                "Subject_ID": int(subject_id),
                "Gender": gender,
                "Age": age,
                "measurements": {
                    "Initial_Weight_kg": initial_weight,
                    "Final_Weight_kg": final_weight,
                    "Total_Water_Consumed_ml": total_water,
                    "final_readings": {
                        "Gear1": {"Sweat_kg": g1_sweat, "Salt_Lost": g1_salt},
                        "Gear2": {"Sweat_kg": g2_sweat, "Salt_Lost": g2_salt},
                    },
                    "TARGET_True_Water_Loss_kg": target_loss,
                },
            }

            count = insert_subjects(db, [subject_doc])
            if count > 0:
                st.success(f"✅ Subject {subject_id} inserted successfully into MongoDB!")

# --- TAB 2: RETRIEVE SUBJECT ---
with tabs[1]:
    st.subheader("📋 Retrieve Subject Data")

    sub_id = st.number_input("Enter Subject ID to Retrieve", min_value=1, step=1, key="retrieve_id")
    if st.button("Retrieve Data"):
        try:
            record = retrieve_subject_data(hydration_col, int(sub_id))
            st.json(record)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- TAB 3: AI PREDICTION ---
with tabs[2]:
    st.subheader("🤖 Predict Water Loss using DeepRNN")

    sub_id_pred = st.number_input("Enter Subject ID for Prediction", min_value=1, step=1, key="pred_id")
    if st.button("Run AI Prediction"):
        try:
            # Retrieve and preprocess subject data
            record = retrieve_subject_data(hydration_col, int(sub_id_pred))
            df_input = parse_record_to_features(record)

            # Load model and preprocessor
            model, preproc, feat_per_step, seq_len, device = load_model_and_preproc()
            prediction = preprocess_and_predict(df_input, model, preproc, feat_per_step, seq_len, device)

            st.success(f"🎯 Predicted TARGET_True_Water_Loss_kg: **{prediction:.6f} kg**")

            # --- Generate and display water loss visualization ---
            initial_weight = float(record["data"]["Initial_Weight_kg"])
            age = int(record.get("Age", 30))
            gender = record.get("Gender", "male")

            # Use your updated visualization utility
            viz_res = make_water_loss_viz(
                initial_weight_kg=initial_weight,
                predicted_loss_kg=prediction,
                age=age,
                gender=gender,
                image_path="assets/bodywater_by_age.jpg",   # ✅ path variable for your body image
                save_path="assets/hydration_viz.png"      # ✅ output visualization path
            )

            # Display visualization
            st.image(viz_res["viz_path"], use_container_width=True)

            # Display alert/safety message
            if viz_res["warning_>2pct"]:
                st.warning("⚠️ Predicted water loss exceeds 2.25% of body weight — dehydration risk.")
            else:
                st.success("✅ Predicted water loss is within safe range.")

            # Optional: Display computed metrics
            st.markdown(f"""
            **Hydration Summary**
            - Predicted Loss: `{prediction:.3f} kg`
            - % Body Weight Lost: `{viz_res['percent_loss']:.2f}%`
            - Avg Body Water (age={age}): `{viz_res['avg_water_pct']:.1f}%`
            - Remaining Water: `{viz_res['remaining_water_pct']:.2f}%`
            """)

            # Optional: Show input features used for prediction
            with st.expander("🧩 Show Extracted Features"):
                st.dataframe(df_input)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

client.close()
