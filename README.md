# **HYDRA - Hydration Loss Prediction System**
## Comprehensive Documentation

### **Project Overview**
HYDRA is a **NoSQL-based machine learning project** designed to forecast hydration levels and provide dehydration alerts. It combines a **Streamlit web interface**, **MongoDB database**, and a **Deep RNN neural network** to predict water loss in individuals based on their physiological measurements.

---

## **1. HOW TO REPLICATE THIS PROJECT**

### **1.1 System Requirements**
- **Python**: >= 3.11
- **MongoDB**: Local instance (running on `mongodb://localhost:27017/`)
- **Operating System**: Windows, macOS, or Linux

### **1.2 Installation Steps**

#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/YasirAhmadX/hydra-HydrationProjection.git
cd hydra-HydrationProjection
```

#### **Step 2: Create a Virtual Environment** (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### **Step 3: Install Dependencies**
The project uses a `pyproject.toml` file for dependency management:

```bash
pip install -r requirements.txt
# OR using the specified versions:
pip install matplotlib>=3.10.7 pymongo>=4.15.3 seaborn>=0.13.2 streamlit>=1.51.0 torch>=2.9.0
```

**Dependencies:**
- `streamlit` - Interactive web framework
- `pymongo` - MongoDB driver
- `torch` - Deep learning framework
- `matplotlib` & `seaborn` - Data visualization
- `pandas` - Data manipulation

#### **Step 4: Set Up MongoDB**
1. **Install MongoDB** from [mongodb.com](https://www.mongodb.com/try/download/community)
2. **Start MongoDB service**:
   - **Windows**: MongoDB runs as a service by default
   - **macOS**: `brew services start mongodb-community`
   - **Linux**: `sudo systemctl start mongod`
3. Verify connection: `mongosh` (should connect to local instance)

#### **Step 5: Extract Model Artifacts**
```bash
cd model
unzip deeprnn_artifacts.zip
cd ..
```
This extracts:
- `deep_rnn_state_dict.pt` - Trained model weights
- `preprocessor.pkl` - Data preprocessing pipeline

---

## **2. HOW TO RUN THE PROJECT**

### **2.1 Run the Streamlit Dashboard (Recommended)**
```bash
streamlit run main.py
```
- Opens interactive web interface at `http://localhost:8501`
- Provides 3 main tabs: Insert Data, Retrieve Data, Predict & Visualize

### **2.2 Run MongoDB Ingestion Script (CLI)**
```bash
python scripts/data_ingestion.py
```
- Interactively collect subject metadata and measurements
- Stores data directly in MongoDB

### **2.3 Run Batch Data Ingestion**
```bash
python scripts/data_ingestion_batch.py
```
- For bulk loading multiple subject records

### **2.4 Run Standalone Inference**
```bash
python scripts/mongo_ml_pipeline.py
```
- Query database and run predictions on specific subjects
- Command-line based inference

---

## **3. PROJECT STRUCTURE & FLOW**

### **3.1 Directory Layout**
```
hydra-HydrationProjection/
├── main.py                          # 🎯 Main Streamlit application
├── main_local.py                    # Alternative local version
├── pyproject.toml                   # Project configuration & dependencies
├── uv.lock                          # Dependency lock file
│
├── scripts/                         # 📚 Modular utility scripts
│   ├── data_ingestion.py           # MongoDB data collection (interactive)
│   ├── data_ingestion_batch.py     # Batch data loading
│   ├── data_retrieval.py           # Query subject data
│   ├── mongo_ml_pipeline.py        # MongoDB + ML pipeline integration
│   ├── model_inference.py          # DeepRNN model loading & prediction
│   └── visualization_utils.py      # Hydration visualization generation
│
├── model/                          # 🧠 ML model artifacts
│   ├── NoSQL_Project.ipynb        # Jupyter notebook (model development)
│   ├── deeprnn_artifacts.zip      # Compressed model files
│   └── deeprnn_artifacts/         # Extracted artifacts
│       ├── deep_rnn_state_dict.pt # Model weights
│       └── preprocessor.pkl       # Preprocessing pipeline
│
├── data/                           # 📊 Data directory (for local files)
├── assets/                         # 🖼️ Images & visualizations
│   ├── hydration_viz.png          # Example output visualization
│   └── bodywater_by_age.jpg       # Reference body image (optional)
│
└── README.md                        # Project description
```

### **3.2 Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYDRA System Flow                             │
└─────────────────────────────────────────────────────────────────┘

1️⃣ DATA INGESTION PHASE
   ↓
   User Input (Streamlit UI or CLI)
   ↓
   data_ingestion.py
   ↓
   Create MongoDB Document:
   {
     Subject_ID: int,
     Gender: str,
     Age: float,
     measurements: {
       Initial_Weight_kg: float,
       Final_Weight_kg: float,
       Total_Water_Consumed_ml: float,
       final_readings: {
         Gear1: {Sweat_kg, Salt_Lost},
         Gear2: {Sweat_kg, Salt_Lost}
       },
       TARGET_True_Water_Loss_kg: float
     }
   }
   ↓
   MongoDB Collection: hydration_data

2️⃣ DATA RETRIEVAL & FEATURE EXTRACTION
   ↓
   mongo_ml_pipeline.py (retrieve_subject_data)
   ↓
   Query: db.hydration_data.findOne({Subject_ID: X})
   ↓
   parse_record_to_features() - Maps MongoDB fields to model FEATURES:
   - Gender (categorical)
   - Age (numeric)
   - Initial_Weight_kg (numeric)
   - Total_Water_Consumed_ml (numeric)
   - Final_Gear1_Sweat_kg (numeric)
   - Final_Salt_Lost_1 (numeric)
   - Final_Gear2_Sweat_kg (numeric)
   - Final_Salt_Lost_2 (numeric)

3️⃣ MODEL INFERENCE PHASE
   ↓
   model_inference.py (load_model_and_preproc)
   ↓
   Load:
   - Preprocessor: Scales/encodes features
   - DeepRNN Model: 3-layer RNN with 128 hidden units
   ↓
   Reshape features: (1, 4, 8) → (batch=1, seq_len=4, feat_per_step=8)
   ↓
   Forward Pass → Prediction
   ↓
   Output: TARGET_True_Water_Loss_kg (float)

4️⃣ VISUALIZATION & ALERTS
   ↓
   visualization_utils.py (make_water_loss_viz)
   ↓
   Compute:
   - % Body Weight Lost = (predicted_loss / initial_weight) × 100
   - Dehydration Risk = if % > 2.25%
   - Remaining Water % = avg_body_water % - loss %
   ↓
   Generate Visual:
   - Body composition image
   - Summary table
   - Hydration bar chart
   - Risk alerts (⚠️ or ✅)
   ↓
   Save: assets/hydration_viz.png
```

---

## **4. KEY COMPONENTS EXPLAINED**

### **4.1 Data Ingestion (`scripts/data_ingestion.py`)**
- **Purpose**: Collect and store subject hydration measurements
- **Functions**:
  - `connect_mongo()` - Establish DB connection
  - `collect_metadata()` - Store dataset metadata
  - `collect_subject_data()` - Interactive data collection loop
  - `insert_subjects()` - Batch insert into MongoDB
- **Database**: MongoDB (HYDRA database, hydration_data collection)

### **4.2 MongoDB-ML Pipeline (`scripts/mongo_ml_pipeline.py`)**
- **Purpose**: Bridge between MongoDB and ML model
- **Key Functions**:
  - `retrieve_subject_data()` - Query MongoDB for subject
  - `parse_record_to_features()` - Convert flexible MongoDB schema to fixed DataFrame
  - Handles schema variations (e.g., "data" vs "measurements" fields)

### **4.3 Model Inference (`scripts/model_inference.py`)**
- **Model Type**: DeepRNN (Recurrent Neural Network)
- **Architecture**:
  ```
  Input (features: 8) 
    ↓ Preprocess
  Input (batch=1, seq_len=4, features=8)
    ↓ RNN (3 layers, hidden=128, dropout=0.3)
  Hidden State
    ↓ FC (hidden → hidden/2 → 1)
  Output: Water Loss Prediction
  ```
- **Key Functions**:
  - `load_model_and_preproc()` - Load trained weights & preprocessor
  - `preprocess_and_predict()` - Transform input and generate prediction

### **4.4 Visualization (`scripts/visualization_utils.py`)**
- **Purpose**: Create interpretable hydration status visualization
- **Outputs**:
  - Summary table with key metrics
  - Horizontal bar chart showing water loss
  - Dehydration risk alerts (2.25% threshold)
  - Age-adjusted body water percentage

### **4.5 Streamlit Dashboard (`main.py`)**
- **Interface**: 3-tab interactive dashboard
  - **Tab 1: ➕ Insert Subject** - Form-based data entry
  - **Tab 2: 📋 Retrieve Subject** - Query and display MongoDB records
  - **Tab 3: 🤖 AI Prediction** - Run inference and display results

---

## **5. FEATURE ENGINEERING & MODEL INPUTS**

The model expects exactly **8 features** in this order:

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | male/female/trans |
| Age | Numeric | Age in years |
| Initial_Weight_kg | Numeric | Starting body weight |
| Total_Water_Consumed_ml | Numeric | Fluid intake during period |
| Final_Gear1_Sweat_kg | Numeric | Sweat collected from Gear 1 |
| Final_Salt_Lost_1 | Numeric | Electrolytes lost (Gear 1) |
| Final_Gear2_Sweat_kg | Numeric | Sweat collected from Gear 2 |
| Final_Salt_Lost_2 | Numeric | Electrolytes lost (Gear 2) |

**Target Variable**: `TARGET_True_Water_Loss_kg`

---

## **6. WORKFLOW EXAMPLES**

### **Example 1: Using Streamlit Dashboard**
```bash
# Terminal 1: Start MongoDB
mongod

# Terminal 2: Launch Streamlit
streamlit run main.py

# In Browser (http://localhost:8501):
# 1. Insert a subject with their measurements
# 2. Retrieve to verify data
# 3. Run prediction and view visualization
```

### **Example 2: CLI-based Workflow**
```bash
# 1. Insert data via script
python scripts/data_ingestion.py
# → Enter metadata and subject info interactively

# 2. Run inference on inserted data
python scripts/mongo_ml_pipeline.py
# → Enter Subject_ID when prompted
# → Displays prediction and extracted features
```

---

## **7. IMPORTANT NOTES & TROUBLESHOOTING**

| Issue | Solution |
|-------|----------|
| "MongoDB connection refused" | Ensure MongoDB is running (`mongod` or service) |
| "Model checkpoint not found" | Extract `deeprnn_artifacts.zip` in the `model/` directory |
| "Module not found" | Install dependencies: `pip install -r requirements.txt` |
| Path errors on Windows | Use `/` or `\\` consistently; model paths use `\` |

---

## **8. CUSTOMIZATION & EXTENSION**

To modify the project:
1. **Add new features**: Update FEATURES list in `model_inference.py`
2. **Change MongoDB URI**: Edit `MONGO_URI` in scripts
3. **Adjust model**: Retrain using the Jupyter notebook (`model/NoSQL_Project.ipynb`)
4. **Customize UI**: Edit Streamlit layout in `main.py`

---

## **License**
See `LICENSE` file for terms.

---

**Created by**: Yasir Ahmad  
**Last Updated**: February 2026

