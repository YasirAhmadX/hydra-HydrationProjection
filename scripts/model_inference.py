# stdin_infer.py
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_DIR = Path(r"model\deeprnn_artifacts")
MODEL_CHECKPOINT = MODEL_DIR / "deep_rnn_state_dict.pt"
PREPROC_PATH = MODEL_DIR / "preprocessor.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# make sure these match training FEATURES order
FEATURES = [
    "Gender",
    "Age",
    "Initial_Weight_kg",
    "Total_Water_Consumed_ml",
    "Final_Gear1_Sweat_kg",
    "Final_Salt_Lost_1",
    "Final_Gear2_Sweat_kg",
    "Final_Salt_Lost_2"
]

# Define model class exactly as used during training
class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, h_n = self.rnn(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)

def load_model_and_preproc(model_path=MODEL_CHECKPOINT, preproc_path=PREPROC_PATH, device=DEVICE):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preproc_path}")

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    input_size = cfg.get("input_size")
    hidden_size = cfg.get("hidden_size", 128)
    num_layers = cfg.get("num_layers", 3)
    dropout = cfg.get("dropout", 0.3)
    feat_per_step = ckpt.get("feat_per_step", None)
    seq_len = ckpt.get("seq_len", None) or 4

    model = DeepRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)

    return model, preproc, int(feat_per_step), int(seq_len), device

def read_features_from_stdin():
    """
    Prompts user for values for each feature in FEATURES.
    For numeric features enter a number (or blank to use NaN -> imputed).
    For categorical features (Gender) enter a string token.
    """
    print("Enter feature values. To leave a numeric value missing, press Enter (will be treated as NaN and imputed).")
    vals = {}
    # define numeric vs categorical here
    numeric_feats = set([
        "Age", "Initial_Weight_kg", "Total_Water_Consumed_ml",
        "Final_Gear1_Sweat_kg", "Final_Salt_Lost_1",
        "Final_Gear2_Sweat_kg", "Final_Salt_Lost_2"
    ])
    for feat in FEATURES:
        raw = input(f"  {feat}: ").strip()
        if raw == "":
            # blank -> NaN for numeric, empty string for categorical
            if feat in numeric_feats:
                vals[feat] = np.nan
            else:
                vals[feat] = ""  # let imputer/encoder handle it
        else:
            if feat in numeric_feats:
                try:
                    vals[feat] = float(raw)
                except ValueError:
                    print(f"  Warning: couldn't parse '{raw}' as float for {feat}. Using NaN.")
                    vals[feat] = np.nan
            else:
                vals[feat] = raw
    # create single-row DataFrame
    df = pd.DataFrame([vals], columns=FEATURES)
    return df

def preprocess_and_predict(df_input: pd.DataFrame, model, preproc, feat_per_step, seq_len, device=DEVICE):
    # transform with preprocessor
    Xpr = preproc.transform(df_input[FEATURES])
    # ensure divisible and pad zeros if needed
    n_features = Xpr.shape[1]
    needed = feat_per_step * seq_len
    if n_features < needed:
        pad_cols = needed - n_features
        Xpr = np.concatenate([Xpr, np.zeros((Xpr.shape[0], pad_cols))], axis=1)
    elif n_features > needed:
        # trim extra columns (training used [:feat_per_step*seq_len])
        Xpr = Xpr[:, :needed]

    Xseq = Xpr.reshape(-1, seq_len, feat_per_step)
    Xt = torch.tensor(Xseq, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(Xt).cpu().numpy().flatten()[0]
    return float(pred)

def main():
    model, preproc, feat_per_step, seq_len, device = load_model_and_preproc()
    print(f"Loaded model (feat_per_step={feat_per_step}, seq_len={seq_len}) on device={device}\n")
    df_input = read_features_from_stdin()
    try:
        pred = preprocess_and_predict(df_input, model, preproc, feat_per_step, seq_len, device)
        print(f"\nPredicted TARGET_True_Water_Loss_kg = {pred:.6f}")
    except Exception as e:
        print("Error during prediction:", e)

if __name__ == "__main__":
    main()
