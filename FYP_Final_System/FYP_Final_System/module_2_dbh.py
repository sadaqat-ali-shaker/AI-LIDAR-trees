import pandas as pd
import joblib
from pathlib import Path
import sys

MODEL_PATH = "dbh_geometry_model.joblib" 

def predict_dbh(input_csv):
    """
    Loads Module 1 CSV, applies AI model, returns new CSV path.
    """
    print(f"[Module 2] Loading features from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model file {MODEL_PATH} not found!")
        print("[Module 2] Using Dummy Formula (Training mode skipped)...")
        # Simple formula for demo if model is missing
        df['Predicted_DBH'] = (df['Tree_Height'] * 0.8) + (df['Crown_Diameter'] * 2.0)
    else:
        
        print(f"[Module 2] Loading AI Model: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
       
        
        features_for_model = pd.DataFrame()
        features_for_model['Height [m]'] = df['Tree_Height']
        features_for_model['Crown diameter [m]'] = df['Crown_Diameter']
        
        
        df['Predicted_DBH'] = model.predict(features_for_model)

    # Save
    output_csv = input_csv.parent / f"M2_With_DBH_{input_csv.name}"
    df.to_csv(output_csv, index=False)
    print(f"[Module 2] DBH Predicted. Saved to {output_csv}")
    return output_csv