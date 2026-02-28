import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


GT_FILE = Path("Demo_Ground_Truth.csv")


AUTO_FILE = Path("Auto_Extracted_Results/Auto_Inventory.csv")


MATCH_DISTANCE_THRESHOLD = 2.0 

def calculate_accuracy():

    if not GT_FILE.exists() or not AUTO_FILE.exists():
        print("[ERROR] Files not found! Make sure both CSVs are in the folder.")
        return

    df_gt = pd.read_csv(GT_FILE)
    df_auto = pd.read_csv(AUTO_FILE)

    print(f"Loaded Ground Truth: {len(df_gt)} trees")
    print(f"Loaded Auto Results: {len(df_auto)} trees")

    gt_coords = df_gt[['Easting [m]', 'Northing [m]']].values
    

    auto_coords = df_auto[['X_Location', 'Y_Location']].values
    tree = cKDTree(auto_coords)
    

    distances, indices = tree.query(gt_coords, k=1)

    matches = 0
    matched_gt_indices = []
    
    for i, dist in enumerate(distances):
        if dist <= MATCH_DISTANCE_THRESHOLD:
            matches += 1
            matched_gt_indices.append(i)

    
    true_positives = matches
    false_negatives = len(df_gt) - matches  
    false_positives = len(df_auto) - matches

    recall = (true_positives / len(df_gt)) * 100   
    precision = (true_positives / len(df_auto)) * 100
    
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

  
    print("\n" + "="*40)
    print("      ACCURACY REPORT CARD")
    print("="*40)
    print(f"Ground Truth Trees:     {len(df_gt)}")
    print(f"Automated Trees Found:  {len(df_auto)}")
    print("-" * 40)
    print(f"✅ Correct Matches (TP): {true_positives}")
    print(f"❌ Missed Trees (FN):    {false_negatives}")
    print(f"⚠️ Extra/Noise (FP):     {false_positives}")
    print("-" * 40)
    print(f"🎯 Detection Rate (Recall): {recall:.2f}%")
    print(f"💎 Precision:               {precision:.2f}%")
    print(f"🏆 F1 Score:                {f1_score:.2f}%")
    print("="*40)
    
    if recall > 80:
        print("\nCONCLUSION: The Algorithm is HIGHLY ACCURATE.")
    elif recall > 60:
        print("\nCONCLUSION: The Algorithm is ACCEPTABLE (Needs minor tuning).")
    else:
        print("\nCONCLUSION: The Algorithm needs Improvement (Under-segmentation).")

if __name__ == "__main__":
    calculate_accuracy()