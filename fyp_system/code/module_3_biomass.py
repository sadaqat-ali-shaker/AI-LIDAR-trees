import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path


EXTERNAL_DB_PATH = "species_density.csv" 

def calculate_biomass_final(input_csv):
    """
    Matches extracted trees with external DB, assigns Species/WD, calc Biomass.
    """
    print(f"[Module 3] matching with External Database: {EXTERNAL_DB_PATH}...")
    
    df_extracted = pd.read_csv(input_csv)
    
    if not Path(EXTERNAL_DB_PATH).exists():
        raise FileNotFoundError("External Species Database not found!")
        
    df_ref = pd.read_csv(EXTERNAL_DB_PATH)
    
    ref_coords = df_ref[['Easting [m]', 'Northing [m]']].values
    ext_coords = df_extracted[['Easting', 'Northing']].values
    
    # --- SPATIAL MATCHING ---
    print("[Module 3] Running Spatial Matching...")
    tree = cKDTree(ref_coords)
    dists, idxs = tree.query(ext_coords, k=1)
    
    # Threshold for matching (e.g., 2.5 meters)
    MATCH_LIMIT = 2.5
    
    final_rows = []
    
    for i, dist in enumerate(dists):
        # Data from Extraction
        row_ext = df_extracted.iloc[i]
        
        if dist < MATCH_LIMIT:
            # Match Found! Get Species/WD from Reference
            ref_idx = idxs[i]
            row_ref = df_ref.iloc[ref_idx]
            
            real_id = row_ref['Tree_ID']
            species = row_ref['Specie']
            # Handle column name variations for Wood Density
            wd_col = 'Wood density' if 'Wood density' in df_ref.columns else 'Wood_Density'
            wd = row_ref.get(wd_col, 0.6)
            
        else:
            real_id = f"Unknown_{row_ext['Temp_ID']}"
            species = "Unknown"
            wd = 0.5 # Default generic wood density
            
        # --- BIOMASS CALCULATION ---
        # Formula: AGB = 0.0673 * (WD * DBH^2 * H)^0.976 (Chave et al. 2014)
        # Assuming DBH is in cm, H in m. Result in kg.
        dbh_cm = row_ext['Predicted_DBH']
        h_m = row_ext['Tree_Height']
        
        agb = 0.0673 * ((wd * (dbh_cm**2) * h_m) ** 0.976)
        
        final_rows.append({
            "Tree_ID": real_id,
            "Specie": species,
            "Tree_Height": h_m,
            "Crown_Diameter": row_ext['Crown_Diameter'],
            "Wood_Density": wd,
            "DBH": round(dbh_cm, 2),
            "Biomass_kg": round(agb, 2),
            "Northern_Point": row_ext['Northing'],
            "Eastern_Point": row_ext['Easting'],
            "Date": row_ext['Date'],
            "PLOT_NUMBER": row_ext['Plot_Number']
        })
        
    df_final = pd.DataFrame(final_rows)
    
    # Save Final Result
    final_path = input_csv.parent / "FINAL_INVENTORY_REPORT.csv"
    df_final.to_csv(final_path, index=False)
    
    print("\n" + "="*40)
    print("      ✅ FINAL PIPELINE COMPLETE")
    print("="*40)
    print(f"Final Report Generated: {final_path}")
    print(f"Total Trees Processed: {len(df_final)}")
    
    return final_path