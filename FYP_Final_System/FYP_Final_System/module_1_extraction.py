import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import sys

# --- SETTINGS ---
GRID_SIZE = 0.1
MIN_TREE_HEIGHT = 2.0
MIN_DISTANCE_PIXELS = 7  # 0.7m separation
SMOOTHING_SIGMA = 1.0

def process_uls_data(laz_path, plot_number, survey_date, output_dir):
    """
    Main function to process ULS data.
    Returns path to the generated CSV.
    """
    input_path = Path(laz_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / f"M1_Extracted_{plot_number}.csv"
    
    print(f"[Module 1] Loading {input_path.name}...")
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} not found.")
        
    las = laspy.read(str(input_path))
    xyz = np.vstack((las.x, las.y, las.z)).T

    # 1. DTM (Local Ground)
    print("[Module 1] Building Terrain Model...")
    x_min, x_max = xyz[:,0].min(), xyz[:,0].max()
    y_min, y_max = xyz[:,1].min(), xyz[:,1].max()
    res = 2.0
    cols = int((x_max - x_min) / res) + 1
    rows = int((y_max - y_min) / res) + 1
    x_idx = ((xyz[:,0] - x_min) / res).astype(int)
    y_idx = ((xyz[:,1] - y_min) / res).astype(int)
    df_grid = pd.DataFrame({'r': y_idx, 'c': x_idx, 'z': xyz[:,2]})
    dtm_grid = df_grid.groupby(['r', 'c'])['z'].min().reset_index()
    dtm = np.full((rows, cols), np.nan)
    dtm[dtm_grid['r'], dtm_grid['c']] = dtm_grid['z']
    global_min = np.nanmin(dtm)
    dtm[np.isnan(dtm)] = global_min 

    # 2. CHM & Detection
    print("[Module 1] Segmenting Trees...")
    cols_chm = int((x_max - x_min) / GRID_SIZE) + 1
    rows_chm = int((y_max - y_min) / GRID_SIZE) + 1
    x_idx_chm = ((xyz[:,0] - x_min) / GRID_SIZE).astype(int)
    y_idx_chm = ((xyz[:,1] - y_min) / GRID_SIZE).astype(int)
    
    chm = np.zeros((rows_chm, cols_chm))
    df_chm = pd.DataFrame({'r': y_idx_chm, 'c': x_idx_chm, 'z': xyz[:,2]})
    grid_max = df_chm.groupby(['r', 'c'])['z'].max().reset_index()
    chm[grid_max['r'], grid_max['c']] = grid_max['z']
    
    chm_norm = chm - global_min
    chm_norm[chm_norm < 0] = 0
    chm_smooth = ndimage.gaussian_filter(chm_norm, sigma=SMOOTHING_SIGMA)
    
    local_maxi = peak_local_max(chm_smooth, min_distance=MIN_DISTANCE_PIXELS, threshold_abs=MIN_TREE_HEIGHT)
    markers = np.zeros_like(chm_smooth, dtype=int)
    for i, (r, c) in enumerate(local_maxi): markers[r, c] = i + 1
    labels = watershed(-chm_smooth, markers, mask=chm_smooth > MIN_TREE_HEIGHT)

    # 3. Extraction
    print("[Module 1] Extracting Features & Saving LAZ segments...")
    max_r, max_c = labels.shape
    valid_mask = (x_idx_chm >= 0) & (x_idx_chm < max_c) & (y_idx_chm >= 0) & (y_idx_chm < max_r)
    point_labels = np.zeros(len(xyz), dtype=int)
    point_labels[valid_mask] = labels[y_idx_chm[valid_mask], x_idx_chm[valid_mask]]
    
    unique_ids = np.unique(point_labels)
    inventory = []
    
    for tree_id in unique_ids:
        if tree_id == 0: continue
        mask = point_labels == tree_id
        if np.count_nonzero(mask) > 50:
            sub_las = laspy.LasData(las.header)
            sub_las.points = las.points[mask]
            
            # Local Ground Logic
            tree_x, tree_y = np.mean(sub_las.x), np.mean(sub_las.y)
            r = int((tree_y - y_min) / res); c = int((tree_x - x_min) / res)
            r = max(0, min(r, dtm.shape[0]-1)); c = max(0, min(c, dtm.shape[1]-1))
            local_ground = dtm[r, c]
            
            # Features
            h = np.max(sub_las.z) - local_ground
            dx = np.max(sub_las.x) - np.min(sub_las.x)
            dy = np.max(sub_las.y) - np.min(sub_las.y)
            cd = (dx + dy) / 2
            
            # Save Individual LAZ
            fname = f"{plot_number}_Tree_{tree_id}.laz"
            sub_las.write(str(output_path / fname))
            
            inventory.append({
                "Temp_ID": tree_id,
                "Easting": round(tree_x, 3),   # Eastern Point
                "Northing": round(tree_y, 3),  # Northern Point
                "Tree_Height": round(h, 2),
                "Crown_Diameter": round(cd, 2),
                "Plot_Number": plot_number,
                "Date": survey_date
            })
            
    df = pd.DataFrame(inventory)
    df.to_csv(csv_path, index=False)
    print(f"[Module 1] Extracted {len(df)} trees. Saved to {csv_path}")
    return csv_path