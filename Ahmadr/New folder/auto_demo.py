import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import time

# ==========================================
#   CONFIGURATION
# ==========================================
INPUT_FILE = Path("demo.laz")  
OUTPUT_DIR = Path("Auto_Extracted_Results") 
REPORT_FILE = OUTPUT_DIR / "Auto_Inventory.csv"


GRID_SIZE = 0.1          


MIN_TREE_HEIGHT = 3.5  

# 3. Distance (Sab se important)
# 5 par 94 aya, 12 par 19 aya.
# Hum inka average le rahe hain: 7 pixels (0.7 meters)
MIN_DISTANCE_PIXELS = 6


def load_laz(path):
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return None, None
    las = laspy.read(str(path))
    xyz = np.vstack((las.x, las.y, las.z)).T
    return las, xyz

def create_chm(xyz, grid_size):
    """
    Creates a 2D Canopy Height Model (Image) from 3D points.
    Think of this as converting the forest into a 'Height Map'.
    """
    x_min, x_max = xyz[:,0].min(), xyz[:,0].max()
    y_min, y_max = xyz[:,1].min(), xyz[:,1].max()

    cols = int((x_max - x_min) / grid_size) + 1
    rows = int((y_max - y_min) / grid_size) + 1
    
    print(f"   -> Creating CHM Grid: {rows}x{cols} pixels")
    

    chm = np.zeros((rows, cols))
    

    x_idx = ((xyz[:,0] - x_min) / grid_size).astype(int)
    y_idx = ((xyz[:,1] - y_min) / grid_size).astype(int)
    

    df = pd.DataFrame({'r': y_idx, 'c': x_idx, 'z': xyz[:,2]})
    grid_max = df.groupby(['r', 'c'])['z'].max().reset_index()
    

    chm[grid_max['r'], grid_max['c']] = grid_max['z']
    


    ground_level = np.percentile(xyz[:,2], 1)
    chm_normalized = chm - ground_level
    chm_normalized[chm_normalized < 0] = 0
    
    return chm_normalized, x_min, y_min

def calculate_features(x, y, z):
    """Calculates H, CD, Area for extracted trees"""
    height = np.max(z) - np.min(z)
    width_x = np.max(x) - np.min(x)
    width_y = np.max(y) - np.min(y)
    diameter = (width_x + width_y) / 2
    try:
        hull = ConvexHull(np.vstack([x, y]).T)
        area = hull.volume
    except:
        area = 0.0
    return height, diameter, area


if __name__ == "__main__":
    print("### STARTING AUTOMATED (UNSUPERVISED) EXTRACTION ###")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    

    print(f"Loading {INPUT_FILE}...")
    las, xyz = load_laz(INPUT_FILE)
    if las is None: exit()


    print("Generating Canopy Height Model...")
    chm, x_off, y_off = create_chm(xyz, GRID_SIZE)
    

    chm = ndimage.gaussian_filter(chm, sigma=1)


    print("Detecting Tree Tops...")
    local_maxi = peak_local_max(chm, min_distance=MIN_DISTANCE_PIXELS, threshold_abs=MIN_TREE_HEIGHT)
    print(f"   -> Detected {len(local_maxi)} potential trees.")
    

    markers = np.zeros_like(chm, dtype=int)
    for i, (r, c) in enumerate(local_maxi):
        markers[r, c] = i + 1
    

    print("Running Watershed Segmentation...")
    labels = watershed(-chm, markers, mask=chm > MIN_TREE_HEIGHT)


    print("Extracting Point Clouds (This is the heavy part)...")
    

    x_idx = ((xyz[:,0] - x_off) / GRID_SIZE).astype(int)
    y_idx = ((xyz[:,1] - y_off) / GRID_SIZE).astype(int)
    

    max_r, max_c = labels.shape
    valid_mask = (x_idx >= 0) & (x_idx < max_c) & (y_idx >= 0) & (y_idx < max_r)
    

    point_tree_ids = np.zeros(len(xyz), dtype=int)
    point_tree_ids[valid_mask] = labels[y_idx[valid_mask], x_idx[valid_mask]]
    
    unique_ids = np.unique(point_tree_ids)
    inventory = []
    
    for tree_id in unique_ids:
        if tree_id == 0: continue 
        
        mask = point_tree_ids == tree_id
        
        
        if np.count_nonzero(mask) > 50:
            sub_las = laspy.LasData(las.header)
            sub_las.points = las.points[mask]
            
        
            h, cd, area = calculate_features(sub_las.x, sub_las.y, sub_las.z)
            
        
            fname = f"Auto_Tree_{tree_id}.laz"
            sub_las.write(str(OUTPUT_DIR / fname))
            
        
            inventory.append({
                "Tree_ID": f"Tree_{tree_id}",
                "X_Location": round(np.mean(sub_las.x), 2),
                "Y_Location": round(np.mean(sub_las.y), 2),
                "Height_m": round(h, 2),
                "Diameter_m": round(cd, 2),
                "Points": np.count_nonzero(mask)
            })

    
    if inventory:
        df = pd.DataFrame(inventory)
        df.to_csv(REPORT_FILE, index=False)
        print("\n" + "="*40)
        print("       AUTOMATION RESULTS")
        print("="*40)
        print(f"Total Trees Extracted: {len(df)}")
        print(f"Files saved in: {OUTPUT_DIR}")
        print(f"Report saved as: {REPORT_FILE}")
    else:
        print("No trees found.")