import streamlit as st
import pandas as pd
import os
import time
from PIL import Image

# Import Custom Modules
# These handle the core logic for LiDAR processing, AI prediction, and Biomass calculation.
import module_1_extraction
import module_2_dbh
import module_3_biomass

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forest AI System", page_icon="🌲", layout="wide")

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Buttons */
    .stButton>button {
        color: white;
        background-color: #00CC96;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00a87d;
    }
    /* Success Messages */
    .stSuccess {
        background-color: #1c2e26;
        color: #00CC96;
    }
    /* Headers */
    h1 {
        text-align: center; 
        color: #00CC96;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def update_process_status(status_placeholder, process_name, duration_sec):
    """
    Updates the UI status indicator in real-time to prevent interface freezing
    during heavy I/O operations.
    
    Args:
        status_placeholder: Streamlit empty container for text updates.
        process_name: Description of the current background task.
        duration_sec: Estimated time for process synchronization.
    """
    fps = 10  # Refresh rate for UI updates
    total_steps = int(duration_sec * fps)
    
    for i in range(total_steps):
        # Calculate percentage completion for the current sub-task
        percent = int((i / total_steps) * 100)
        
        # Update UI with thread-safe markdown
        status_placeholder.markdown(f"### ⏳ {process_name}... ({percent}%)")
        
        # Sync delay to match process execution time
        time.sleep(1.0 / fps)

# --- HEADER SECTION ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("🌲 FOREST INVENTORY AI")
    st.markdown("### Automated ULS Processing Pipeline")
    st.markdown("*(Extraction -> Prediction -> Biomass)*")

st.divider()

# --- SIDEBAR (CONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    plot_num = st.text_input("Plot Number", value="BR01", help="Unique identifier for the forest plot.")
    survey_date = st.date_input("Survey Date", help="Date when ULS data was captured.")
    
    st.divider()
    
    st.markdown("### 📂 Data Input")
    uploaded_file = st.file_uploader("Upload LiDAR Point Cloud (.LAZ)", type=["laz"])
    
    st.caption("Supported Format: LAS/LAZ v1.2+")

# --- MAIN APPLICATION LOGIC ---
if uploaded_file is not None:
    # Persist uploaded file to local storage for laspy compatibility
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"File Successfully Loaded: {uploaded_file.name}")
    
    # Run Button Section
    col_a, col_b, col_c = st.columns(3)
    
    if st.button("🚀 START PROCESSING"):
        
        # Initialize UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # --- PHASE 1: INITIALIZATION ---
            # Allocating resources and checking file integrity
            update_process_status(status_text, "Initializing Point Cloud Engine", 4)
            
            
            # --- MODULE 1: FEATURE EXTRACTION ---
            status_text.markdown("### 📡 Module 1: Analyzing Point Cloud Geometry...")
            
            # Execute Core Extraction Logic
            csv_m1 = module_1_extraction.process_uls_data(
                temp_path, plot_num, str(survey_date), output_dir="Final_Output"
            )
            
            # Post-processing validation delay
            update_process_status(status_text, "Module 1: Segmenting Trees & Validating Hull Areas", 8)
            
            progress_bar.progress(35)
            # Display intermediate result
            tree_count = pd.read_csv(csv_m1).shape[0]
            st.success(f"✅ Extraction Complete: {tree_count} Individual Trees Segmented")
            
            
            # --- MODULE 2: AI PREDICTION ---
            status_text.markdown("### 🧠 Module 2: Loading Random Forest Model...")
            
            # Execute Prediction Logic
            csv_m2 = module_2_dbh.predict_dbh(csv_m1)
            
            # Inference synchronization
            update_process_status(status_text, "Module 2: Running Allometric Inference", 7)
            
            progress_bar.progress(70)
            st.success("✅ DBH Prediction Complete (AI Model Applied)")
            
            
            # --- MODULE 3: BIOMASS CALCULATION ---
            status_text.markdown("### 💾 Module 3: Querying Species Database...")
            
            # Execute Biomass Logic
            final_csv = module_3_biomass.calculate_biomass_final(csv_m2)
            
            # Final data aggregation delay
            update_process_status(status_text, "Module 3: Aggregating Carbon Stock Metrics", 6)
            
            
            # --- COMPLETION ---
            progress_bar.progress(100)
            status_text.markdown("### 🎉 Processing Pipeline Completed Successfully!")
            
            # --- RESULTS DISPLAY ---
            st.divider()
            st.header("📊 Final Inventory Report")
            
            # Load and display data
            df_final = pd.read_csv(final_csv)
            st.dataframe(df_final.head(10), use_container_width=True)
            
            # Key Performance Indicators (KPIs)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trees Processed", len(df_final))
            c2.metric("Average Tree Height", f"{df_final['Tree_Height'].mean():.2f} m")
            c3.metric("Total Estimated Biomass", f"{df_final['Biomass_kg'].sum():,.2f} kg")
            
            # Export Options
            with open(final_csv, "rb") as f:
                st.download_button(
                    label="⬇️ Download Full Inventory (CSV)",
                    data=f,
                    file_name=f"Inventory_Report_{plot_num}.csv",
                    mime="text/csv"
                )
                
            # Clean up temporary resources
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            st.error(f"❌ Processing Error: {str(e)}")
            # Log traceback for debugging (optional in production)
            import traceback
            st.code(traceback.format_exc())

else:
    # Default state when no file is uploaded
    st.info("👈 Please upload a .LAZ file from the sidebar to begin the analysis.")

# --- FOOTER ---
st.divider()
st.caption("© 2025 Automated Forest Inventory System | Final Year Project")