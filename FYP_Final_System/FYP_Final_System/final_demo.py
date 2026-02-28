import sys
import module_1_extraction
import module_2_dbh
import module_3_biomass

def main():
    print("=========================================")
    print("   🌳 AUTOMATED FOREST INVENTORY SYSTEM   ")
    print("       (ULS -> AI -> Biomass)            ")
    print("=========================================")
    
    # 1. User Inputs
    print("\n--- STEP 1: CONFIGURATION ---")
    laz_file = "demo.laz"
    plot_num = input("Enter Plot Number (e.g., BR01): ").strip()
    date_in  = input("Enter Survey Date (YYYY-MM-DD): ").strip()
    
    if not plot_num: plot_num = "BR01"
    if not date_in: date_in = "2023-12-10"
    
    try:
        # 2. Call Module 1
        print("\n--- STEP 2: MODULE 1 (EXTRACTION) ---")
        csv_m1 = module_1_extraction.process_uls_data(
            laz_file, plot_num, date_in, output_dir="Final_Output"
        )
        
        # 3. Call Module 2
        print("\n--- STEP 3: MODULE 2 (DBH PREDICTION) ---")
        csv_m2 = module_2_dbh.predict_dbh(csv_m1)
        
        # 4. Call Module 3
        print("\n--- STEP 4: MODULE 3 (BIOMASS & FINALIZATION) ---")
        final_csv = module_3_biomass.calculate_biomass_final(csv_m2)
        
        print(f"\nSuccessfully completed! Open '{final_csv}' to view results.")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()