import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_load import process_data
from geodiscovery import discover_locations_and_relations
from geoparsing import process_geoparsing
from analytics_consolidado import run_complete_analysis

def main():
    parser = argparse.ArgumentParser(
        description='SIMIEC Pipeline: Complete data processing and analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --start-date 2024-11-05 --end-date 2024-12-05
  python main.py --start-date 2024-11-01
  python main.py --skip-discovery
  python main.py --only-analysis --start-date 2024-11-05 --end-date 2024-12-05
        """
    )
    
    parser.add_argument('--input-file', type=str, default='data/result.json',
                        help='Input JSON file path (default: data/result.json)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for analysis filtering (format: YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for analysis filtering (format: YYYY-MM-DD)')
    
    parser.add_argument('--skip-clean', action='store_true',
                        help='Skip data cleaning step (use existing datos_rescatados.csv)')
    
    parser.add_argument('--skip-discovery', action='store_true',
                        help='Skip location discovery step (use existing candidatos_lugares.csv)')
    
    parser.add_argument('--skip-geoparsing', action='store_true',
                        help='Skip geoparsing step (use existing datos_georeferenciados.csv)')
    
    parser.add_argument('--only-analysis', action='store_true',
                        help='Only run analysis (skip all processing steps)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 SIMIEC PIPELINE - Starting Complete Processing")
    print("="*70 + "\n")
    
    if args.only_analysis:
        print("📊 Running analysis only...")
        run_complete_analysis(start_date=args.start_date, end_date=args.end_date)
        return
    
    if not args.skip_clean:
        print("\n" + "-"*70)
        print("STEP 1: Data Cleaning and Filtering")
        print("-"*70)
        df_clean = process_data(input_file=args.input_file)
        if df_clean is None:
            print("❌ Pipeline stopped: Data cleaning failed.")
            return
    else:
        print("\n⏭️  Skipping data cleaning step...")
    
    if not args.skip_discovery:
        print("\n" + "-"*70)
        print("STEP 2: Location Discovery")
        print("-"*70)
        df_candidates = discover_locations_and_relations()
        if df_candidates is None:
            print("❌ Pipeline stopped: Location discovery failed.")
            return
    else:
        print("\n⏭️  Skipping location discovery step...")
    
    if not args.skip_geoparsing:
        print("\n" + "-"*70)
        print("STEP 3: Geoparsing and Relationship Generation")
        print("-"*70)
        df_geo, df_rel, df_block_rel = process_geoparsing()
        if df_geo is None:
            print("❌ Pipeline stopped: Geoparsing failed.")
            return
    else:
        print("\n⏭️  Skipping geoparsing step...")
    
    print("\n" + "-"*70)
    print("STEP 4: Statistical Analysis and Visualization")
    print("-"*70)
    run_complete_analysis(start_date=args.start_date, end_date=args.end_date)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved in:")
    print(f"  - CSV files: results/")
    print(f"  - Figures: results/figures/")
    print()

if __name__ == "__main__":
    main()

