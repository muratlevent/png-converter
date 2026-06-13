import argparse
import os
import sys
import pandas as pd
from converter import extract_tables

def format_excel_workbook(xlsx_path):
    """
    Auto-adjusts Excel column widths to fit the text beautifully.
    """
    from openpyxl import load_workbook
    try:
        wb = load_workbook(xlsx_path)
        for sheet in wb.worksheets:
            for col in sheet.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                # Set width with some padding
                sheet.column_dimensions[col_letter].width = max(max_len + 3, 11)
        wb.save(xlsx_path)
    except Exception as e:
        print(f"Warning: Could not format Excel column widths: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract Diamond Details and Metal Details from jewelry design sheet images.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image file (PNG/JPG).")
    parser.add_argument("-o", "--output", default="output", help="Base name for the output files (default: output).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        sys.exit(1)
        
    print(f"Processing image: {args.image} ...")
    try:
        df_dia, df_met = extract_tables(args.image)
    except Exception as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)
        
    # Build output paths
    csv_dia_path = f"{args.output}_diamond.csv"
    csv_met_path = f"{args.output}_metal.csv"
    xlsx_path = f"{args.output}.xlsx"
    
    print("\nExporting data...")
    try:
        # Save CSV files
        df_dia.to_csv(csv_dia_path, index=False)
        df_met.to_csv(csv_met_path, index=False)
        print(f"  Saved Diamond Details CSV to: {csv_dia_path}")
        print(f"  Saved Metal Details CSV to: {csv_met_path}")
        
        # Save Excel file
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df_dia.to_excel(writer, sheet_name='Diamond Details', index=False)
            df_met.to_excel(writer, sheet_name='Metal Details', index=False)
            
        # Format Excel column widths
        format_excel_workbook(xlsx_path)
        print(f"  Saved Excel workbook to: {xlsx_path}")
        
    except Exception as e:
        print(f"Error saving output files: {e}")
        sys.exit(1)
        
    print("\n=== DIAMOND DETAILS SUMMARY ===")
    print(df_dia.to_string(index=False))
    print("\n=== METAL DETAILS SUMMARY ===")
    print(df_met.to_string(index=False))
    print("\nProcessing completed successfully!")

if __name__ == '__main__':
    main()
