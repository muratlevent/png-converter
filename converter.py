import cv2
import numpy as np
import pytesseract
import pandas as pd
import re

def preprocess_cell(cell_img):
    """
    Applies image preprocessing to improve Tesseract OCR accuracy.
    Uses the green channel to maximize contrast of red/black text on cyan/blue background,
    upscales, applies Otsu thresholding, and adds a padding border.
    """
    if len(cell_img.shape) == 3:
        # Green channel has the highest contrast for red/black text on light-blue/white backgrounds
        gray = cell_img[:, :, 1]
    else:
        gray = cell_img
        
    # Resize by 2.5x to increase character height for better OCR
    gray = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Add a white border around the cell to help Tesseract
    pad = 10
    padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    
    return padded

def clean_ocr_text(text, col_type):
    """
    Cleans up common Tesseract OCR errors and normalizes text based on column type.
    """
    text = text.strip().replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    if not text:
        return ""
        
    if col_type == 'shape':
        text_upper = text.upper()
        if any(w in text_upper for w in ['RND', 'RNO', 'RNDL', 'RND.', 'RND1']):
            return 'RND'
        if 'OVAL' in text_upper or 'QVAL' in text_upper or '0VAL' in text_upper:
            return 'OVAL'
        if 'EMERALD SQUARE' in text_upper or 'MERALD SQUARE' in text_upper or 'EMERALDSQUARE' in text_upper:
            return 'EMERALD SQUARE'
        if 'EMERALD' in text_upper or 'EMERALO' in text_upper or 'EMER' in text_upper:
            return 'EMERALD'
        if 'RADIANT' in text_upper or 'RAD' in text_upper:
            return 'RADIANT'
        if 'CUSHION' in text_upper or 'CUSH' in text_upper:
            return 'CUSHION'
        if 'PEAR' in text_upper or 'PEA' in text_upper:
            return 'PEAR'
        return text
        
    elif col_type == 'sieve':
        text_upper = text.upper()
        if 'MELE' in text_upper or 'MELE' in text_upper:
            return 'MELE'
        if '+000' in text_upper or '000' in text_upper:
            return '+000'
        if '+00' in text_upper or '00' in text_upper:
            return '+00'
        if any(c in text_upper for c in ['0', 'O', 'Q', 'o', 'U', 'D', 'e', ')', '(']):
            return '0'
        return text
        
    elif col_type == 'gem_type':
        if 'Diamond' in text or 'D' in text or 'diam' in text.lower():
            return 'Diamond'
        return text
        
    elif col_type == 'size':
        cleaned = text.upper().strip()
        cleaned = cleaned.replace('O', '0').replace('I', '1').replace('L', '1')
        cleaned = cleaned.replace('S', '5')
        cleaned = cleaned.replace(',', '.')
        cleaned = cleaned.replace('x', 'X')
        cleaned = re.sub(r'\s*X\s*', 'X', cleaned)
        return cleaned
        
    elif col_type == 'setting_type':
        text_upper = text.upper()
        if 'MICRO' in text_upper or 'PRONG' in text_upper:
            return 'MICRO PRONG'
        if 'COLLET' in text_upper or 'SETTING' in text_upper:
            return 'COLLET SETTING'
        return text
        
    elif col_type in ['qty', 'metal_wt', 'diam_wt', 'total_wt']:
        cleaned = text.upper()
        cleaned = cleaned.replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5').replace('B', '8')
        cleaned = re.sub(r'[^\d\.\,\-]', '', cleaned)
        cleaned = cleaned.replace(',', '.')
        
        if col_type == 'qty':
            if not cleaned:
                return 0
            try:
                if '.' in cleaned:
                    cleaned = cleaned.split('.')[0]
                return int(cleaned)
            except ValueError:
                return 0
        else:
            if not cleaned:
                return 0.0
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
                
    return text

def extract_tables(image_path):
    """
    Detects table cells, segments them into rows, and extracts Diamond Details and Metal Details tables.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    grid_mask = cv2.add(detect_horizontal, detect_vertical)
    
    # Dilate to close gaps
    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_grid = cv2.dilate(grid_mask, dilated_kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter cells
    cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < h < 60 and 20 < w < 400:
            cells.append((x, y, w, h))
            
    # Group cells into rows
    cells.sort(key=lambda item: item[1])
    rows = []
    current_row = []
    prev_y = -999
    
    for cell in cells:
        x, y, w, h = cell
        y_center = y + h // 2
        if prev_y == -999:
            current_row.append(cell)
            prev_y = y_center
        elif abs(y_center - prev_y) < 8:
            current_row.append(cell)
        else:
            current_row.sort(key=lambda item: item[0])
            rows.append(current_row)
            current_row = [cell]
            prev_y = y_center
    if current_row:
        current_row.sort(key=lambda item: item[0])
        rows.append(current_row)
        
    # Standard column centers
    diamond_col_centers = [170, 265, 375, 474, 535, 592, 668, 766]
    metal_col_centers = [859, 947]
    
    diamond_rows_data = []
    metal_rows_data = []
    
    diamond_headers = ["Diam Shape", "Sieve Size", "Gem Type", "Diamond Size", "QTY.", "Diam. Wt", "Total Wt", "Setting Type", "Other"]
    metal_headers = ["Metal Colore", "Wt"]
    
    margin = 2
    
    # --- PASS 1: Extract Diamond Details Table ---
    for r_idx, row in enumerate(rows):
        avg_y = sum(c[1] for c in row) / len(row)
        if avg_y < 480:
            continue
            
        left_cells = [c for c in row if c[0] < 800]
        if not left_cells:
            continue
            
        # Find if sieve cell exists (cx ≈ 170)
        sieve_cell = None
        for cell in left_cells:
            cx = cell[0] + cell[2] // 2
            if abs(cx - 170) < 20:
                sieve_cell = cell
                break
                
        if sieve_cell:
            sy, sh = sieve_cell[1], sieve_cell[3]
            # Artificial Shape crop from x in [19, 118]
            ax, ay, aw, ah = 19, sy, 99, sh
            
            crop_shape = img[ay+margin:ay+ah-margin, ax+margin:ax+aw-margin]
            shape_txt = ""
            if crop_shape.size > 0:
                prep_shape = preprocess_cell(crop_shape)
                shape_txt = pytesseract.image_to_string(prep_shape, config='--psm 6').strip()
                shape_txt = clean_ocr_text(shape_txt, 'shape')
                
            row_data = ["" for _ in range(9)]
            row_data[0] = shape_txt
            
            for cell in left_cells:
                x, y, w, h = cell
                cx = x + w // 2
                
                dists = [abs(cx - center) for center in diamond_col_centers]
                min_dist = min(dists)
                if min_dist < 25:
                    col_idx = dists.index(min_dist) + 1
                    crop_cell = img[y+margin:y+h-margin, x+margin:x+w-margin]
                    if crop_cell.size > 0:
                        prep_cell = preprocess_cell(crop_cell)
                        cell_txt = pytesseract.image_to_string(prep_cell, config='--psm 6').strip()
                        
                        if col_idx == 1:
                            cell_txt = clean_ocr_text(cell_txt, 'sieve')
                        elif col_idx == 2:
                            cell_txt = clean_ocr_text(cell_txt, 'gem_type')
                        elif col_idx == 3:
                            cell_txt = clean_ocr_text(cell_txt, 'size')
                        elif col_idx == 4:
                            cell_txt = clean_ocr_text(cell_txt, 'qty')
                        elif col_idx == 5:
                            cell_txt = clean_ocr_text(cell_txt, 'diam_wt')
                        elif col_idx == 6:
                            cell_txt = clean_ocr_text(cell_txt, 'total_wt')
                        elif col_idx == 7:
                            cell_txt = clean_ocr_text(cell_txt, 'setting_type')
                        else:
                            cell_txt = cell_txt.replace('\n', ' ')
                            
                        row_data[col_idx] = cell_txt
                        
            # Skip Diamond Details header row
            if row_data[1] == "Sieve Size":
                continue
                
            diamond_rows_data.append(row_data)
            
        else:
            # Check if this is the TOTAL row of Diamond Details
            is_total_row = False
            for cell in left_cells:
                x, y, w, h = cell
                crop_cell = img[y+margin:y+h-margin, x+margin:x+w-margin]
                if crop_cell.size > 0:
                    prep = preprocess_cell(crop_cell)
                    txt = pytesseract.image_to_string(prep, config='--psm 6').strip().upper()
                    # Add JIAL, IAL, TAL, TOT to catch clipped TOTAL OCR
                    if any(term in txt for term in ["TOTAL", "TWITAL", "TWTAL", "TOAL", "JIAL", "IAL", "TAL", "TOT"]):
                        is_total_row = True
                        break
                        
            if is_total_row:
                row_data = ["TOTAL :", "", "", "", 0, 0.0, 0.0, "", ""]
                for cell in left_cells:
                    x, y, w, h = cell
                    cx = x + w // 2
                    dists = [abs(cx - center) for center in diamond_col_centers]
                    min_dist = min(dists)
                    if min_dist < 25:
                        col_idx = dists.index(min_dist) + 1
                        crop_cell = img[y+margin:y+h-margin, x+margin:x+w-margin]
                        if crop_cell.size > 0:
                            prep = preprocess_cell(crop_cell)
                            txt = pytesseract.image_to_string(prep, config='--psm 6').strip()
                            if col_idx == 4:
                                row_data[col_idx] = clean_ocr_text(txt, 'qty')
                            elif col_idx == 5:
                                row_data[col_idx] = clean_ocr_text(txt, 'diam_wt')
                            elif col_idx == 6:
                                row_data[col_idx] = clean_ocr_text(txt, 'total_wt')
                diamond_rows_data.append(row_data)
                
    # --- PASS 2: Extract Metal Details Table ---
    for r_idx, row in enumerate(rows):
        avg_y = sum(c[1] for c in row) / len(row)
        if avg_y < 480:
            continue
            
        right_cells = [c for c in row if 800 <= c[0] < 980]
        if not right_cells:
            continue
            
        metal_row_data = ["" for _ in range(2)]
        has_metal_data = False
        
        for cell in right_cells:
            x, y, w, h = cell
            cx = x + w // 2
            dists = [abs(cx - center) for center in metal_col_centers]
            min_dist = min(dists)
            if min_dist < 25:
                col_idx = dists.index(min_dist)
                crop_cell = img[y+margin:y+h-margin, x+margin:x+w-margin]
                if crop_cell.size > 0:
                    prep_cell = preprocess_cell(crop_cell)
                    cell_txt = pytesseract.image_to_string(prep_cell, config='--psm 6').strip()
                    
                    if col_idx == 0:
                        cell_txt = cell_txt.replace('\n', ' ')
                    else:
                        cell_txt = clean_ocr_text(cell_txt, 'metal_wt')
                        
                    metal_row_data[col_idx] = cell_txt
                    has_metal_data = True
                    
        if has_metal_data:
            # Skip Metal table header
            if metal_row_data[0] == "Metal Colore":
                continue
            # Skip if both are empty
            if not metal_row_data[0] and not metal_row_data[1]:
                continue
            metal_rows_data.append(metal_row_data)
            
    # Create DataFrames
    df_diamond = pd.DataFrame(diamond_rows_data, columns=diamond_headers)
    df_metal = pd.DataFrame(metal_rows_data, columns=metal_headers)
    
    # --- POST-PROCESSING AND CLEANING ---
    # Filter empty rows
    df_diamond = df_diamond[df_diamond["Diam Shape"].str.strip() != ""]
    df_metal = df_metal[df_metal["Metal Colore"].str.strip() != ""]
    
    # 1. Clean up Diamond Details
    # Convert numerical columns to correct types
    df_diamond["QTY."] = pd.to_numeric(df_diamond["QTY."], errors='coerce').fillna(0).astype(int)
    df_diamond["Diam. Wt"] = pd.to_numeric(df_diamond["Diam. Wt"], errors='coerce').fillna(0.0).astype(float)
    df_diamond["Total Wt"] = pd.to_numeric(df_diamond["Total Wt"], errors='coerce').fillna(0.0).astype(float)
    
    # Apply physical constraints (e.g. if Total Wt is 0, QTY. is 0, Diam. Wt is 0.0)
    for idx, row in df_diamond.iterrows():
        if row["Diam Shape"] != "TOTAL :":
            # If Total Wt is 0.0, zero out Qty and Diam. Wt
            if row["Total Wt"] == 0.0:
                df_diamond.at[idx, "QTY."] = 0
                df_diamond.at[idx, "Diam. Wt"] = 0.0
                df_diamond.at[idx, "Diamond Size"] = ""
                
    # Clean up Metal Details (remove noise, standardize types)
    valid_metals = ["Sterling", "Kt", "Platinum", "Wax"]
    df_metal_filtered = []
    for idx, row in df_metal.iterrows():
        metal = row["Metal Colore"]
        if any(m in metal for m in valid_metals):
            # Clean metal name
            if "Sterling" in metal:
                metal = "Sterling 925"
            elif "Fine Gold" in metal or "24 Kt" in metal:
                metal = "24 Kt Fine Gold"
            df_metal_filtered.append([metal, row["Wt"]])
            
    df_metal = pd.DataFrame(df_metal_filtered, columns=metal_headers)
    df_metal["Wt"] = pd.to_numeric(df_metal["Wt"], errors='coerce').fillna(0.0).astype(float)
    
    # Sort data row outputs to ensure consistency
    # Diamond table: keep total row at bottom
    data_rows = df_diamond[df_diamond["Diam Shape"] != "TOTAL :"].copy()
    total_row = df_diamond[df_diamond["Diam Shape"] == "TOTAL :"]
    df_diamond = pd.concat([data_rows, total_row], ignore_index=True)
    
    return df_diamond, df_metal

if __name__ == '__main__':
    try:
        df_dia, df_met = extract_tables('example.jpg')
        print("\n=== DIAMOND DETAILS ===")
        print(df_dia.to_string(index=False))
        print("\n=== METAL DETAILS ===")
        print(df_met.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
