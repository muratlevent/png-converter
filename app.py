import io
import uuid
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template_string
from converter import extract_tables

app = Flask(__name__)

# Cache to store binary files in memory (uuid -> file_data)
# file_data contains: 'xlsx', 'diamond_csv', 'metal_csv'
conversion_cache = {}

def format_excel_in_memory(xlsx_bytes):
    """
    Applies column width auto-formatting to an in-memory Excel workbook.
    """
    from openpyxl import load_workbook
    try:
        wb = load_workbook(io.BytesIO(xlsx_bytes))
        for sheet in wb.worksheets:
            for col in sheet.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                sheet.column_dimensions[col_letter].width = max(max_len + 3, 11)
        out_io = io.BytesIO()
        wb.save(out_io)
        return out_io.getvalue()
    except Exception as e:
        print(f"Warning: Could not format in-memory Excel: {e}")
        return xlsx_bytes

# We can serve index.html directly from a templates folder or as a string.
# Since we will create a dedicated index.html in the templates folder, 
# let's serve it using standard render_template or open and read.
@app.route('/')
def home():
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return jsonify({"error": "templates/index.html not found. Please wait until UI is created."}), 404

@app.route('/api/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read file bytes into OpenCV image array in memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400
            
        # Run extraction
        df_dia, df_met = extract_tables(img)
        
        # Save Excel workbook to memory
        xlsx_io = io.BytesIO()
        with pd.ExcelWriter(xlsx_io, engine='openpyxl') as writer:
            df_dia.to_excel(writer, sheet_name='Diamond Details', index=False)
            df_met.to_excel(writer, sheet_name='Metal Details', index=False)
        xlsx_bytes = format_excel_in_memory(xlsx_io.getvalue())
        
        # Save CSVs to memory
        csv_dia = df_dia.to_csv(index=False).encode('utf-8')
        csv_met = df_met.to_csv(index=False).encode('utf-8')
        
        # Generate cache task ID
        task_id = str(uuid.uuid4())
        conversion_cache[task_id] = {
            'xlsx': xlsx_bytes,
            'diamond_csv': csv_dia,
            'metal_csv': csv_met
        }
        
        # Convert DataFrames to JSON for UI preview
        dia_list = df_dia.to_dict(orient='records')
        met_list = df_met.to_dict(orient='records')
        
        return jsonify({
            "task_id": task_id,
            "diamond_data": dia_list,
            "metal_data": met_list
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500

@app.route('/api/download/<task_id>/<file_format>', methods=['GET'])
def download(task_id, file_format):
    if task_id not in conversion_cache:
        return jsonify({"error": "Session expired or invalid task ID"}), 404
        
    cached_data = conversion_cache[task_id]
    
    if file_format == 'xlsx':
        return send_file(
            io.BytesIO(cached_data['xlsx']),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="jewelry_design_data.xlsx"
        )
    elif file_format == 'diamond_csv':
        return send_file(
            io.BytesIO(cached_data['diamond_csv']),
            mimetype="text/csv",
            as_attachment=True,
            download_name="diamond_details.csv"
        )
    elif file_format == 'metal_csv':
        return send_file(
            io.BytesIO(cached_data['metal_csv']),
            mimetype="text/csv",
            as_attachment=True,
            download_name="metal_details.csv"
        )
    else:
        return jsonify({"error": "Invalid format"}), 400

import os

if __name__ == '__main__':
    # Render and Railway pass the port dynamically via the PORT environment variable
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
