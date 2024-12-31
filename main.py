from fastapi import FastAPI, File, UploadFile, HTTPException
from omniparser import Omniparser  # Adjust based on the OmniParser library
from typing import Dict
import tempfile
import os
import numpy as np

app = FastAPI()
config = {
    'som_model_path': 'weights/icon_detect/best.pt',
    'device': 'cpu',
    'caption_model_path': 'microsoft/Florence-2-base',
    'caption_model_name': 'florence2',
    'draw_bbox_config': {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    },
    'BOX_TRESHOLD': 0.05
}
parser = Omniparser(config)  # Initialize OmniParser (adjust if necessary)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert scalars to Python scalars
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(value) for value in obj]
    return obj

@app.post("/parse-screenshot")
async def parse_document(file: UploadFile = File(...)) -> Dict:
    """
    Parse a document file using OmniParser.
    """
    print("Received request")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read the uploaded file and parse it using OmniParser
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Pass the temporary file path to the parser
        parsed_data = parser.parse(temp_file_path)
        clean_data = convert_numpy_types(parsed_data)

        # Clean up the temporary file after parsing
        os.remove(temp_file_path)
        return {"parsed_data": clean_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

