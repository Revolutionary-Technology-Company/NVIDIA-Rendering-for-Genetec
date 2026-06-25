#!/usr/bin/env python3
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR with specific angle classification for vertically/horizontally flipped technical blueprints
blueprint_ocr = PaddleOCR(use_gpu=True, lang='en', structure=True, show_log=False)

def process_engineering_blueprint(frame_rgba, final_boxes, final_class_ids):
    """
    Isolates Class 7 technical wall blueprints, divides the bounding box 
    into structural text fragments, and reads high-fidelity engineering dimensions.
    """
    blueprint_records = []
    
    for idx, class_id in enumerate(final_class_ids):
        # Filter strictly for Class 7 (Technical Wall Blueprints / Engineering Documents)
        if class_id != 7:
            continue
            
        x1, y1, x2, y2 = final_boxes[idx]
        h, w, _ = frame_rgba.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        # Guard clause for minimal surface scanning size
        if (x2 - x1) < 100 or (y2 - y1) < 100:
            continue
            
        # 1. Capture and isolate the blueprint canvas from GPU frame mapping
        cropped_blueprint = frame_rgba[y1:y2, x1:x2]
        
        # 2. Image Optimization: Engineering prints often have high contrast noise
        gray = cv2.cvtColor(cropped_blueprint, cv2.COLOR_RGBA2GRAY)
        
        # Adaptive thresholding preserves thin architectural lines and text labels
        binary_blueprint = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3-channel layout required by the structured text runner
        input_canvas = cv2.cvtColor(binary_blueprint, cv2.COLOR_GRAY2RGB)
        
        # 3. Structural layout parsing to extract text blocks, omitting structural diagrams
        # This isolates lines from alpha-numeric data
        structure_results = blueprint_ocr(input_canvas, cls=True)
        
        extracted_lines = []
        if structure_results and structure_results[0]:
            for block in structure_results[0]:
                text_content = block[0]  # Extracted text data string
                confidence = block[1]    # Extraction reliability score
                
                if confidence > 0.55:
                    extracted_lines.append({
                        "text": text_content.strip(),
                        "score": float(confidence)
                    })
                    
        if extracted_lines:
            blueprint_records.append({
                "blueprint_id": idx,
                "global_coords": (x1, y1, x2, y2),
                "data_payload": extracted_lines
            })
            
    return blueprint_records
