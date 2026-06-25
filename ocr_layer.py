#!/usr/bin/env python3
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR globally using GPU lanes
# 'use_angle_cls' handles plates viewed from oblique camera angles
ocr_engine = PaddleOCR(use_gpu=True, lang='en', use_angle_cls=True, show_log=False)

def process_license_plates_ocr(frame_rgba, final_boxes, final_class_ids):
    """
    Crops detected license plates out of the hardware frame buffer 
    and extracts text asynchronously.
    """
    ocr_results = []
    
    # Target Class 2 = Cars/License Plates from your parser output
    for idx, class_id in enumerate(final_class_ids):
        if class_id != 2:
            continue
            
        # Extract pixel coordinates adjusted by the JIT parser
        x1, y1, x2, y2 = final_boxes[idx]
        
        # Guard clause: Ensure boundaries do not overshoot frame size boundaries
        h, w, _ = frame_rgba.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        # Skip invalid or minuscule bounding box crops
        if (x2 - x1) < 20 or (y2 - y1) < 10:
            continue
            
        # Crop the plate from the RGBA frame mapping
        cropped_plate = frame_rgba[y1:y2, x1:x2]
        
        # Convert to Grayscale and optimize contrast for long-distance readability
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_RGBA2GRAY)
        optimized_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Convert back to 3-channel RGB for PaddleOCR parsing standard
        input_img = cv2.cvtColor(optimized_plate, cv2.COLOR_GRAY2RGB)
        
        # Run OCR character recognition
        result = ocr_engine.ocr(input_img, cls=True)
        
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]        # Parsed text string
                confidence = line[1][1]  # Character accuracy score
                
                # Filter noise (e.g., standard plates must have alphanumeric content)
                if confidence > 0.65:
                    cleaned_text = "".join([c for c in text if c.isalnum()]).upper()
                    ocr_results.append({
                        "text": cleaned_text,
                        "confidence": float(confidence),
                        "coords": (x1, y1, x2, y2)
                    })
                    
    return ocr_results
