#!/usr/bin/env python3
import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=True, cache=True)
def parse_yolov8_raw_output_jit(raw_output, conf_threshold, num_classes):
    """
    Compiled JIT function utilizing parallel multi-core slicing.
    Transforms raw shape [1, 84, 8400] -> Filtered candidates arrays.
    
    YOLOv8 Output format per column:
    [0:4]  -> xc, yc, w, h (Bounding Box Centers & Dimensions)
    [4:84] -> Class scores for each individual object category
    """
    # Remove batch dimension: shape becomes [84, 8400]
    data = raw_output[0]
    num_anchors = data.shape[1]
    
    # Pre-allocate worst-case tracking arrays
    valid_boxes = np.zeros((num_anchors, 4), dtype=np.float64)
    valid_scores = np.zeros(num_anchors, dtype=np.float64)
    valid_class_ids = np.zeros(num_anchors, dtype=np.int32)
    
    count = 0
    
    # Multi-threaded loop over thousands of candidate anchor matrices
    for i in prange(num_anchors):
        # Extract class slice across row vectors 4 to end
        class_scores = data[4:, i]
        
        # Find peak class confidence anchor
        max_score = 0.0
        class_id = 0
        for c in range(num_classes):
            if class_scores[c] > max_score:
                max_score = class_scores[c]
                class_id = c
                
        # Filter against your distance-quality confidence parameters
        if max_score >= conf_threshold:
            # Thread-safe increment or structural assignment
            # Convert center format (xc, yc, w, h) to corner format (x1, y1, x2, y2)
            xc = data[0, i]
            yc = data[1, i]
            w  = data[2, i]
            h  = data[3, i]
            
            x1 = xc - (w / 2.0)
            y1 = yc - (h / 2.0)
            x2 = xc + (w / 2.0)
            y2 = yc + (h / 2.0)
            
            # Local critical assignment
            valid_boxes[i] = [x1, y1, x2, y2]
            valid_scores[i] = max_score
            valid_class_ids[i] = class_id

    # Clean out empty slots from allocation pool
    # Non-zero evaluation filtering
    active_indices = np.where(valid_scores > 0.0)[0]
    
    cleaned_boxes = np.zeros((len(active_indices), 4), dtype=np.float64)
    cleaned_scores = np.zeros(len(active_indices), dtype=np.float64)
    cleaned_class_ids = np.zeros(len(active_indices), dtype=np.int32)
    
    for idx, real_pos in enumerate(active_indices):
        cleaned_boxes[idx] = valid_boxes[real_pos]
        cleaned_scores[idx] = valid_scores[real_pos]
        cleaned_class_ids[idx] = valid_class_ids[real_pos]
        
    return cleaned_boxes, cleaned_scores, cleaned_class_ids


@njit(fastmath=True, cache=True)
def nms_fastmath_jit(boxes, scores, iou_threshold):
    """
    Highly parallelized Non-Maximum Suppression algorithm.
    Removes overlapping bounding boxes to keep only the highest confidence match.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int32)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return np.array(keep, dtype=np.int32)


def triton_output_to_objects(raw_triton_tensor, conf_thresh=0.45, iou_thresh=0.5, num_classes=80):
    """
    Primary wrapper called by DeepStream custom metadata structures or 
    standalone Triton clients.
    """
    # 1. Run quickmath matrix compiler across CPU cores
    boxes, scores, class_ids = parse_yolov8_raw_output_jit(raw_triton_tensor, conf_thresh, num_classes)
    
    # 2. Filter redundant overlays via JIT NMS
    keep_indices = nms_fastmath_jit(boxes, scores, iou_thresh)
    
    # 3. Compile clean arrays
    final_boxes = boxes[keep_indices]
    final_scores = scores[keep_indices]
    final_class_ids = class_ids[keep_indices]
    
    return final_boxes, final_scores, final_class_ids
