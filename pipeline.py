#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from numba import njit
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
import yolov8_triton_parser as parser
import ocr_layer as ocr
import blueprint_reader as blueprint  # Import your new module

# System Constants
MUXER_OUTPUT_WIDTH = 3840
MUXER_OUTPUT_HEIGHT = 2160
GPU_ID = 0

import yolov8_triton_parser as parser

# Inside your deepstream pad probe function:
# 1. Extract raw tensor layer array from nvinferserver metadata
raw_array = get_raw_tensor_from_meta(frame_meta) # Shape: [1, 84, 8400]

# 2. Decode matrix instantly with your Numba multi-core parser
boxes, scores, class_ids = parser.triton_output_to_objects(
    raw_array, 
    conf_thresh=0.50, 
    iou_thresh=0.45, 
    num_classes=80
)

# 3. Boxes now contains precise [x1, y1, x2, y2] metrics 
# Feed these directly to your OCR module for license plates or blueprint reading
for i in range(len(boxes)):
    print(f"Detected Class {class_ids[i]} at {boxes[i]} with conf {scores[i]:.2f}")


# --- NUMBA MULTI-CORE JIT OPTIMIZATION ---
@njit(fastmath=True, cache=True)
def analyze_distance_metrics_jit(bbox, tracking_id, class_id):
    """
    Compiled with LLVM to run at raw C-speed on multi-core processors.
    Calculates center anomalies and pixel-scale area dimensions 
    to filter distant objects vs close-up camera blur.
    """
    left, top, width, height = bbox
    area = width * height
    center_x = left + (width / 2.0)
    center_y = top + (height / 2.0)
    
    # Custom distance prioritization weighting algorithm
    priority_score = 0.0
    if class_id == 0:    # Personnel Tracking
        priority_score = area * 1.2
    elif class_id == 2:  # Long-range LPR
        priority_score = area * 2.5
    elif class_id == 7:  # Blueprint Engineering Wall
        priority_score = area * 3.0
        
    return area, center_x, center_y, priority_score

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Unified Production Metadata Probe: Handles JIT optimization,
    NVIDIA hardware frame access, personnel tracking, long-distance LPR,
    and engineering room blueprint matrix evaluation simultaneously.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to grab GstBuffer frame stream.")
        return Gst.PadProbeReturn.OK

    # 1. Retrieve the native DeepStream batch metadata context
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # ------------------------------------------------------------------
        # FEATURE A: ZERO-COPY GPU FRAME BUFFER CAPTURE
        # ------------------------------------------------------------------
        # Map the active hardware memory block directly within GPU VRAM lanes.
        # This allows PaddleOCR to run without copying frames back to CPU RAM.
        try:
            surface_transformer = pyds.get_nvds_Layer_Array(hash(gst_buffer), frame_meta.frame_num)
            frame_rgba = np.array(surface_transformer, copy=False, order='C')
        except Exception as e:
            print(f"Warning: Frame buffer surface mapping skipped: {str(e)}")
            frame_rgba = None

        # ------------------------------------------------------------------
        # FEATURE B: JIT MULTI-CORE MATRIX DECODING & TRITON PARSING
        # ------------------------------------------------------------------
        # Fetch the raw tensor memory array emitted by Triton/nvinfer
        # (Assuming custom user meta tensor attachments or deepstream structures)
        raw_triton_tensor = get_raw_tensor_from_meta(frame_meta) 
        
        # Pass the matrix into the multi-threaded Numba parser compiled with C-speed
        final_boxes, final_scores, final_class_ids = parser.triton_output_to_objects(
            raw_triton_tensor, conf_thresh=0.45, iou_thresh=0.45, num_classes=80
        )

        # ------------------------------------------------------------------
        # FEATURE C: SECURE ASYNCHRONOUS LICENSE PLATE CHARACTER OCR
        # ------------------------------------------------------------------
        if frame_rgba is not None:
            # Process the RGBA frame matrix specifically looking for Class 2 (Plates)
            captured_plates = ocr.process_license_plates_ocr(frame_rgba, final_boxes, final_class_ids)
            
            for plate in captured_plates:
                px1, py1, px2, py2 = plate['coords']
                print(f"[LPR CAPTURE] Plate text: {plate['text']} | Accuracy: {plate['confidence']:.2f}%")
                
                # Dynamic On-Screen Display Text Construction over the car/plate
                txt_params = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_labels = 1
                
                label = display_meta.text_params[0]
                label.display_text = f"PLATE: {plate['text']} ({plate['confidence']*100:.0f}%)"
                label.x_offset = px1
                label.y_offset = max(10, py1 - 25)
                label.font_params.font_name = "Serif"
                label.font_params.font_size = 14
                label.font_params.font_color.set(1.0, 1.0, 1.0, 1.0) # White Text
                label.set_bg_clr = 1
                label.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)            # Semi-transparent black background
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # --------------------------------------------------------------
        # RUN FEATURE 2: WALL BLUEPRINT STRUCTURAL READER
        # --------------------------------------------------------------
        blueprints_read = blueprint.process_engineering_blueprint(frame_rgba, final_boxes, final_class_ids)
        for bp in blueprints_read:
            print(f"\n--- [BLUEPRINT READ SUCCESS: ID {bp['blueprint_id']}] ---")
            for line in bp['data_payload']:
                print(f" > Extracted Label: {line['text']} (Conf: {line['score']:.2f})")
            print("--------------------------------------------------\n")

        # ------------------------------------------------------------------
        # FEATURE D: GENETEC METADATA DISPLAY, COLOR ENCODING & VISUALS
        # ------------------------------------------------------------------
        # Cycle through objects tracked on the frame to apply your UI rule constraints
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Distance Quality Parameter: Reject shaky tracking anchors
            if obj_meta.tracker_confidence > 0.40:
                
                # Package coordinates into an array for quick multi-core math evaluation
                bbox_data = np.array([
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height
                ], dtype=np.float64)
                
                # Execute compiled distance prioritizing logic
                area, cx, cy, priority_score = analyze_distance_metrics_jit(
                    bbox_data, obj_meta.object_id, obj_meta.class_id
                )
                
                # Priority Level 1: Personnel Tracking (Class 0)
                if obj_meta.class_id == 0:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Solid Green
                    obj_meta.text_params.display_text = f"Staff ID:{obj_meta.object_id} [Conf:{obj_meta.tracker_confidence:.2f}]"
                    
                # Priority Level 2: Long-Range Vehicle/Plate Anchors (Class 2)
                elif obj_meta.class_id == 2:
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # Bright Red Border
                    
                # Priority Level 3: Wall Blueprint Reading from Engineering Room (Class 7)
                elif obj_meta.class_id == 7:
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0)  # Deep Blue Border
                    obj_meta.text_params.display_text = f"ENG BLUEPRINT | Scale Area: {area:.0f}px"
                    
                    # Optional: Forward blueprint coordinates to OCR if textual reading is required
                    if frame_rgba is not None:
                        # Extract blueprint text blocks here using structural croppers 
                        pass

                # Print clean, actionable engineering logs for Git repo ingestion
                if priority_score > 6000.0 or obj_meta.class_id == 7:
                    print(f"[METRIC LOG] Frame #{frame_meta.frame_num} | Track ID: {obj_meta.object_id} | "
                          f"Class: {obj_meta.class_id} | Center: ({cx:.0f}, {cy:.0f}) | P-Score: {priority_score:.1f}")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
                
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK




def parse_args():
    parser = argparse.ArgumentParser(description="Genetec + DeepStream JIT Pipeline")
    parser.add_argument("--gpu-type", choices=["rtx6000", "rtx50"], required=True, help="rtx6000 or rtx50 toggle")
    return parser.parse_args()


def main():
    args = parse_args()
    Gst.init(None)
    loop = GLib.MainLoop()

    print(f"Booting JIT Engine for: {args.gpu_type.upper()}")
    pipeline = Gst.Pipeline()

    # Capture Core Infrastructure
    source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
    source.set_property("location", "rtsp://GENETEC_SERVER_IP:554/LiveOS/Cameras/YOUR_CAMERA_ID")
    source.set_property("latency", 200)

    rtppay = Gst.ElementFactory.make("rtph264depay", "depayl")
    parse = Gst.ElementFactory.make("h264parse", "parse1")
    
    nvdec = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    nvdec.set_property("gpu-id", GPU_ID)

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-multiplexer")
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("gpu-id", GPU_ID)

    # Architectural Switch Toggle
    if args.gpu_type == "rtx6000":
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", "config_infer_rtx6000.txt")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
        pgie.set_property("config-file-path", "config_infer_rtx50_triton.txt")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 960)
    tracker.set_property("tracker-height", 544)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "config_tracker_nvdcft.txt")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreen-display")
    nvvideo_convert = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    nvvideo_convert.set_property("gpu-id", GPU_ID)

    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property("sync", 0)

    # Assemble Pipeline
    components = [source, rtppay, parse, nvdec, streammux, pgie, tracker, nvvideo_convert, nvosd, sink]
    for element in components:
        pipeline.add(element)

    def cb_newpad(src, pad, data):
        sink_pad = data.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    source.connect("pad-added", cb_newpad, rtppay)
    rtppay.link(parse)
    parse.link(nvdec)

    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = nvdec.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvideo_convert)
    nvvideo_convert.link(nvosd)
    nvosd.link(sink)

    # Inject metadata probe
    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())
