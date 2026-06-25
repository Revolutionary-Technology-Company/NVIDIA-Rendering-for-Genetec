#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from numba import njit
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

# System Constants
MUXER_OUTPUT_WIDTH = 3840
MUXER_OUTPUT_HEIGHT = 2160
GPU_ID = 0

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
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            if obj_meta.tracker_confidence > 0.4:
                # Structure raw data array for the Numba njit compiler
                bbox_data = np.array([
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height
                ], dtype=np.float64)
                
                # Execute ultra-fast compiled math
                area, cx, cy, p_score = analyze_distance_metrics_jit(
                    bbox_data, obj_meta.object_id, obj_meta.class_id
                )
                
                # Class mapping UI colors
                if obj_meta.class_id == 0:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0) # Green People
                elif obj_meta.class_id == 2:
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0) # Red Plates
                elif obj_meta.class_id == 7:
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0) # Blue Prints

                # Debug analytical streaming logs
                if p_score > 5000.0:
                    print(f"[HIGH PRIORITY] Object {obj_meta.object_id} | Class {obj_meta.class_id} | "
                          f"Center: ({cx:.1f}, {cy:.1f}) | Score: {p_score:.1f}")
            
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
