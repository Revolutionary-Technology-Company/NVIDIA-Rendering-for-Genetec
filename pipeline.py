#!/usr/bin/env python3
import sys
import argparse
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

# Target resolution matching your high-quality Genetec 4K stream
MUXER_OUTPUT_WIDTH = 3840
MUXER_OUTPUT_HEIGHT = 2160
GPU_ID = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Genetec + DeepStream Unified Architecture Pipeline")
    parser.add_argument(
        "--gpu-type", 
        choices=["rtx6000", "rtx50"], 
        required=True, 
        help="Toggle hardware profile: 'rtx6000' (Ada Lovelace Native) or 'rtx50' (Blackwell Triton Backend)"
    )
    return parser.parse_args()

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
                # Class assignment based on training metadata maps (0=People, 2=Plates, 7=Blueprints)
                if obj_meta.class_id == 0:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0) # Green for People
                elif obj_meta.class_id == 2:
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0) # Red for Plates
                elif obj_meta.class_id == 7:
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0) # Blue for Blueprints

                print(f"Frame #{frame_meta.frame_num} | ID: {obj_meta.object_id} | Class: {obj_meta.class_id}")
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream reached.\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err.message}: {debug}\n")
        loop.quit()
    return True

def main():
    args = parse_args()
    Gst.init(None)
    loop = GLib.MainLoop()

    print(f"Configuring pipeline toggle for architecture profile: {args.gpu-type.upper()}")
    pipeline = Gst.Pipeline()

    # Core Capture Components
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

    # --- DYNAMIC TOGGLE SWITCH LOGIC ---
    if args.gpu_type == "rtx6000":
        # Native DeepStream Engine (Optimized for Ada Lovelace 48GB Enterprise)
        print("Loading Native nvinfer element for RTX 6000 Pro...")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", "config_infer_rtx6000.txt")
    else:
        # Triton Server Engine (Bypasses SM errors on Blackwell RTX 50-series)
        print("Loading nvinferserver Triton element for ASUS TUF RTX 50...")
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
        pgie.set_property("config-file-path", "config_infer_rtx50_triton.txt")
    # -----------------------------------

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

    # Link Pipeline Elements
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

    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Pipeline running successfully...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())
