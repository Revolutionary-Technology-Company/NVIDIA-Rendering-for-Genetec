#!/usr/bin/env tuple Python
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

# Target resolution matching your high-quality Genetec 4K stream
MUXER_OUTPUT_WIDTH = 3840
MUXER_OUTPUT_HEIGHT = 2160
GPU_ID = 0

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Metadata probe tracking objects at a distance and extracting metrics 
    for license plate parsing and blueprint readers.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Retrieve DeepStream batch metadata
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
            
            # Distance and Quality optimization: Filter out poor tracking scores
            if obj_meta.tracker_confidence > 0.4:
                # Class mapping based on your primary YOLO engine configuration
                # 0 = Person, 2 = Car/Plate, 7 = Document/Blueprint
                if obj_meta.class_id == 0:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0) # Green for People
                elif obj_meta.class_id == 2:
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0) # Red for Plates
                elif obj_meta.class_id == 7:
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0) # Blue for Blueprints

                # Extracted metadata printout for downstream API / RTC repository ingestion
                print(f"Frame #{frame_meta.frame_num} | ID: {obj_meta.object_id} | "
                      f"Class: {obj_meta.class_id} | Conf: {obj_meta.tracker_confidence:.2f} | "
                      f"BBox: L:{obj_meta.rect_params.left:.0f} T:{obj_meta.rect_params.top:.0f}")
            
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
    Gst.init(None)
    loop = GLib.MainLoop()

    print("Initializing DeepStream Pipeline for RTX 6000 Ada...")
    pipeline = Gst.Pipeline()

    # 1. Source element: RTSP from Genetec Media Gateway
    source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
    # Paste your Genetec Media Gateway live stream URL here
    source.set_property("location", "rtsp://GENETEC_SERVER_IP:554/LiveOS/Cameras/YOUR_CAMERA_ID")
    source.set_property("latency", 200)

    # 2. Parsing and Decoding elements
    rtppay = Gst.ElementFactory.make("rtph264depay", "depayl")
    parse = Gst.ElementFactory.make("h264parse", "parse1")
    
    # NVDEC hardware decoder leveraging RTX 6000 Ada silicon
    nvdec = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    nvdec.set_property("gpu-id", GPU_ID)

    # 3. Batch Stream Multiplexer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-multiplexer")
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("gpu-id", GPU_ID)

    # 4. Primary Detector (YOLOv8x Engine File configured for high-res distance inference)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "config_infer_yolov8.txt")

    # 5. NVIDIA Tracker (NVTracker) for persistent people tracking
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 960)
    tracker.set_property("tracker-height", 544)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "config_tracker_nvdcft.txt")

    # 6. On-Screen Display & Visualizer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreen-display")
    nvvideo_convert = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    nvvideo_convert.set_property("gpu-id", GPU_ID)

    # 7. Render Sink for Engineering Room Monitor
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property("sync", 0)

    # Add elements to pipeline
    components = [source, rtppay, parse, nvdec, streammux, pgie, tracker, nvvideo_convert, nvosd, sink]
    for element in components:
        if not element:
            sys.stderr.write(f"Failed to create element {element}\n")
            sys.exit(1)
        pipeline.add(element)

    # Dynamic Pad Linking for rtspsrc
    def cb_newpad(src, pad, data):
        sink_pad = data.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    source.connect("pad-added", cb_newpad, rtppay)

    # Link standard elements static paths
    rtppay.link(parse)
    parse.link(nvdec)

    # Link Decoder to Streammuxer Pad
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = nvdec.get_static_pad("src")
    srcpad.link(sinkpad)

    # Link the rest of the stream chain
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvideo_convert)
    nvvideo_convert.link(nvosd)
    nvosd.link(sink)

    # Attach the Metadata Probe to the OSD sink pad to track priorities
    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Start loop
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Pipeline built successfully. Starting playback...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    print("Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())
