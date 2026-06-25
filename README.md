# 🛡️ Genetec + NVIDIA Unified Threat Matrix & Telemetry Pipeline

An enterprise-grade, edge-optimized computer vision pipeline designed to pull high-fidelity 4K security streams from Genetec Media Gateway. The system performs real-time multi-object tracking, weapon proximity risk scaling, License Plate Recognition (LPR), and architectural blueprint processing. 

Optimized to run seamlessly across both enterprise **NVIDIA RTX 6000 Pro (Ada Lovelace)** and consumer **ASUS TUF RTX 50-Series (Blackwell)** architectures via a unified runtime toggle.

---

## 🏗️ Architecture Ecosystem Overview

The platform uses a decoupled, non-blocking pipeline approach to keep frame rates steady at 30+ FPS even under peak analytical loads:

1. **Inference Backend Layer (`pipeline.py`)**: Written via Python DeepStream (`pyds`) bindings. Dynamically swaps between native `nvinfer` (RTX 6000) and `nvinferserver` (RTX 50 Triton Backend) to bypass hardware SM architectural limitations.
2. **Matrix QuickMath Core (`yolov8_triton_parser.py`)**: Uses multi-core CPU threading and **Numba JIT (`@njit`)** execution to parse thousands of raw candidate boxes and evaluate threat-to-person coordinate proximity in microseconds.
3. **Structured Text Extractors (`ocr_layer.py` / `blueprint_reader.py`)**: Zero-copy GPU memory mappings crop target boundaries directly in VRAM. Text processing is managed using **PaddleOCR Layout Analysis** to parse data without moving frames back to host RAM.
4. **Telemetry & Visuals (`telemetry_producer.py` / `dashboard.py`)**: An asynchronous background MQTT client forwards security JSON data points to a multi-threaded **Streamlit** dashboard.

---

## 📂 Repository Structure

```text
├── Dockerfile.app             # Consolidated container construction layer
├── docker-compose.yml         # Multi-container multi-architecture orchestrator
├── pipeline.py                # Primary GStreamer entry loop application
├── yolov8_triton_parser.py    # JIT-compiled NMS and weapon distance calculations
├── ocr_layer.py               # In-GPU memory License Plate OCR processor
├── blueprint_reader.py        # Technical Wall Blueprint layout analyst
├── telemetry_producer.py      # Async multi-threaded MQTT messaging client
├── dashboard.py               # Streamlit-powered local telemetry interface
├── config_infer_rtx6000.txt   # Native TensorRT configuration parameters
├── config_infer_rtx50_triton.txt # Triton server Blackwell integration rules
├── config_tracker_nvdcft.txt  # NVTracker object association parameters
└── triton_model_repo/         # Triton specific local model hierarchy 
    └── yolov8x_blackwell/
        ├── config.pbtxt       # Blackwell optimization mapping structure
        └── 1/
            └── model.trt      # Hardware-compiled Blackwell engine binary
```

---

## ⚙️ Host Prerequisites

Before spinning up the containers, ensure your host workstation has the correct hardware communication bridge installed:

1. **NVIDIA Display Drivers**: Version `550.x` or newer.
2. **Docker Engine & Compose**: Native Docker runtime setup.
3. **NVIDIA Container Toolkit**: Mandatory layer to expose your GPU to the container spaces.

```bash
# Verify driver integration and compute availability
nvidia-smi

# Verify Docker has access to physical GPU layers
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

---

## 🚀 Rapid Production Deployment

The entire system is containerized for easy distribution across different monitoring rooms and camera rigs.

### 1. Model Engine Compilation
TensorRT engine files (`.engine` / `.trt`) are hardware-dependent. You must compile your raw ONNX models separately on each distinct machine type.

```bash
# Run this inside your target workstation terminal to compile your YOLO engine
trtexec --onnx=yolov8x.onnx --saveEngine=yolov8x_optimized.engine --fp16
```
* Rename and place your compiled engine file into the root directory (for the RTX 6000) or into `triton_model_repo/yolov8x_blackwell/1/model.trt` (for the ASUS TUF RTX 50).

### 2. Configure Environment Parameters
Open `pipeline.py` and assign your network parameters:
```python
# Line 77: Insert your active Genetec Media Gateway stream URL
source.set_property("location", "rtsp://GENETEC_SERVER_IP:554/LiveOS/Cameras/YOUR_CAMERA_ID")
```

### 3. Launching the Container Stack
Open `docker-compose.yml` and verify the hardware command string matching your current deployment room environment:

```yaml
# Inside docker-compose.yml -> camera-pipeline service
command: ["--gpu-type", "rtx6000"] # Swap to "rtx50" for ASUS TUF Blackwell nodes
```

Now build and run the services in the background:
```bash
# Build the images locally
docker compose build

# Boot the telemetry broker, pipeline engine, and analytical dashboard
docker compose up -d
```

### 4. Viewing the Monitoring Interface
Once the container states show active status, access your real-time tracking metrics dashboard by opening any browser on your network and navigating to:
```text
http://<WORKSTATION_IP>:8501
```

---

## 📡 Analytical Telemetry Schema

The pipeline streams high-priority alerts across distinct MQTT topics. Downstream applications can subscribe to these data layers:

### Threat Alerts (`security/analytics/threats`)
```json
{
  "timestamp": 1718873452.124,
  "camera": "ENGINEERING_ROOM_CAM_01",
  "alert": "ARMED_INDIVIDUAL",
  "person_id": 42
}
```

### Vehicle LPR Data (`security/analytics/lpr`)
```json
{
  "timestamp": 1718873455.589,
  "camera_id": "ENGINEERING_ROOM_CAM_01",
  "event_type": "LICENSE_PLATE_DETECTION",
  "data": {
    "plate_number": "7XYZ89",
    "confidence_score": 0.9421
  }
}
```

---

## 🛠️ Operational Diagnostics & Development

### View Real-Time Logging Pipeline:
```bash
docker logs -f security_deepstream_core
```

### Clean System Reset:
```bash
docker compose down -v
```

---
**Developed by the Revolutionary Technology Company Engineering Team.**  
For assistance adjusting Numba thread allocation bounds or custom YOLO class fine-tuning, open an official issue ticket within this repository.
