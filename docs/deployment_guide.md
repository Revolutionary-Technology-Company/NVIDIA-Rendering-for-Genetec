# 🛡️ Deployment Guide: Unified Genetec + NVIDIA DeepStream Pipeline

This guide outlines the end-to-end setup, model compilation, infrastructure configuration, and maintenance commands required to run the unified tracking ecosystem on both **NVIDIA RTX 6000 Pro (Ada Lovelace)** and **ASUS TUF RTX 50-Series (Blackwell)** architectures.

---

## 📋 1. Workstation Host Provisioning

Before deploying the containerized stack, the physical host machine must have the necessary NVIDIA compute layers installed natively on the host OS.

### Step A: Verify NVIDIA GPU Drivers
Ensure your drivers support CUDA 12.x+. 
```bash
nvidia-smi
```

### Step B: Install the NVIDIA Container Toolkit
The Container Toolkit exposes physical GPU hardware devices, video decoders (NVDEC), and Tensor Cores to the Docker layer.

```bash
# Setup the repository package feeds
curl -fsSL https://github.io | sudo gpg2 --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://github.io | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.p.d/nvidia-container-toolkit.list

# Install the toolkit packages
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart the local Docker service daemon to apply the runtime hook
sudo systemctl restart docker
```

---

## 🧠 2. TensorRT Model Engine Compilation Tutorial

TensorRT engine files (`.engine` / `.trt`) are compiled for a specific GPU architecture and **cannot** be moved between different card types. You must generate the engine file directly on the target machine.

### Option A: Compiling for the RTX 6000 Pro (Ada Lovelace - SM 8.9)
On the RTX 6000 Pro workstation, open a terminal in your project directory and execute `trtexec` inside a temporary container wrapper:

```bash
docker run --rm --gpus all -v \$(pwd):/workspace -w /workspace nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
  trtexec --onnx=yolov8x.onnx \
          --saveEngine=yolov8x_ada_6000.engine \
          --fp16 \
          --minShapes=images:1x3x2160x3840 \
          --optShapes=images:1x3x2160x3840 \
          --maxShapes=images:4x3x2160x3840
```

### Option B: Compiling for the ASUS TUF RTX 50 (Blackwell - SM 10.0)
On the RTX 50-Series workstation, compile the model to output a layout matched to your Triton local repository structure:

```bash
docker run --rm --gpus all -v \$(pwd):/workspace -w /workspace nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
  trtexec --onnx=yolov8x.onnx \
          --saveEngine=triton_model_repo/yolov8x_blackwell/1/model.trt \
          --fp16 \
          --minShapes=images:1x3x2160x3840 \
          --optShapes=images:1x3x2160x3840 \
          --maxShapes=images:4x3x2160x3840
```

---

## 🚀 3. Core Deployment Execution Commands

Deploying the stack is entirely standardized using Docker Compose profiles.

### Command 1: Clean Build the Application Stack
Run this whenever you modify underlying Python scripts (`pipeline.py`, `ocr_layer.py`, etc.) to bake the latest code changes into the local image:
```bash
docker compose build --no-cache
```

### Command 2: Boot Up Infrastructure (Foreground Mode)
Highly recommended during initial setup or debugging to view raw pipeline logs and handle camera streaming issues in real time:
```bash
docker compose up
```

### Command 3: Boot Up Infrastructure (Production Daemon Mode)
Launches the system in the background for permanent monitoring. It configures containers to restart automatically if an unexpected network drop happens:
```bash
docker compose up -d
```

### Command 4: Tear Down Stack and Clean Storage Volumes
Stops active streaming sessions and safely clears container network structures:
```bash
docker compose down -v
```

---

## 🛠️ 4. Active Operational Diagnostic Controls

Use these commands to monitor health and verify performance metrics inside your engineering monitoring center.

### Command 5: Stream Real-Time Pipeline Component Logs
Monitor live JIT processing outputs, LPR text captures, blueprint matrix extractions, and threat evaluations:
```bash
docker logs -f security_deepstream_core
```

### Command 6: Inspect Hardware Resource Overhead
Monitor real-time VRAM allocation bounds, GPU utilization spikes, and temperature conditions on your RTX hardware:
```bash
docker exec -it security_deepstream_core nvidia-smi
```

### Command 7: Audit Incoming Telemetry Message Streams
To verify that the edge pipeline is communicating correctly with your telemetry hub, hook into the MQTT broker stream directly to view outbound JSON payloads:
```bash
docker exec -it security_mqtt_broker mosquitto_sub -t "security/analytics/#" -v
```

---

## 📋 5. Rapid Workstation Architecture Swapping Checklist

When deploying to a new node, use this quick checklist to configure your target profile:

1. **Verify Stream Endpoint**: Ensure the RTSP URL parameter matching your target Genetec camera node is defined correctly inside `pipeline.py`.
2. **Toggle the Hardware Flag**: Open `docker-compose.yml`, locate the `camera-pipeline` execution line, and adjust the `--gpu-type` switch:
   * Use `["--gpu-type", "rtx6000"]` for enterprise Ada Lovelace rigs.
   * Use `["--gpu-type", "rtx50"]` for consumer Blackwell gaming rigs.
3. **Launch the Node Stack**: Run `docker compose up -d`.
4. **Access the Monitoring Interface**: Open a web browser on any machine within the local network and navigate to `http://<WORKSTATION_IP>:8501`.
