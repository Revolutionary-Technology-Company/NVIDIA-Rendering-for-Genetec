# Dockerfile.app
# Utilizing the official unified DeepStream runtime base image
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# Prevent interactive prompts from blocking automated provisioning layers
ENV DEBIAN_FRONTEND=noninteractive

# Install core Linux video, compilation, and system frameworks
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-gi \
    python3-gst-1.0 \
    python3-numpy \
    libgstrtspsrc-1.0 \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-dev \
    libgl1-mesa-glx \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade base pip dependencies and pre-install LLVM compiler for Numba
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install Python requirements matching our high-speed analytics pipeline modules
RUN pip3 install --no-cache-dir \
    numba>=0.59.0 \
    paho-mqtt>=2.0.0 \
    opencv-python-headless \
    paddleocr \
    paddlepaddle-gpu \
    layoutparser \
    streamlit \
    pandas \
    plotly

# Copy your local engineering codebase straight into the image frame
COPY pipeline.py yolov8_triton_parser.py ocr_layer.py blueprint_reader.py telemetry_producer.py dashboard.py ./
COPY config_infer_rtx6000.txt config_infer_rtx50_triton.txt config_tracker_nvdcft.txt ./

# Establish standard entry execution target paths
ENTRYPOINT ["python3", "pipeline.py"]
