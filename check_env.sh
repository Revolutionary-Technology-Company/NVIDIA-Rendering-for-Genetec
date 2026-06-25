#!/usr/bin/env bash

# Terminal text formatting configurations
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}🛡️  GENETEC + NVIDIA DEEPESTREAM RUNTIME PRE-FLIGHT VERIFIER     ${NC}"
echo -e "${BLUE}================================================================${NC}"

ERRORS_FOUND=0

# 1. VERIFY PHYSICAL HARDWARE DRIVER STATUS
echo -n "[CHECK] Inspecting Host NVIDIA Driver Availability... "
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo -e "${GREEN}SUCCESS (Detected Driver v${DRIVER_VER})${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo -e "        -> Reason: 'nvidia-smi' utility not found on host environment system path."
    ((ERRORS_FOUND++))
fi

# 2. VERIFY DOCKER COMPUTE RUNTIME HOOKS
echo -n "[CHECK] Inspecting Docker NVIDIA Runtime Extensions... "
if docker info | grep -iq "nvidia"; then
    echo -e "${GREEN}SUCCESS (NVIDIA Runtime Registered)${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo -e "        -> Reason: Docker container backend does not have access to host GPU layers."
    echo -e "        -> Remediate: Install the 'nvidia-container-toolkit' package."
    ((ERRORS_FOUND++))
fi

# 3. VERIFY ESSENTIAL PIPELINE MODULE CONFIGURATION FILES
echo -n "[CHECK] Verifying Pipeline Configuration Files... "
REQUIRED_CONFIGS=("config_infer_rtx6000.txt" "config_infer_rtx50_triton.txt" "config_tracker_nvdcft.txt" "docker-compose.yml")
CONFIG_MISSING=0

for cfg in "${REQUIRED_CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo -e "\n        ${RED}>> Missing Critical Config File: $cfg${NC}"
        CONFIG_MISSING=1
    fi
done

if [ $CONFIG_MISSING -eq 0 ]; then
    echo -e "${GREEN}SUCCESS (All Core Configurations Present)${NC}"
else
    echo -e "        ${RED}FAILED Configuration Audit${NC}"
    ((ERRORS_FOUND++))
fi

# 4. VERIFY HARDWARE-SPECIFIC COMPILED AI TENSORRT MODEL ENGINES
echo -n "[CHECK] Checking Architectural TensorRT Models... "
TRITON_ENGINE="triton_model_repo/yolov8x_blackwell/1/model.trt"
NATIVE_ENGINE="yolov8x_ada_6000.engine"

if [ -f "$TRITON_ENGINE" ] || [ -f "$NATIVE_ENGINE" ]; then
    echo -e "${GREEN}SUCCESS (Compiled Engine Binary Framework Detected)${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo -e "        -> Reason: Neither '$NATIVE_ENGINE' nor '$TRITON_ENGINE' was found."
    echo -e "        -> Action Required: Compile models via 'trtexec' as documented in docs/deployment_guide.md."
fi

# 5. ASSESS FINAL VALIDATION STATUS
echo -e "${BLUE}================================================================${NC}"
if [ $ERRORS_FOUND -eq 0 ]; then
    echo -e "${GREEN}🚀 PRE-FLIGHT CHECKS PASSED. SYSTEM READY FOR LAUNCH.${NC}"
    echo -e "   Execute: 'docker compose up -d' to start the tracking engine."
    echo -e "${BLUE}================================================================${NC}"
    exit 0
else
    echo -e "${RED}❌ PRE-FLIGHT VERIFICATION FAILED with $ERRORS_FOUND Core System Error(s).${NC}"
    echo -e "   Remediate the infrastructural faults highlighted above before booting the pipeline."
    echo -e "${BLUE}================================================================${NC}"
    exit 1
fi
