#!/usr/bin/env python3
import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
from queue import Queue
import plotly.express as px

# Dashboard Global Configurations
MQTT_BROKER_HOST = "localhost" # Match your central server IP
MQTT_BROKER_PORT = 1883
TOPICS = [
    ("security/analytics/threats", 2),
    ("security/analytics/lpr", 1),
    ("security/analytics/blueprints", 0)
]

# Thread-safe queue to pass background network metrics into Streamlit's state cache
if "metrics_queue" not in st.session_state:
    st.session_state.metrics_queue = Queue()
if "threat_log" not in st.session_state:
    st.session_state.threat_log = []
if "lpr_log" not in st.session_state:
    st.session_state.lpr_log = []

# --- BACKGROUND MQTT SUBSCRIBER THREAD ---
def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())
        payload["topic"] = message.topic
        st.session_state.metrics_queue.put(payload)
    except Exception as e:
        pass

@st.cache_resource
def start_mqtt_listener():
    """Initializes a permanent background consumer connection."""
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        for topic, qos in TOPICS:
            client.subscribe(topic, qos=qos)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"Failed to connect to central MQTT Broker: {str(e)}")
        return None

# Start background network thread
mqtt_client = start_mqtt_listener()

# --- STREAMLIT USER INTERFACE LAYOUT ---
st.set_page_config(page_title="Genetec + NVIDIA Threat Matrix", layout="wide", page_icon="🛡️")

st.title("🛡️ Engineering Room Threat Matrix & Telemetry Dashboard")
st.caption("Real-Time DeepStream Pipeline Analytics Pipeline Interface")

# Top Row KPI Blocks
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

# Process outstanding background network queue items
while not st.session_state.metrics_queue.empty():
    data = st.session_state.metrics_queue.get()
    topic = data.get("topic")
    
    if "threats" in topic:
        st.session_state.threat_log.insert(0, {
            "Time": time.strftime("%H:%M:%S", time.localtime(data.get("timestamp", time.time()))),
            "Camera": data.get("camera", "Unknown"),
            "Alert": data.get("alert", "WARNING"),
            "Target ID": data.get("person_id", "N/A")
        })
    elif "lpr" in topic:
        lpr_data = data.get("data", {})
        st.session_state.lpr_log.insert(0, {
            "Time": time.strftime("%H:%M:%S", time.localtime(data.get("timestamp", time.time()))),
            "Camera": data.get("camera_id", "Unknown"),
            "Plate": lpr_data.get("plate_number", "N/A"),
            "Confidence": f"{lpr_data.get('confidence_score', 0)*100:.1f}%"
        })

# Maintain max window sizing for log dataframes
st.session_state.threat_log = st.session_state.threat_log[:50]
st.session_state.lpr_log = st.session_state.lpr_log[:50]

# Render Dynamic KPI Statistics
with kpi_col1:
    threat_count = len([t for t in st.session_state.threat_log if t["Alert"] == "ARMED_INDIVIDUAL"])
    if threat_count > 0:
        st.metric(label="⚠️ ACTIVE WEAPON THREATS", value=threat_count, delta="CRITICAL STATUS", delta_color="inverse")
    else:
        st.metric(label="⚠️ ACTIVE WEAPON THREATS", value=0, delta="SECURE")

with kpi_col2:
    st.metric(label="🚗 VEHICLES IDENTIFIED (LPR)", value=len(st.session_state.lpr_log))

with kpi_col3:
    st.metric(label="📊 SYSTEM NETWORK CHANNELS", value="3 ACTIVE TOPICS", delta="RTX COMPATIBLE")

st.markdown("---")

# Main Content Layout Columns
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🚨 Critical Threat Escalation Logs")
    if st.session_state.threat_log:
        df_threats = pd.DataFrame(st.session_state.threat_log)
        
        # Color coding style injection for visual scannability 
        def highlight_threats(row):
            return ['background-color: #4a0e17; color: white' if row['Alert'] == 'ARMED_INDIVIDUAL' else '' for _ in row]
            
        st.dataframe(df_threats.style.apply(highlight_threats, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("No weapon threats or critical perimeter violations detected across nodes.")

with col_right:
    st.subheader("💳 License Plate OCR Captures")
    if st.session_state.lpr_log:
        df_lpr = pd.DataFrame(st.session_state.lpr_log)
        st.dataframe(df_lpr, use_container_width=True, hide_index=True)
    else:
        st.info("Awaiting incoming vehicle plate detections...")

# Refresh control mechanisms for live streaming rendering updates
time.sleep(0.5) # Protects CPU lanes from aggressive interface loop thrashing
st.rerun()
