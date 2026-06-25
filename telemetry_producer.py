#!/usr/bin/env python3
import json
import time
import paho.mqtt.client as mqtt

# Network Topology Configuration
MQTT_BROKER_HOST = "YOUR_CENTRAL_SERVER_IP"
MQTT_BROKER_PORT = 1883
MQTT_KEEPALIVE = 60

# Structured Telemetry Topics
TOPIC_LPR = "security/analytics/lpr"
TOPIC_BLUEPRINT = "security/analytics/blueprints"

class TelemetryProducer:
    def __init__(self):
        # Using Callback API v2 for robust asynchronous edge loops
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[TELEMETRY] Connected to Central MQTT Messaging Broker.")
        else:
            print(f"[TELEMETRY] Connection failed with error code: {rc}")

    def _on_disconnect(self, client, userdata, disconnect_flags, rc, properties=None):
        print("[TELEMETRY] Connection lost. Attempting automatic edge reconnection...")

    def boot_client(self):
        """Starts a background thread to handle messaging traffic non-blockingly."""
        try:
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEPALIVE)
            self.client.loop_start() # Spawns a background thread
        except Exception as e:
            print(f"[TELEMETRY ERROR] Could not initialize broker connection: {str(e)}")

    def send_lpr_payload(self, camera_id, plate_text, confidence):
        """Serializes and broadcasts captured license plate text."""
        payload = {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "event_type": "LICENSE_PLATE_DETECTION",
            "data": {
                "plate_number": plate_text,
                "confidence_score": round(confidence, 4)
            }
        }
        # QoS 1 guarantees delivery to the central tracking database
        self.client.publish(TOPIC_LPR, json.dumps(payload), qos=1)

    def send_blueprint_payload(self, camera_id, blueprint_id, extracted_data):
        """Serializes and broadcasts engineering room schematic data."""
        payload = {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "event_type": "BLUEPRINT_METRIC_SCAN",
            "blueprint_instance_id": blueprint_id,
            "scanned_payload": extracted_data # List of {"text": ..., "score": ...}
        }
        # QoS 0 minimizes overhead for high-frequency canvas data updates
        self.client.publish(TOPIC_BLUEPRINT, json.dumps(payload), qos=0)

    def shutdown(self):
        """Gracefully disconnects loops on pipeline stop."""
        self.client.loop_stop()
        self.client.disconnect()
