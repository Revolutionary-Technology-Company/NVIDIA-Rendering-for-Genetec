#!/usr/bin/env python3
import json
import time
import random
import paho.mqtt.client as mqtt

# Network Topology Alignment
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883

TOPIC_THREATS = "security/analytics/threats"
TOPIC_LPR = "security/analytics/lpr"
TOPIC_BLUEPRINTS = "security/analytics/blueprints"

# Synthetic Asset Pools
CAMERAS = ["ENGINEERING_ROOM_CAM_01", "PERIMETER_GATE_04", "LOADING_DOCK_CAM_02"]
PLATES = ["7XYZ89", "1ABC23", "9LMN45", "4STU67", "2VWX89"]
STAFF_IDS = [102, 105, 108, 142, 199]
SYSTEM_CODES = ["SYS-HVC-01", "GRID-ZONE-B", "ELEV-A-REV3", "HV-TRANS-02"]

def generate_synthetic_payloads():
    """Generates a randomized simulation packet to test downstream UI states."""
    event_roll = random.random()
    camera = random.choice(CAMERAS)
    timestamp = time.time()

    # 1. SIMULATE WEAPON CRITICAL THREATS (30% Probability)
    if event_roll < 0.30:
        person_id = random.choice(STAFF_IDS)
        # Randomize close vs long distance environments to verify dashboard styling changes
        env_status = "LONG_DISTANCE_HIGH_SENSITIVITY" if random.random() > 0.5 else "CLOSE_RANGE"
        
        payload = {
            "timestamp": timestamp,
            "camera": camera,
            "alert": "ARMED_INDIVIDUAL",
            "person_id": int(person_id),
            "environment": env_status,
            "confidence": round(random.uniform(0.72, 0.98), 4)
        }
        return TOPIC_THREATS, payload

    # 2. SIMULATE LICENSE PLATE RECOGNITION (40% Probability)
    elif event_roll < 0.70:
        payload = {
            "timestamp": timestamp,
            "camera_id": camera,
            "event_type": "LICENSE_PLATE_DETECTION",
            "data": {
                "plate_number": random.choice(PLATES),
                "confidence_score": round(random.uniform(0.81, 0.99), 4)
            }
        }
        return TOPIC_LPR, payload

    # 3. SIMULATE METRIC BLUEPRINT MATRIX SCANS (30% Probability)
    else:
        # Build multi-line document schema arrays
        lines = []
        for _ in range(random.randint(1, 3)):
            lines.append({
                "text": f"{random.choice(SYSTEM_CODES)} | DIM: {random.randint(10,500)}mm",
                "score": round(random.uniform(0.65, 0.95), 4)
            })
            
        payload = {
            "timestamp": timestamp,
            "camera_id": camera,
            "event_type": "BLUEPRINT_METRIC_SCAN",
            "blueprint_instance_id": random.randint(1000, 9999),
            "scanned_payload": lines
        }
        return TOPIC_BLUEPRINTS, payload

def main():
    print("[SIMULATOR] Booting Synthetic Edge Telemetry Simulator...")
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        print(f"[SIMULATOR] Connected to MQTT Broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        print("[SIMULATOR] Pushing randomized live alert patterns. Press Ctrl+C to terminate.\n")
        
        while True:
            # Generate a new random packet
            topic, payload = generate_synthetic_payloads()
            
            # Use QoS matching your production architectural spec
            qos_target = 2 if topic == TOPIC_THREATS else (1 if topic == TOPIC_LPR else 0)
            
            client.publish(topic, json.dumps(payload), qos=qos_target)
            print(f" -> Broadcasted payload to [{topic}] | QoS: {qos_target}")
            
            # Throttle the data generation cadence (randomized interval between 0.5s and 2.5s)
            time.sleep(random.uniform(0.5, 2.5))
            
    except KeyboardInterrupt:
        print("\n[SIMULATOR] Gracefully stopping event simulation loops.")
    except Exception as e:
        print(f"[SIMULATOR ERROR] Broker communication failed: {str(e)}")

if __name__ == "__main__":
    main()
