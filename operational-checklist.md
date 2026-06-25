* * * * *

🏥 System Operational Hand-Off Checklist
----------------------------------------

System Name: Unified Threat Matrix & Video Analytics Platform\
Target Hardware: NVIDIA RTX 6000 Pro / ASUS TUF RTX 50-Series\
Core Stack: NVIDIA DeepStream, Triton Inference Server, MQTT Telemetry, Streamlit

* * * * *

🔒 1. Network & Port Provisioning
---------------------------------

The hospital IT infrastructure team must open and secure the following internal ports:

-   Port 554 (Inbound): Unrestricted RTSP connection from the Genetec Media Gateway to the edge computer vision node.
-   Port 1883 (Internal Bridge): MQTT messaging broker data channel between the DeepStream container and the local telemetry dashboard.
-   Port 8501 (Outbound/Internal): HTTP access to allow approved engineering and security room workstations to view the Streamlit web dashboard.

📁 2. Local Storage & Log Management
------------------------------------

To comply with standard hospital retention rules without overloading local edge server storage drives:

-   Local Logging Limit: Verify that the Docker daemon configuration (`/etc/docker/daemon.json`) is active with a maximum file cap of 3 files at 100MB each per container.
-   Retention Schedule: Local logs will auto-purge after 7 days via `copytruncate`.
-   Backup Ingestion: If the hospital requires permanent storage of security event histories, the IT department should configure their central backup tools (e.g., Splunk, Veeam, Syslog) to ingest the active MQTT telemetry stream directly from `security/analytics/#`.

🔑 3. Credential & Parameter Inventory
--------------------------------------

Before running the deployment script in a production zone, update the following placeholders inside the workspace root:

-   Genetec Stream Endpoint: Set the correct production RTSP camera path inside `pipeline.py`.
-   MQTT Broker Host: Update `MQTT_BROKER_HOST` inside `telemetry_producer.py` and `dashboard.py` to target the hospital's authorized internal server IP address.

🛠️ 4. Daily Operational Health Verification
--------------------------------------------

The hospital's system administration team can verify the integrity of the platform using these three commands:

-   Pre-Flight Verification: Run `./check_env.sh` to automatically scan and verify NVIDIA drivers, container runtimes, and model configurations before system boot.
-   Live Telemetry Audit: Run `docker logs -f security_deepstream_core` to verify that active frame captures, staff tracking, and blueprint coordinates are executing without errors.
-   Hardware Overhead Check: Run `nvidia-smi` inside the host terminal to ensure GPU VRAM utilization remains stable under 85%, leaving adequate headroom for local OCR image croppers.

* * * * *
