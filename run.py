import subprocess
import time

# Start FastAPI backend
print("Starting FastAPI backend...")
backend_process = subprocess.Popen(["python", "backend.py"])

# Give the backend some time to start
time.sleep(3)

# Start Streamlit frontend
print("Starting Streamlit frontend...")
frontend_process = subprocess.Popen(["streamlit", "run", "app.py"])

# Keep the script running
backend_process.wait()
frontend_process.wait()