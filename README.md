
# 🚛 Haul Truck Sensor Monitoring and Alerting System
---
## 📌 Overview

This project simulates an industrial monitoring system for haul trucks using time-series sensor data. It detects abnormal operating conditions, generates alerts, and evaluates detection performance to support proactive maintenance decisions.

This project demonstrates how industrial sensor data can be transformed into actionable maintenance insights using a structured and interpretable anomaly detection system.

## 🎯 Problem

Mining operations rely heavily on haul trucks, where unexpected equipment failures can lead to significant downtime and operational costs. Detecting early warning signs from sensor data is critical for improving reliability and reducing maintenance risk.

## 💡 Solution

This project implements an end-to-end monitoring pipeline that:

- Simulates realistic haul truck sensor data
- Injects failure patterns (overheating, hydraulic issues, vibration)
- Detects anomalies using rule-based logic and rolling statistics
- Assigns a health score to represent equipment condition
- Generates alerts (normal / warning / critical)
- Evaluates detection performance using precision and recall

## 📊 Data Simulation

The dataset is fully simulated to reflect real-world industrial conditions:

- 3 haul trucks
- 14 days of hourly data
- Sensors:
  - Engine temperature
  - Hydraulic pressure
  - Vibration
  - Speed
    
### Injected Failure Scenarios

- **Overheating:** gradual temperature increase with reduced speed
- **Hydraulic issue:** pressure drop with increased vibration
- **Mechanical vibration:** sudden spike in vibration
  
## ⚙️ Methodology

### 1. Rule-Based Detection

Threshold-based rules identify abnormal conditions:

- High engine temperature
- Low hydraulic pressure
- High vibration
- Low speed
  
### 2. Rolling Anomaly Detection

A rolling window approach detects deviations from recent behavior:

- Compares current values to rolling mean and standard deviation
- Helps identify gradual changes not captured by fixed thresholds

### 3. Health Score

A composite health score (0–100) is calculated:

- Penalizes abnormal sensor conditions
- Provides an interpretable measure of equipment health
- Drives alert classification

### 4. Alert System

Alerts are categorized based on health score:

- **Normal:** healthy operation
- **Warning:** early signs of abnormal behavior
- **Critical:** multiple or severe issues detected

## 📈 Results

Detection performance was evaluated by comparing simulated ground truth (`status`) with detected alerts:

```text
Precision: 0.86
Recall:    0.57
```

### Interpretation

- **High precision (0.86):** alerts are reliable with few false positives
- **Moderate recall (0.57):** some early-stage anomalies are missed

This reflects a conservative alerting strategy, prioritizing trust in alerts over sensitivity.

## 🔍 Key Insights

- Rule-based systems can effectively detect equipment anomalies in industrial settings
- There is a natural tradeoff between false alarms and missed detections
- Health scoring improves interpretability for operators and maintenance teams
- Rolling statistics help capture gradual failure patterns
  
## 🧱 Project Structure

```bash
├── simulate_sensor_data.py   # Generate synthetic haul truck sensor data
├── detect_anomalies.py      # Detect anomalies and generate alerts
├── evaluate_alerts.py       # Evaluate detection performance
├── plot_sensor_data.py      # Visualize sensor trends
├── haul_truck_sensor_data.csv
├── haul_truck_alerts.csv
└── README.md
```
## 🚀 How to Run

### 1. Generate data
```bash
python simulate_sensor_data.py
```
### 2. Detect anomalies
```bash
python detect_anomalies.py
```
### 3. Evaluate performance
```bash
python evaluate_alerts.py
```
### 4. (Optional) Visualize data
```bash
python plot_sensor_data.py
```
## 🔮 Future Improvements

- Machine learning-based anomaly detection
- Real-time streaming pipeline (Kafka / Spark)
- Adaptive thresholds per truck or operating condition
- Integration with maintenance scheduling systems
- Dashboard for live monitoring and alert visualization

## 🧠 Skills Demonstrated
- Time-series analysis
- Anomaly detection
- Data simulation
- Feature engineering
- System design for monitoring workflows
- Performance evaluation (precision / recall)
- Python (pandas, numpy, matplotlib)
## 📌 Summary

This project demonstrates how sensor data can be transformed into actionable insights through a structured monitoring pipeline. It highlights the challenges of anomaly detection in industrial environments and the importance of balancing detection sensitivity with operational reliability.
