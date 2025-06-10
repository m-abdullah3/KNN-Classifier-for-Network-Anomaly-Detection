# KNN-Classifier-for-Network-Anomaly-Detection
This project implements the K-Nearest Neighbors (KNN) classifier in Python. We'll build models by varying training sizes (40%, 60%, 80%), K-values (1, 5, 15), and feature sets (all features, feature removal, random feature addition). We'll evaluate performance using accuracy, precision, recall, and confusion matrices.

Project Overview
This project uses the K-Nearest Neighbors (KNN) algorithm to detect anomalies in network logs. It evaluates how different parameters—training data size, number of neighbors (K-values), and feature selection—impact the model's ability to classify normal vs. anomalous traffic. Performance is measured using accuracy, precision, recall, and confusion matrices.

Features
Anomaly Detection: Binary classification (ANOMALY label) to identify irregular network behavior.

Parameter Testing:

Training splits (40%, 60%, 80%) to optimize data usage.

K-values (1, 5, 15) to balance the bias-variance tradeoff.

Feature experiments (all features, removed key features, random noise) to assess robustness.

Visual Diagnostics: Confusion matrices and accuracy plots for interpretability.

Project Structure


├── network-logs.xls          # Dataset

├── preprocessing.py          # Data cleaning & label analysis (e.g., ANOMALY distribution)

├── var-trainSize.py          # Tests how training size affects anomaly detection  

├── diff-cols.py              # Evaluates feature importance for anomaly classification 

├── diff-neighbors.py         # Compares K-values for detection stability  

└── KNN-Model.py             # Core KNN implementation with anomaly metrics  

Installation & Setup
Install dependencies:

pip install pandas numpy scikit-learn matplotlib  
Run scripts (example):

python KNN-Model.py  # Outputs anomaly detection metrics  

Usage Examples
Anomaly Detection Metrics (KNN-Model.py)
python
accuracy = 0.9250  
precision = 0.8562  
recall = 0.9250  
True Positives (TP): 12  # Correctly flagged anomalies  
False Positives (FP): 8  # Normal traffic misclassified as anomalies  

Feature Impact (diff-cols.py)
Removing the LATENCY feature reduced accuracy by 5%, showing its importance for anomaly detection.

Tech Stack
Python 3

Libraries: Scikit-learn (KNN, metrics), Pandas (data handling), Matplotlib (visualization)

License
MIT License. See LICENSE for details.
