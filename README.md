# A-Novel-Intelligent-Cyber-Risk-Detection-System
An intelligent cyber risk detection framework that combines Artificial Intelligence (AI), Natural Language Processing (NLP), Anomaly Detection, and Blockchain Logging to proactively detect and mitigate threats while ensuring immutable and verifiable audit trails.


# Features
	•	AI Layer: XGBoost / LSTM / CNN-LSTM for cyberattack classification.
	•	NLP Layer: Extracts entities and insights from threat intelligence reports.
	•	Anomaly Detection: Flags unusual network behaviors.
	•	Blockchain Logging: Stores security events securely and tamper-proof.
	•	Performance Evaluation: Confusion Matrix, ROC Curve, F1-score, and more.
 
# project/
├── data/               # Dataset files (CSV)
├── main.py             # Main execution script
├── requirements.txt    # Python dependencies
├── README.md           # Project description and usage
└── outputs/            # Reports, plots, blockchain logs

# Dataset
	•	CIC-IDS2017 or UNSW-NB15 datasets
	
 
 # Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py

# Expected Output
	•	Classification metrics: Accuracy, Recall, Precision, F1-score
	•	Visualizations: Confusion Matrix, ROC Curve, Feature Importance
	•	Blockchain transaction logs for anomalies
	•	NLP-generated threat reports
