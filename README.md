# A-Novel-Intelligent-Cyber-Risk-Detection-System
An intelligent cyber risk detection framework that combines Artificial Intelligence (AI), Natural Language Processing (NLP), Anomaly Detection, and Blockchain Logging to proactively detect and mitigate threats while ensuring immutable and verifiable audit trails.


# 🎯Project Goals
	•	Combine AI, NLP, Anomaly Detection, and Blockchain Logging for proactive threat mitigation.
	•	Classify attacks using XGBoost, LSTM, or CNN-LSTM models.
	•	Generate NLP-based threat reports from textual data.
	•	Detect anomalies and log them immutably on the blockchain.
	•	Evaluate performance using metrics like F1-score, ROC Curve, and Confusion Matrix.

# 📂 Project Structure
	•	main.py : Main execution script
	•	data/ : Dataset files (CSV)
	•	requirements.txt : Python dependencies list
	•	README.md : Project description and usage
	•	outputs/ : Reports, plots, blockchain logs

# 📊 Dataset
	•	CIC-IDS2017 or UNSW-NB15 datasets

# ⚙️ How to Run

 Create a virtual environment
python -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows

Install dependencies
pip install -r requirements.txt

Run the model
python main.py

# ✅ Expected Output
	•	Accuracy, Precision, Recall, and F1-score
	•	Confusion Matrix & ROC Curve visualizations
	•	Blockchain transaction logs for detected anomalies
	•	NLP-generated threat intelligence report
 
