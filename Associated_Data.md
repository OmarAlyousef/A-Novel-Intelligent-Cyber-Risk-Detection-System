#  Associated Data

# Link Dataset 
https://www.stratosphereips.org/datasets-iot23

# 🔍 Dataset Description
	•	File Used: CTU-IoT-Malware-Capture-*.csv
	•	Source: CTU University, Czech Republic – IoT Malware Capture Dataset (IoT-23)
	•	Rows: Typically over 1 million records per capture file
	•	Columns: Over 80 features extracted from network flows (numeric & categorical)

# 🎯 Purpose

Detect and classify malicious network flows in IoT traffic, including Botnet, DDoS, and other attack types.

# 💾 Notes
	•	Multi-class labels representing various IoT attack categories.
	•	Missing or infinite values handled during preprocessing.
	•	Data extracted from .pcap files using feature engineering scripts.


#  Associated Data

# Link Dataset
https://www.unb.ca/cic/datasets/ids-2017.html

# 🔍 Dataset Description
- **File Used:** MachineLearningCSV.zip (CSV files extracted from network traffic flows)
- **Source:** Canadian Institute for Cybersecurity – University of New Brunswick (UNB)
- **Rows:** Over 2.2 million records across all combined files
- **Columns:** 79 columns (78 numeric features + 1 label column)

# 🎯 Purpose
Provide realistic and labeled data for testing and evaluating Intrusion Detection Systems (IDS), including both normal traffic and multiple modern attack scenarios such as Brute Force, DoS/DDoS, Heartbleed, Web Attacks, Infiltration, and Botnet.

# 💾 Notes
- Multi-class labels representing various attack categories alongside normal traffic
- Imbalanced dataset, with attack traffic comprising approximately 20% of the total
- Features extracted from `.pcap` files using the CICFlowMeter tool and converted to CSV
- Some files contain missing or infinite values that need to be handled during preprocessing
