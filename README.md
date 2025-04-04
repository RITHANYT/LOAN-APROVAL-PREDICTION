💰 Loan Approval Prediction System
This is a Machine Learning web application built with Flask and Support Vector Machine (SVM) classifier that predicts whether a user's loan will be approved or not based on their inputs such as income, credit history, dependents, etc.

📊 Dataset
Source: loan_dataset.csv

The dataset includes fields like:
--Gender
--Marital Status
--Number of Dependents
--Education
--Employment Status
--Applicant & Coapplicant Income
--Loan Amount & Term
--Credit History
--Property Area
--Loan Status (Target)

🛠️ Tech Stack
Layer	                Technology
Frontend    	      HTML (via index.html)
Backend	            Flask (Python)
Machine Learning	  scikit-learn (SVM Classifier)
Data Handling	      pandas, NumPy

##Installation Steps
--Clone the Repository: git clone https://github.com/RITHANYT/LOAN-APROVAL-PREDICTION.git
  cd loan-approval-prediction
  
--Install Dependencies: pip install -r requirements.txt

--Place the Dataset Make sure loan_dataset.csv is present in the project root.

##Run the App
python app.py
Visit in Browser Go to http://127.0.0.1:5000/ to access the web app.

##🧠 How It Works
The model is trained using SVM with a linear kernel.
Input fields are collected through an HTML form.
After form submission, the data is passed to the backend, where the ML model predicts approval.
The result is shown as a success or rejection message.
