import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

# Load the loan dataset
loan_dataset = pd.read_csv('loan_dataset.csv')
# Replace '3+' with 3 in the 'Dependents' column
loan_dataset['Dependents'].replace('3+', 3, inplace=True)
# Preprocess the dataset
loan_dataset = loan_dataset.dropna()
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# Split the dataset into features and target variable
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

# Create a Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Get the feature values from the web form
    gender = int(request.form['Gender'])
    married = int(request.form['Married'])
    dependents = int(request.form['Dependents'])
    education = int(request.form['Education'])
    self_employed = int(request.form['Self_Employed'])
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    loan_amount_term = float(request.form['Loan_Amount_Term'])
    credit_history = int(request.form['Credit_History'])
    property_area = int(request.form['Property_Area'])

    # Create a numpy array with the feature values
    feature_values = np.array([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])

    # Make the loan approval prediction using the trained SVM classifier
    prediction = classifier.predict(feature_values)

    # Return a message indicating whether the loan is approved or not
    if prediction[0] == 1:
        return '<div class="message approved">Congratulations! Your loan has been approved.</div>'
    else:
        return '<div class="message rejected">Sorry, your loan has been rejected.</div>'
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

    # Predict the loan approval status
    output = classifier.predict(features_value)


    # Display the loan approval status
    if output == 1:
        return render_template('index.html', prediction_text='Loan Approved')
    else:
        return render_template('index.html', prediction_text='Loan Not Approved')
    
if __name__ == "__main__":
    app.run(debug=True)