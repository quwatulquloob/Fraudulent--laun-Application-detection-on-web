import numpy as np
import pickle
import warnings

# Ignore specific UserWarning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but SVC was fitted with feature names")

# Load the saved model from the pickle file
with open("model.pkl", "rb") as file:
    classifier = pickle.load(file)

# Input data
Gender=1
Married =1
Dependents=4
Education=1
Self_Employed=0
ApplicantIncome=4106
CoapplicantIncome=0.0
LoanAmount=40.0
Loan_Amount_Term=180.0
Credit_History=1.0
Property_area=1
input_data = (Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_area)

# Convert input data to a numpy array and reshape it
input_data_reshaped = np.asarray(input_data).reshape(1, -1)

# Prediction
prediction = classifier.predict(input_data_reshaped)

# Output the prediction
if prediction[0] == 0:
    print('The Loan is not Approved')
else:
    print('The Loan is Approved')
