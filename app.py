from flask import Flask, render_template, request
import numpy as np
import pickle
import warnings

app = Flask(__name__)

# Ignore specific UserWarning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but SVC was fitted with feature names")

# Load the saved model from the pickle file
with open("model.pkl", "rb") as file:
    classifier = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def home():
    loan_status = None

    if request.method == 'POST':
        Gender = int(request.form['Gender'])
        Married = int(request.form['Married'])
        Dependents = int(request.form['Dependents'])
        Education = int(request.form['Education'])
        Self_Employed = int(request.form['Self_Employed'])
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_area = (request.form['Property_area'])

        # Convert input data to a numpy array and reshape it
        input_data = np.array([Gender, Married, Dependents, Education, Self_Employed,
                               ApplicantIncome, CoapplicantIncome, LoanAmount,
                               Loan_Amount_Term, Credit_History, Property_area]).reshape(1, -1)

        # Perform prediction
        prediction = classifier.predict(input_data)

        # Output the prediction
        loan_status = 'Approved' if prediction[0] == 1 else 'Not Approved'

    return render_template("index.html", Loan_Status=loan_status)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
