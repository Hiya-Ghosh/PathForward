from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def career():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        # Get form data as dictionary
        form_data = request.form.to_dict(flat=True)
        print("Form Data:", form_data)

        # Convert form data values to numeric type (float)
        arr = [float(value) for value in form_data.values()]
        data = np.array(arr).reshape(1, -1)
        print("Input Data:", data)

        # Load the machine learning model
        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))

        # Make predictions
        predictions = loaded_model.predict(data)
        print("Predictions:", predictions)

        # Get predicted probabilities
        pred_probabilities = loaded_model.predict_proba(data)
        print("Predicted Probabilities:", pred_probabilities)

        # Thresholding probabilities
        pred_thresholded = pred_probabilities > 0.05

        # Extract selected categories based on thresholding
        selected_categories = np.where(pred_thresholded)[1]

        # Filter out the predicted category
        final_res = {index: value for index, value in enumerate(selected_categories) if value != predictions[0]}
        print("Final Result:", final_res)

        # Dictionary mapping job indices to job titles
        jobs_dict = {0:'AI ML Specialist',
                   1:'API Integration Specialist',
                   2:'Application Support Engineer',
                   3:'Business Analyst',
                   4:'Customer Service Executive',
                   5:'Cyber Security Specialist',
                   6:'Data Scientist',
                   7:'Database Administrator',
                   8:'Graphics Designer',
                   9:'Hardware Engineer',
                   10:'Helpdesk Engineer',
                   11:'Information Security Specialist',
                   12:'Networking Engineer',
                   13:'Project Manager',
                   14:'Software Developer',
                   15:'Software Tester',
                   16:'Technical Writer'}

        
        predicted_job = jobs_dict.get(predictions[0])
        print("Predicted Job:", predicted_job)

        return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=predicted_job)

if __name__ == '__main__':
    app.run(debug=True)
