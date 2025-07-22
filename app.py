from flask import Flask, render_template, request as flask_request, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Load the trained model
model = joblib.load('model.joblib')

def preprocess_form(form):
    # Only use the 7 features the model expects
    return [
        int(form.get('gender')),
        int(form.get('age')),
        int(form.get('bp')),
        int(form.get('chol')),
        int(form.get('diabetes')),
        int(form.get('smoking')),
        int(form.get('chestpain'))
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    history = session.get('history', [])
    return render_template('history.html', history=history)

@app.route('/project', methods=['GET', 'POST'])
def project():
    prediction = None
    if flask_request.method == 'POST':
        form = flask_request.form
        data = {
            'Gender': form.get('gender'),
            'Age': form.get('age'),
            'Blood Pressure (mmHg)': form.get('bp'),
            'Cholesterol (mg/dL)': form.get('chol'),
            'Has Diabetes': form.get('diabetes'),
            'Smoking Status': form.get('smoking'),
            'Chest Pain Type': form.get('chestpain'),
            'Treatment': form.get('treatment'),
            'Family History': form.get('family'),
        }
        try:
            features = np.array([preprocess_form(form)])
            pred_num = model.predict(features)[0]
            if str(pred_num) == '0':
                prediction = 'Low Risk of Heart Attack'
            elif str(pred_num) == '1':
                prediction = 'High Risk of Heart Attack'
            else:
                prediction = f'Unknown Risk (model output: {pred_num})'
        except Exception as e:
            prediction = f"Error: {e}"
        # Store in session history
        if 'history' not in session:
            session['history'] = []
        session['history'].append({'inputs': data, 'result': prediction})
        session.modified = True
        return render_template('project.html', form_data=data, prediction=prediction)
    return render_template('project.html', form_data=None, prediction=None)

@app.route('/cleveland.html')
def cleveland():
    return render_template('cleveland.html')

if __name__ == '__main__':
    app.run(debug=True)
