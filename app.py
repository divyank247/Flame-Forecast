from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template("forest_fire.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input features from the form
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        print(f"Input features: {int_features}")
        print(f"Final array: {final}")

        # Predicting the probability of fire
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)
        
        # Determining the message based on the prediction
        if float(output) > 0.5:
            return render_template('forest_fire.html', pred=f'Your Forest is in Danger.\nProbability of fire occurring is {output}', bhai="kuch karna hain iska ab?")
        else:
            return render_template('forest_fire.html', pred=f'Your Forest is safe.\nProbability of fire occurring is {output}', bhai="Your Forest is Safe for now")
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('forest_fire.html', pred='Error occurred during prediction', bhai="Please check your input values.")

if __name__ == '__main__':
    app.run(debug=True)
