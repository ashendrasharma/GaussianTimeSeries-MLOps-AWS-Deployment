import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd

# Create a Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for the app
cors = CORS(app)

# Load a pre-trained machine learning model from a pickled file
model = pickle.load(open("./Output/model.pkl", "rb"))

# API route for a status check
@app.route('/check', methods=['GET'])
@cross_origin()
def return_status():
    """
    Endpoint for checking the status of the Flask app.
    """
    return "Yay! Flask App is running"

# API route to get time series predictions
@app.route('/', methods=['POST'])
@cross_origin()
def return_model_prediction():
    """
    Endpoint for making time series predictions using a machine learning model.
    Expects a POST request with a CSV file containing time series data.
    Returns the model's predictions as a JSON response.
    """
    try:
        # Read the CSV data from the POST request
        data = pd.read_csv(request.files.get("data"))

        # Convert the 'month' column to timestamps
        data["timestamp"] = data["month"].apply(lambda x: x.timestamp())

        # Make predictions using the loaded model
        predictions = model.predict(data["timestamp"].values.reshape(-1, 1))
        final_predictions = list(predictions)

        # Return predictions in a JSON response
        return {"status_code": 200, "message": "Success", "body": {"preds": final_predictions}}

    except Exception as e:
        # Handle exceptions and return an error message
        print(f"Error occurred: {e}")
        return {"status_code": 404, "message": f"Error: {e}"}

if __name__ == '__main__':
    # Start the Flask app, allowing access from all network interfaces on port 5000
    app.run("0.0.0.0", port=5000)
