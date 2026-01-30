from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load("../model/predictive_ecommerce_model.pkl")
scaler = joblib.load("../model/predictive_ecommerce_scaler.pkl")


def init_customer(time_spent, pages_viewed, shipping_cost, device_type):
    return {
        "time_spent": time_spent,
        "pages_viewed": pages_viewed,
        "shipping_cost": shipping_cost,
        "device_type_Mobile": 1 if device_type == "Mobile" else 0,
        "device_type_Tablet": 1 if device_type == "Tablet" else 0,
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "E-commerce Purchase Prediction API",
            "version": "1.0",
            "endpoints": {
                "/predict": "POST - Make a purchase prediction",
                "/health": "GET - Check API health",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate required fields
        required_fields = ["time_spent", "pages_viewed", "shipping_cost", "device_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Validate device_type
        valid_devices = ["Desktop", "Mobile", "Tablet"]
        if data["device_type"] not in valid_devices:
            return jsonify(
                {"error": f"Invalid device_type. Must be one of: {valid_devices}"}
            ), 400

        # Prepare input data
        customer_data = init_customer(
            time_spent=data["time_spent"],
            pages_viewed=data["pages_viewed"],
            shipping_cost=data["shipping_cost"],
            device_type=data["device_type"],
        )

        # Convert to DataFrame
        input_df = pd.DataFrame([customer_data])

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Prepare response
        result = {
            "prediction": "Purchase Made" if prediction[0] == 1 else "No Purchase",
            "prediction_value": int(prediction[0]),
            "confidence": {
                "no_purchase": float(prediction_proba[0][0]),
                "purchase": float(prediction_proba[0][1]),
            },
            "input_data": data,
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=34521)
