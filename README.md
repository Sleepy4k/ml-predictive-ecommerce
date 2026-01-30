# Predictive E-Commerce

A machine learning project that predicts customer purchase behavior in e-commerce based on browsing patterns and session data. This project includes a complete ML pipeline from data exploration to model deployment via a Flask REST API.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the API Service](#running-the-api-service)
  - [Making Predictions](#making-predictions)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project analyzes customer behavior on an e-commerce platform to predict whether a customer will make a purchase. The prediction model considers factors such as:

- Time spent on the website
- Number of pages viewed
- Shipping cost
- Device type (Desktop, Mobile, Tablet)

The trained Random Forest model achieves high accuracy and is deployed through a Flask API for real-time predictions.

## ✨ Features

- **Data Analysis & Preprocessing**: Complete data cleaning and feature engineering pipeline
- **Multiple Model Comparison**: Evaluation of Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- **Feature Scaling**: StandardScaler implementation to prevent overfitting
- **Model Persistence**: Trained model and scaler saved for production use
- **REST API**: Flask-based API service for real-time predictions
- **Confidence Scores**: Probability estimates for each prediction

## 📁 Project Structure

```
predictive-ecommerce/
├── api/
│   └── app.py              # Flask API service
├── data/
│   └── ecommerce.csv       # Training dataset
├── model/
│   ├── predictive_ecommerce_model.pkl    # Saved Random Forest model
│   └── predictive_ecommerce_scaler.pkl   # Saved StandardScaler
├── notebook/
│   └── main.ipynb          # Jupyter notebook with full ML pipeline
├── pyproject.toml          # Project dependencies
└── README.md               # Project documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.13 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd predictive-ecommerce
   ```

2. **Install dependencies**:
   
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv:
   ```bash
   uv sync
   ```

3. **Verify installation**:
   ```bash
   python -c "import flask, pandas, sklearn; print('All dependencies installed!')"
   ```

## 🚀 Usage

### Training the Model

The complete model training pipeline is available in the Jupyter notebook:

1. **Open the notebook**:
   ```bash
   jupyter notebook notebook/main.ipynb
   ```

2. **Run all cells** to:
   - Load and explore the dataset
   - Preprocess and clean the data
   - Train multiple models (Logistic Regression, Decision Tree, Gradient Boosting, Random Forest)
   - Compare model performance
   - Apply feature scaling
   - Save the best model and scaler to the `model/` directory

The notebook includes:
- **Data Loading**: Reading the e-commerce dataset
- **Data Cleaning**: Removing unnecessary columns and handling missing values
- **Feature Engineering**: One-hot encoding for categorical variables
- **Model Training**: Training and evaluating 4 different algorithms
- **Model Optimization**: Feature scaling to improve performance
- **Model Serialization**: Saving the final model and scaler

### Running the API Service

Start the Flask API server:

```bash
cd api
python app.py
```

Or using uv:
```bash
cd api
uv run app.py
```

The API will start on `http://localhost:5000`

### Making Predictions

**Using curl**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_spent": 25,
    "pages_viewed": 8,
    "shipping_cost": 5.99,
    "device_type": "Mobile"
  }'
```

**Using Python**:
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "time_spent": 25,
    "pages_viewed": 8,
    "shipping_cost": 5.99,
    "device_type": "Mobile"
}

response = requests.post(url, json=data)
print(response.json())
```

## 📚 API Documentation

### Endpoints

#### `GET /`
Returns API information and available endpoints.

**Response**:
```json
{
  "message": "E-commerce Purchase Prediction API",
  "version": "1.0",
  "endpoints": {
    "/predict": "POST - Make a purchase prediction",
    "/health": "GET - Check API health"
  }
}
```

#### `GET /health`
Health check endpoint to verify the API and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

#### `POST /predict`
Make a purchase prediction for a customer.

**Request Body**:
```json
{
  "time_spent": 25,
  "pages_viewed": 8,
  "shipping_cost": 5.99,
  "device_type": "Mobile"
}
```

**Parameters**:
- `time_spent` (integer): Time spent on website in minutes
- `pages_viewed` (integer): Number of pages viewed during session
- `shipping_cost` (float): Shipping cost in dollars
- `device_type` (string): Device used - one of: "Desktop", "Mobile", "Tablet"

**Success Response** (200):
```json
{
  "prediction": "Purchase Made",
  "prediction_value": 1,
  "confidence": {
    "no_purchase": 0.23,
    "purchase": 0.77
  },
  "input_data": {
    "time_spent": 25,
    "pages_viewed": 8,
    "shipping_cost": 5.99,
    "device_type": "Mobile"
  }
}
```

**Error Response** (400):
```json
{
  "error": "Missing required field: time_spent"
}
```

## 📊 Model Performance

The project evaluates multiple machine learning algorithms:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear model |
| **Decision Tree** | Non-linear tree-based classifier |
| **Gradient Boosting** | Ensemble boosting method |
| **Random Forest** ⭐ | Final selected model with feature scaling |

The **Random Forest Classifier** with StandardScaler was selected as the final model for its superior performance and generalization capabilities.

### Model Features
- **Feature Engineering**: One-hot encoding for device types
- **Scaling**: StandardScaler for numerical features
- **Cross-validation**: Train-test split (80-20)
- **Metrics**: Accuracy, Precision, Recall, F1-Score

## 🛠 Technologies Used

- **Python 3.13+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Flask**: REST API framework
- **Joblib**: Model serialization
- **Jupyter Notebook**: Interactive development and analysis

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: Make sure to train the model using the notebook before running the API service, as the API requires the saved model files in the `model/` directory.
