# California Housing Price Prediction API

A FastAPI-based machine learning web application that predicts California housing prices using a Random Forest model trained on scikit-learn's California housing dataset.

## Features

- RESTful API for house price predictions
- Random Forest model trained on California housing dataset
- Single prediction endpoints
- Model information endpoint
- Health check endpoints
- Load testing with Locust

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kingabzpro/Stress-Testing-FastAPI.git
cd Stress-Testing-FastAPI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:
```bash
python run_server.py  
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check
- **GET** `/` - Root endpoint
- **GET** `/health` - Health check endpoint

### 2. Model Information
- **GET** `/model-info` - Get information about the ML model and features

### 3. Predictions
- **POST** `/predict` - Single house price prediction
  
  Request body:
  ```json
  {
    "features": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]
  }
  ```

- **POST** `/batch-predict` - Batch predictions for multiple houses

## Features Description

The model requires 8 features in the following order:

1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

## Load Testing with Locust

### Running Locust with Web UI

```bash
locust -f tests/locustfile.py --host http://localhost:8000
```

Open `http://localhost:8089` in your browser to access the Locust web interface.

### Running Locust in Headless Mode

```bash
locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 60s
```

Parameters:
- `-u 50`: Total number of users to simulate
- `-r 5`: Spawn rate (users per second)
- `-t 60s`: Test duration

### Generate HTML Report

```bash
locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 100 -r 10 -t 120s --html report.html
```

## Model Performance

The Random Forest model is trained on the California housing dataset with:
- 100 estimators
- Maximum depth of 10
- Features are standardized using StandardScaler

## Project Structure

```
ml-fastapi-app/
├── app/
│   ├── __init__.py      # Package initialization
│   ├── main.py          # FastAPI application
│   ├── models.py        # Pydantic models
│   └── ml_model.py      # ML model implementation
├── tests/
│   └── locustfile.py    # Locust test scenarios
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.8+
- FastAPI
- scikit-learn
- NumPy
- Pandas
- Locust

