import json
import logging
import random

from locust import HttpUser, task

# Reduce logging to improve performance
logging.getLogger("urllib3").setLevel(logging.WARNING)


class HousingAPIUser(HttpUser):
    def generate_random_features(self):
        """Generate random but realistic California housing features"""
        return [
            round(random.uniform(0.5, 15.0), 4),  # MedInc
            round(random.uniform(1.0, 52.0), 1),  # HouseAge
            round(random.uniform(2.0, 10.0), 2),  # AveRooms
            round(random.uniform(0.5, 2.0), 2),  # AveBedrms
            round(random.uniform(3.0, 35000.0), 0),  # Population
            round(random.uniform(1.0, 10.0), 2),  # AveOccup
            round(random.uniform(32.0, 42.0), 2),  # Latitude
            round(random.uniform(-124.0, -114.0), 2),  # Longitude
        ]

    @task(1)
    def model_info(self):
        """Test health endpoint"""
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")

    @task(3)
    def single_prediction(self):
        """Test single prediction endpoint"""
        features = self.generate_random_features()

        with self.client.post(
            "/predict", json={"features": features}, catch_response=True, timeout=10
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "prediction" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Failed to parse JSON")
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Status code: {response.status_code}")
