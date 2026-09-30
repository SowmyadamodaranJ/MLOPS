from locust import HttpUser, task, between

class PredictiveMaintenanceUser(HttpUser):
    """
    Locust load testing user.
    Simulates real-world traffic hitting the Smart Factory PDM API.
    """
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1.0, 3.0)
    
    @task(3)
    def predict_endpoint(self):
        """Simulate a machine sending telemetry for prediction."""
        payload = {
            "machineID": "1",
            "datetime": "2024-01-01 10:00:00",
            "volt": 170.5,
            "rotate": 450.2,
            "pressure": 100.1,
            "vibration": 40.5
        }
        with self.client.post("/api/predict", json=payload, catch_response=True) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
                
    @task(1)
    def check_health(self):
        """Simulate a load balancer health check."""
        self.client.get("/api/health")
        
    @task(1)
    def view_dashboard(self):
        """Simulate a user opening the dashboard."""
        self.client.get("/api/dashboard")
