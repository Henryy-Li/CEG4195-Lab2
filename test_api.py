from lab1 import app
def test_predictions():
  client = app.test_client()
  response = client.post("/predict", json={"text": "That was great!"})
  assert response.status_code == 200
  assert "label" in response.get_json()