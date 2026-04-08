# ============================================================
#                       Import Statements
# ============================================================
import base64
import io

from PIL import Image
from lab2 import app

# ============================================================
#                        Main Code
# ============================================================
def test_predictions():
  client = app.test_client()

  artificalTestImage = Image.new("RGB", (256,256), color=(50,100,150))
  buffer = io.BytesIO()
  artificalTestImage.save(buffer, format="PNG")
  image_string_input = base64.b64encode(buffer.getvalue()).decode("utf-8")

  response = client.post("/predict", json={"image": image_string_input})
  assert response.status_code == 200
  assert "house_pixels" in response.get_json()
  assert "total_pixels" in response.get_json()
  assert "house_coverage" in response.get_json()
