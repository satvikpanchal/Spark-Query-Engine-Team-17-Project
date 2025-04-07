import base64
import io
import torch
from PIL import Image, ImageFile
from pyspark.sql.types import FloatType
from torchvision import transforms
from weight_loader import load_model

# ImageFile.LOAD_TRUNCATED_IMAGES = True


class InferenceUDF:
    def __init__(self, model_path="fraud_model_weights.pth"):
        self.model = load_model(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.return_type = FloatType()

    def __call__(self, base64_str, image_id=None):
        try:
            if not base64_str:
                print("Empty base64 string received")
                return -1.0
            if len(base64_str) < 5000:
                print(f"Suspiciously short base64 for image {image_id}")
                return -1.0
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            if image.size == (0, 0):
                print("Invalid image size")
                return -1.0

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(image_tensor)
                return float(output.item())

        except Exception as e:
            print(f"Inference failed for image {image_id}: {e}")
            return -1.0
