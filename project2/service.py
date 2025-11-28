import torch
import torchvision.transforms as T
from PIL import Image as PILImage
import bentoml
import io
import base64


@bentoml.service(resources={"gpu": 1}, traffic={"timeout": 30})
class FaceEmotionService:
    def __init__(self):
        self.model = bentoml.models.get("face_emotion_classifier:latest").load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose(
            [
                T.Resize((96, 96)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.CLASSES = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]

    @bentoml.api
    def classify(self, img: str) -> dict:
        image_bytes = base64.b64decode(img)
        pil_img = PILImage.open(io.BytesIO(image_bytes))

        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

        return {
            "class_index": pred,
            "class_name": self.CLASSES[pred],
            "confidence": float(probs[pred]),
        }
