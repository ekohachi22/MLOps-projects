import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import lightning as L
import bentoml


class FaceEmotionClassifier(L.LightningModule):
    def __init__(
        self, lr: float = 0.001, weight_decay: float = 0, betas: tuple = (0.9, 0.999)
    ):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=None, num_classes=7)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)


def main(args):
    lightning_model = FaceEmotionClassifier.load_from_checkpoint(args.model_path)
    lightning_model.eval()

    model = lightning_model.model
    model.eval()

    bentoml.pytorch.save_model(
        "face_emotion_classifier", model, signatures={"predict": {"batchable": True}}
    )

    print("Saved to BentoML!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to convert models from .ckpt to bentoml format"
    )
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args)
