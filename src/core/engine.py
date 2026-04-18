from io import BytesIO
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.types import Number
from torchvision import transforms

# import torch.nn as nn


class InferenceEngine:
    def __init__(self, path: torch.types.FileLike):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path)
        self.embeddings = {
            k: v.to(self.device) for k, v in checkpoint["embeddings"].items()
        }

        # self.model = self._load_model()
        self.transform = self._build_transform()

    # def _load_model(self):
    #     model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    #     model.classifier = nn.Identity()  # type: ignore
    #     model.eval()
    #     model.to(self.device)
    #     return model

    def _build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((8, 8)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def search(self, query_emb: torch.Tensor) -> tuple[Optional[str], Number]:
        best_key = None
        best_score = -1

        for key, emb in self.embeddings.items():
            score = torch.dot(query_emb, emb).item()

            if score > best_score:
                best_score = score
                best_key = key

        return best_key, best_score

    def encode(self, img_bytes: bytes) -> torch.Tensor:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        tensor = self.transform(img).to(self.device)  # [3, H, W]

        mean_color = tensor.mean(dim=(1, 2))  # [3]
        flat = tensor.flatten()  # [N]

        emb = torch.cat([flat * 0.5, mean_color * 2.0])
        emb = F.normalize(emb, dim=0)

        return emb

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.dot(a, b).item()
