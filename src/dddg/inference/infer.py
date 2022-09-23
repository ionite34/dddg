from collections.abc import Generator
from pathlib import Path
from typing import Union

import numpy as np
import tomli
import torch
import torchvision.transforms as transforms

from dddg.image import pad_image
from dddg.inference.model import MultiOutputModel

LABELS = tomli.loads((Path(__file__).parent / "labels.toml").read_text())
MODEL_PATH = Path(__file__).parent / "model.pt"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

Result = dict[str, dict[str, Union[str, float]]]


class DuckyModel:
    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = MultiOutputModel(
            n_hat_classes=len(LABELS['hat']),
            n_acc_classes=len(LABELS['accessory']),
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def infer(self, image: np.ndarray) -> Result:
        """
        Infer attributes from a single image.

        Args:
            image: Image to infer attributes from.

        Returns:
            Dict containing the predicted attributes.
        """
        return next(self.infer_batch([image]))

    def infer_batch(self, images: list[np.ndarray]) -> Generator[Result, None, None]:
        """
        Infer attributes from a batch of images.

        Args:
            images: Image to infer attributes from.

        Returns:
            Generator of dicts containing the predicted attributes.
        """
        # Image should be smaller than 60x60
        x = [pad_image(image, 60) for image in images]

        # Convert to tensor
        tensor = torch.stack([
            self.transforms(img) for img in x
        ]).to(self.device)

        with torch.no_grad():
            output: dict[str, torch.Tensor] = self.model(tensor)
            prob_hat = torch.nn.functional.softmax(output['hat'], dim=1)
            prob_acc = torch.nn.functional.softmax(output['acc'], dim=1)

        for t_hat, t_acc, p_hat, p_acc in zip(output['hat'], output['acc'], prob_hat, prob_acc):
            # Get predictions (argmax)
            _, predict_hat = t_hat.cpu().max(0)
            _, predict_acc = t_acc.cpu().max(0)
            # Get probabilities
            prob_hat: torch.Tensor = p_hat.cpu()[predict_hat.item()]
            prob_acc: torch.Tensor = p_acc.cpu()[predict_acc.item()]
            res_hat: str = LABELS['hat'][predict_hat.item()]
            res_acc: str = LABELS['accessory'][predict_acc.item()]
            yield {
                'prediction': {
                    'hat': res_hat,
                    'acc': res_acc,
                },
                'confidence': {
                    'hat': float(prob_hat),
                    'acc': float(prob_acc),
                },
            }
