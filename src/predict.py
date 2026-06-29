from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.digit_model import SudokuDigitCNN


class DigitPredictor:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = SudokuDigitCNN(num_classes=10).to(self.device)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. Train with `python -m src.train` first."
            )
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
        self.model.eval()

    @torch.no_grad()
    def predict_digit(self, digit_image: np.ndarray) -> int:
        image = digit_image.astype("float32") / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prediction = int(torch.argmax(logits, dim=1).item())
        return prediction
