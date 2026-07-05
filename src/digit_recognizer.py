from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torchvision import transforms

from src.digit_cnn import SudokuDigitCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/digit_cnn.pt")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


class DigitRecognizer:

    def __init__(self):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        self.model = SudokuDigitCNN(num_classes=10)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict_board(self, cells):

        board = []

        for cell in cells:

            if cell.is_empty:
                board.append(0)
                continue

            image = cv2.resize(cell.image, (28, 28))

            tensor = transform(image).unsqueeze(0).to(DEVICE)

            logits = self.model(tensor)

            pred = torch.argmax(logits, dim=1).item()

            board.append(pred)

        board = [board[i:i + 9] for i in range(0, 81, 9)]

        return board