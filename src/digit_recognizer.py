from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torchvision import transforms

from src.digit_model import SudokuDigitCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/best_sudoku_model_11classes.pth")

IMG_SIZE = 39  # اندازه ورودی مدل جدید

# کلاس‌های 0 (Empty_Clean) و 10 (Empty_Hints) هر دو یعنی «خانه خالی» روی برد
EMPTY_CLASSES = {0, 10}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # مقادیر را به بازه 0..1 نرمال می‌کند (هم‌راستا با آموزش)
])


class DigitRecognizer:

    def __init__(self):
        # مدل جدید مستقیماً با state_dict ذخیره شده (بدون کلید "model_state_dict")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        self.model = SudokuDigitCNN(num_classes=11)
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict_board(self, cells):

        board = []

        for cell in cells:

            if cell.is_empty:
                board.append(0)
                continue

            image = cv2.resize(cell.image, (IMG_SIZE, IMG_SIZE))

            tensor = transform(image).unsqueeze(0).to(DEVICE)

            logits = self.model(tensor)

            pred = torch.argmax(logits, dim=1).item()

            # کلاس 0 و 10 هر دو یعنی خانه خالی روی برد سودوکو
            digit = 0 if pred in EMPTY_CLASSES else pred

            board.append(digit)

        board = [board[i:i + 9] for i in range(0, 81, 9)]

        return board
