# from __future__ import annotations

# from pathlib import Path

# import cv2
# import torch
# from torchvision import transforms

# from src.digit_model import SudokuDigitCNN

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_PATH = Path("models/old_black_white_ds.pt")

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
# ])


# class DigitRecognizer:

#     def __init__(self):
#         checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

#         self.model = SudokuDigitCNN(num_classes=10)
#         self.model.load_state_dict(checkpoint["model_state_dict"])
#         self.model.to(DEVICE)
#         self.model.eval()

#     @torch.no_grad()
#     def predict_board(self, cells):

#         board = []

#         for cell in cells:

#             if cell.is_empty:
#                 board.append(0)
#                 continue

#             image = cv2.resize(cell.image, (28, 28))

#             tensor = transform(image).unsqueeze(0).to(DEVICE)

#             logits = self.model(tensor)

#             pred = torch.argmax(logits, dim=1).item()

#             board.append(pred)

#         board = [board[i:i + 9] for i in range(0, 81, 9)]

#         return board


from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ==================== تعریف معماری مدل LeNet ====================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== تنظیمات اولیه ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/mnist_lenet.pt")

# تبدیلات مورد نیاز برای MNIST (دقیقاً مثل کد شما)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ==================== کلاس اصلی تشخیص‌دهنده ====================
class DigitRecognizer:
    def __init__(self):
        # بارگذاری مدل دقیقاً مثل کد شما
        self.model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        self.model.eval()
        print("✅ مدل LeNet با موفقیت بارگذاری شد!")

    @torch.no_grad()
    def predict_board(self, cells):
        board = []
        
        for cell in cells:
            if cell.is_empty:
                board.append(0)
                continue
            
            # پردازش تصویر سلول (دقیقاً مثل کد شما)
            image = cv2.resize(cell.image, (28, 28))
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # پیش‌بینی
            logits = self.model(tensor)
            pred = torch.argmax(logits, dim=1).item()
            
            board.append(pred)
        
        # تبدیل به ماتریس 9x9
        board = [board[i:i + 9] for i in range(0, 81, 9)]
        return board