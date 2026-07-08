import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# ==================== تعریف مدل ====================

class SudokuCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SudokuCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.15)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.dropout4 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


# ==================== بارگذاری مدل ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 دستگاه: {device}")

# بارگذاری مدل
model = SudokuCNN(num_classes=11)
model.load_state_dict(torch.load('best_sudoku_model_11classes.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['Empty_Clean'] + [str(i) for i in range(1, 10)] + ['Empty_Hints']
print("✅ مدل بارگذاری شد!")

# ==================== پیش‌بینی ====================

# مسیر تصویر
image_path = r"C:\Users\T.Abdellahi\Desktop\term8\vision\proj\FINAL\Sudoku-Solver\results\cells\cell_r0c1.png"

# بارگذاری تصویر
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"❌ تصویر پیدا نشد!")
    exit()

# تغییر اندازه به 39x39
img_resized = cv2.resize(img, (39, 39), interpolation=cv2.INTER_AREA)

# نرمال‌سازی
img_normalized = img_resized.astype(np.float32) / 255.0

# تبدیل به tensor
img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

# پیش‌بینی
with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

predicted_class = predicted.item()
confidence_percent = confidence.item() * 100
class_name = class_names[predicted_class]

# ==================== نمایش نتیجه ====================

print("\n" + "="*50)
print("🔍 نتیجه پیش‌بینی")
print("="*50)
print(f"📷 تصویر: {image_path}")
print(f"🎯 کلاس: {predicted_class} ({class_name})")
print(f"📊 اطمینان: {confidence_percent:.2f}%")
print("="*50)

# نمایش احتمالات
print("\n📊 ۵ کلاس برتر:")
top5 = torch.topk(probabilities, 5)
for i in range(5):
    idx = top5.indices[0][i].item()
    prob = top5.values[0][i].item() * 100
    print(f"   {i+1}. {class_names[idx]}: {prob:.2f}%")

# ==================== نمایش تصویر ====================

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# تصویر اصلی
axes[0].imshow(img, cmap='gray')
axes[0].set_title('تصویر اصلی', fontsize=12)
axes[0].axis('off')

# تصویر تغییر اندازه شده
axes[1].imshow(img_resized, cmap='gray')
axes[1].set_title('تصویر 39x39', fontsize=12)
axes[1].axis('off')

# نتیجه
axes[2].axis('off')
result_text = f"""
╔════════════════════════════════╗
║     نتیجه پیش‌بینی              ║
╠════════════════════════════════╣
║  کلاس: {predicted_class}                   ║
║  برچسب: {class_name}  ║
║  اطمینان: {confidence_percent:.2f}%         ║
╚════════════════════════════════╝
"""
axes[2].text(0.1, 0.5, result_text, fontsize=12, 
            verticalalignment='center', fontfamily='monospace')

plt.tight_layout()
plt.show()