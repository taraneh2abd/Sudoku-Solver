import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# تعریف معماری مدل LeNet
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

# بارگذاری مدل به روش امن
model = LeNet()
model = torch.load('models/mnist_lenet.pt', map_location=torch.device('cpu'), weights_only=False)
model.eval()
print("✅ مدل با موفقیت بارگذاری شد!")
# تبدیلات مورد نیاز
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image_path):
    """تشخیص عدد از روی عکس"""
    try:
        # بارگذاری و تبدیل عکس
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)
        
        # پیش‌بینی
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0][prediction].item() * 100
        
        return prediction.item(), confidence
    except Exception as e:
        return None, f"خطا: {e}"

# استفاده
if __name__ == "__main__":
    # image_path = input("📁 مسیر عکس را وارد کنید: ")
    image_path = r"C:\Users\T.Abdellahi\Desktop\term8\vision\proj\FINAL\Sudoku-Solver\data\processed\digits\train\1\000067_r1_c5.png"  # مسیر عکس را اینجا بگذارید

    if not os.path.exists(image_path):
        print("❌ فایل وجود ندارد!")
    else:
        digit, result = predict_digit(image_path)
        if digit is not None:
            print(f"✅ عدد تشخیص داده شده: {digit}")
            print(f"📊 دقت: {result:.2f}%")
        else:
            print(f"❌ {result}")