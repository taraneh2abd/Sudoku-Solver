import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import numpy as np
import cv2  # برای پردازش بهتر تصاویر

# 1. تعریف معماری ResNet18 برای MNIST
class ResNet18MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18MNIST, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        self.resnet18.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)

# 2. بارگذاری مدل (همان کد قبلی)
def load_model():
    print("📥 در حال بارگذاری مدل...")
    
    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.fc1 = nn.Linear(16*4*4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.pool = nn.AvgPool2d(2, 2)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 16*4*4)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = LeNet5()
    
    try:
        model.load_state_dict(torch.load('mnist_lenet5.pth', map_location='cpu'))
        print("✅ مدل از فایل محلی بارگذاری شد!")
    except:
        print("ℹ️ مدل از ابتدا ساخته شد (بدون وزن آموزش‌دیده)")
        print("💡 برای دقت بالا باید مدل را آموزش دهید یا وزن‌ها را دانلود کنید")
    
    model.eval()
    return model

# 3. پیش‌پردازش پیشرفته برای تصاویر باینری 50×50
def preprocess_binary_image(image_path):
    """
    پیش‌پردازش ویژه برای تصاویر باینری با سایز 50×50
    که نرمالایز نشده‌اند
    """
    # 1. بارگذاری تصویر با OpenCV (بهتر از PIL برای تصاویر باینری)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"❌ تصویر در مسیر {image_path} پیدا نشد!")
    
    print(f"📐 سایز اصلی: {img.shape}")
    
    # 2. اطمینان از باینری بودن (آستانه‌گذاری)
    #    تصاویر شما باینری هستند اما برای اطمینان دوباره اعمال می‌کنیم
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 3. پیدا کردن کانتور رقم (برای کراپ دقیق)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # پیدا کردن بزرگترین کانتور (رقم اصلی)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # اضافه کردن حاشیه 2 پیکسل دور رقم
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(binary.shape[1] - x, w + 2*margin)
        h = min(binary.shape[0] - y, h + 2*margin)
        
        # کراپ کردن رقم
        cropped = binary[y:y+h, x:x+w]
        print(f"✂️ سایز بعد از کراپ: {cropped.shape}")
    else:
        # اگر کانتوری پیدا نشد، از کل تصویر استفاده کن
        cropped = binary
        print("⚠️ کانتوری پیدا نشد، از کل تصویر استفاده می‌شود")
    
    # 4. تغییر سایز به 28×28 با حفظ نسبت و قرارگیری در مرکز
    h, w = cropped.shape
    target_size = 28
    
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    # تغییر سایز با کیفیت بالا
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # ایجاد تصویر خالی 28×28
    final_img = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # قرار دادن رقم در مرکز
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 5. اینورت خودکار (تشخیص بر اساس میانگین پیکسل‌ها)
    #    MNIST: پس‌زمینه مشکی (0) و رقم سفید (255)
    if np.mean(final_img) > 127:
        final_img = 255 - final_img
        print("🔄 تصویر اینورت شد (پس‌زمینه سفید → مشکی)")
    else:
        print("✅ تصویر نیازی به اینورت ندارد")
    
    # 6. تبدیل به تنسور PyTorch و نرمال‌سازی با مقادیر MNIST
    #    تبدیل به تنسور (0-255 → 0-1)
    transform = transforms.Compose([
        transforms.ToTensor(),  # این کار نرمال‌سازی اولیه را انجام می‌دهد
        transforms.Normalize((0.1307,), (0.3081,))  # نرمال‌سازی استاندارد MNIST
    ])
    
    # تبدیل numpy array به PIL Image برای استفاده از transforms
    pil_image = Image.fromarray(final_img)
    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)  # اضافه کردن بعد batch
    
    print(f"📏 سایز نهایی تنسور: {tensor.shape}")
    print(f"📊 محدوده مقادیر پیکسل: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    return tensor, final_img

# 4. تابع پیش‌بینی با پیش‌پردازش جدید
def predict_digit(image_path, model):
    """پیش‌بینی رقم با پیش‌پردازش پیشرفته"""
    # پیش‌پردازش با متد جدید
    tensor, processed_img = preprocess_binary_image(image_path)
    
    # نمایش تصویر پردازش‌شده (اختیاری)
    from matplotlib import pyplot as plt
    plt.imshow(processed_img, cmap='gray')
    plt.title('تصویر پس از پیش‌پردازش (28×28)')
    plt.axis('off')
    plt.show()
    
    # پیش‌بینی
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities

# 5. اجرای اصلی
def main():
    # بارگذاری مدل
    model = load_model()
    
    # مسیر تصویر شما
    image_path = r"C:\Users\T.Abdellahi\Desktop\term8\vision\proj\FINAL\Sudoku-Solver\data\processed\digits\train\1\000029_r8_c0.png"
    
    print(f"\n🔍 در حال تحلیل تصویر: {image_path}")
    print("-" * 50)
    
    try:
        digit, confidence, probabilities = predict_digit(image_path, model)
        print(f"✅ رقم پیش‌بینی‌شده: {digit}")
        print(f"📊 اطمینان مدل: {confidence*100:.2f}%")
        
        # نمایش احتمالات همه ارقام
        print("\n📊 احتمالات همه ارقام:")
        for i in range(10):
            print(f"  رقم {i}: {probabilities[0][i].item()*100:.2f}%")
            
    except Exception as e:
        print(f"❌ خطا در پردازش تصویر: {e}")

if __name__ == "__main__":
    main()