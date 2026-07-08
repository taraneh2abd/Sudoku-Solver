import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import random
import gzip
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== بخش ۱: خواندن دیتاست MNIST ====================

class MNISTLoader:
    """بارگذاری دیتاست MNIST"""
    
    def __init__(self, mnist_path=None):
        if mnist_path is None:
            possible_paths = [
                'MNIST/raw',
                '../FINAL/Sudoku-Solver/data/MNIST/raw',
                './MNIST/raw',
                './data/MNIST/raw',
                'C:/Users/T.Abdellahi/Desktop/term8/vision/proj/FINAL/Sudoku-Solver/data/MNIST/raw',
                'C:/Users/T.Abdellahi/Desktop/term8/vision/proj/FINAL/Sudoku-Solver/data/raw',
            ]
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.mnist_path = path
                    found = True
                    print(f"✅ مسیر MNIST پیدا شد: {path}")
                    break
            
            if not found:
                print("❌ مسیر MNIST پیدا نشد!")
                user_path = input("لطفاً مسیر دقیق پوشه MNIST/raw رو وارد کن: ").strip()
                if os.path.exists(user_path):
                    self.mnist_path = user_path
                else:
                    raise FileNotFoundError(f"مسیر {user_path} وجود ندارد!")
        else:
            self.mnist_path = mnist_path
        
        print(f"📂 مسیر MNIST: {self.mnist_path}")
        
    def read_idx(self, filename):
        """خواندن فایل‌های idx فرمت MNIST"""
        if not os.path.exists(filename):
            if os.path.exists(filename + '.gz'):
                filename = filename + '.gz'
            else:
                raise FileNotFoundError(f"فایل {filename} یا {filename}.gz پیدا نشد!")
        
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                data = f.read()
        else:
            with open(filename, 'rb') as f:
                data = f.read()
        
        magic = int.from_bytes(data[:4], 'big')
        dims = []
        offset = 4
        
        for _ in range(magic & 0xFF):
            dims.append(int.from_bytes(data[offset:offset+4], 'big'))
            offset += 4
        
        data_array = np.frombuffer(data[offset:], dtype=np.uint8)
        data_array = data_array.reshape(dims)
        
        return data_array
    
    def load_data(self, samples_per_digit=5000):
        """بارگذاری داده‌ها (فقط ارقام ۱ تا ۹) با تعداد مشخص"""
        print("📂 بارگذاری دیتاست MNIST (فقط ۱ تا ۹)...")
        
        train_images = os.path.join(self.mnist_path, 'train-images-idx3-ubyte')
        train_labels = os.path.join(self.mnist_path, 'train-labels-idx1-ubyte')
        test_images = os.path.join(self.mnist_path, 't10k-images-idx3-ubyte')
        test_labels = os.path.join(self.mnist_path, 't10k-labels-idx1-ubyte')
        
        X_train = self.read_idx(train_images)
        y_train = self.read_idx(train_labels)
        X_test = self.read_idx(test_images)
        y_test = self.read_idx(test_labels)
        
        X_all = np.concatenate([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        
        # فقط ارقام ۱ تا ۹
        mask = (y_all >= 1) & (y_all <= 9)
        X_filtered = X_all[mask]
        y_filtered = y_all[mask]
        
        # انتخاب تعداد مشخص از هر کلاس
        X_selected = []
        y_selected = []
        for digit in range(1, 10):
            digit_mask = (y_filtered == digit)
            digit_indices = np.where(digit_mask)[0]
            
            if len(digit_indices) > samples_per_digit:
                selected = np.random.choice(digit_indices, samples_per_digit, replace=False)
            else:
                selected = digit_indices
            
            X_selected.append(X_filtered[selected])
            y_selected.append(y_filtered[selected])
        
        X_filtered = np.concatenate(X_selected)
        y_filtered = np.concatenate(y_selected)
        
        print(f"✅ MNIST بارگذاری شد! {len(X_filtered)} نمونه (۵۰۰۰ از هر رقم)")
        
        return X_filtered, y_filtered


# ==================== بخش ۲: تولید دیتاست سودوکو ====================

class SudokuDatasetGenerator:
    """تولید دیتاست سودوکو"""
    
    def __init__(self, img_size=39):
        self.img_size = img_size
        self.fonts = self._get_fonts()
        
    def _get_fonts(self):
        """لیست فونت‌های موجود در سیستم"""
        possible_fonts = [
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf',
            '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/Arial.ttf',
            'C:/Windows/Fonts/Arial.ttf',
            'C:/Windows/Fonts/Calibri.ttf',
            'C:/Windows/Fonts/Tahoma.ttf',
            'C:/Windows/Fonts/Verdana.ttf',
        ]
        
        font_paths = [f for f in possible_fonts if os.path.exists(f)]
        
        if not font_paths:
            font_paths = [None]
        return font_paths
    
    def _apply_rotation(self, img, angle_range=(-6, 6)):
        """چرخش کل تصویر"""
        angle = random.uniform(*angle_range)
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    def _add_noise(self, img):
        """نویز ملایم"""
        if random.random() < 0.2:
            noise = np.random.normal(0, random.randint(1, 3), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img
    
    def generate_digit_manual(self, digit, num_samples):
        """تولید اعداد دستی با فونت‌های مختلف"""
        X, y = [], []
        
        for _ in tqdm(range(num_samples), desc=f"   ساخت عدد {digit}"):
            font_path = random.choice(self.fonts)
            font_size = random.randint(24, 38)
            
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            img = Image.new('L', (self.img_size, self.img_size), color=0)
            draw = ImageDraw.Draw(img)
            
            bbox = draw.textbbox((0, 0), str(digit), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)
            pos_x = (self.img_size - text_width) // 2 + offset_x
            pos_y = (self.img_size - text_height) // 2 + offset_y
            
            draw.text((pos_x, pos_y), str(digit), fill=255, font=font)
            img_np = np.array(img)
            
            # ۴۰٪ نمونه‌ها تغییرات ملایم
            if random.random() < 0.4:
                img_np = self._apply_rotation(img_np, (-8, 8))
                img_np = self._add_noise(img_np)
            
            X.append(img_np)
            y.append(digit)
        
        return np.array(X), np.array(y)
    
    def generate_empty_with_lines(self, num_samples):
        """تولید empty با خطوط عمودی/افقی در حاشیه‌ها"""
        X, y = [], []
        
        for _ in tqdm(range(num_samples), desc="   ساخت empty با خطوط"):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            line_combination = random.choice([
                'top', 'bottom', 'left', 'right',
                'top_bottom', 'left_right',
                'top_left', 'top_right', 'bottom_left', 'bottom_right',
                'top_bottom_left', 'top_bottom_right',
                'top_left_right', 'bottom_left_right',
                'top_bottom_left_right',
            ])
            
            thickness = random.randint(1, 3)
            intensity = random.randint(200, 240)
            
            # خط بالا
            if 'top' in line_combination:
                y_pos = random.randint(0, min(6, self.img_size // 5))
                start_x = random.randint(0, 3)
                end_x = self.img_size - random.randint(0, 3)
                skew = random.randint(-2, 2)
                cv2.line(img, (start_x, y_pos), (end_x + skew, y_pos + thickness), intensity, thickness)
            
            # خط پایین
            if 'bottom' in line_combination:
                y_pos = self.img_size - random.randint(0, min(6, self.img_size // 5)) - thickness
                start_x = random.randint(0, 3)
                end_x = self.img_size - random.randint(0, 3)
                skew = random.randint(-2, 2)
                cv2.line(img, (start_x, y_pos), (end_x + skew, y_pos + thickness), intensity, thickness)
            
            # خط چپ
            if 'left' in line_combination:
                x_pos = random.randint(0, min(6, self.img_size // 5))
                start_y = random.randint(0, 3)
                end_y = self.img_size - random.randint(0, 3)
                skew = random.randint(-2, 2)
                cv2.line(img, (x_pos, start_y), (x_pos + thickness, end_y + skew), intensity, thickness)
            
            # خط راست
            if 'right' in line_combination:
                x_pos = self.img_size - random.randint(0, min(6, self.img_size // 5)) - thickness
                start_y = random.randint(0, 3)
                end_y = self.img_size - random.randint(0, 3)
                skew = random.randint(-2, 2)
                cv2.line(img, (x_pos, start_y), (x_pos + thickness, end_y + skew), intensity, thickness)
            
            # چرخش ملایم
            if random.random() < 0.15:
                img = self._apply_rotation(img, (-3, 3))
            
            # نویز ملایم
            if random.random() < 0.2:
                img = self._add_noise(img)
            
            X.append(img)
            y.append(0)
        
        return np.array(X), np.array(y)
    
    def generate_empty_hints(self, num_samples):
        """تولید empty_hints روشن‌تر با شبکه 3x3"""
        X, y = [], []
        
        cell_size = self.img_size // 3
        
        for _ in tqdm(range(num_samples), desc="   ساخت empty_hints"):
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            img_pil = Image.fromarray(img, mode='L')
            draw = ImageDraw.Draw(img_pil)
            
            font_path = random.choice(self.fonts)
            font_size = random.randint(12, 17)
            
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            num_hints = random.randint(1, 8)
            hint_positions = random.sample(range(9), num_hints)
            
            for idx in range(9):
                row = idx // 3
                col = idx % 3
                
                x1 = col * cell_size
                y1 = row * cell_size
                
                if idx in hint_positions:
                    digit = idx + 1
                    
                    bbox = draw.textbbox((0, 0), str(digit), font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    pos_x = x1 + (cell_size - text_width) // 2 + random.randint(-2, 2)
                    pos_y = y1 + (cell_size - text_height) // 2 + random.randint(-2, 2)
                    
                    gray_value = random.randint(200, 240)
                    draw.text((pos_x, pos_y), str(digit), fill=gray_value, font=font)
            
            img_np = np.array(img_pil)
            
            if random.random() < 0.2:
                img_np = self._apply_rotation(img_np, (-6, 6))
            
            if random.random() < 0.2:
                img_np = self._add_noise(img_np)
            
            X.append(img_np)
            y.append(10)
        
        return np.array(X), np.array(y)


# ==================== بخش ۳: ساخت دیتاست نهایی ====================

class MixedDatasetBuilder:
    """ساخت دیتاست نهایی با ۱۵۰۰۰ نمونه از هر کلاس"""
    
    def __init__(self, img_size=39, output_dir='mixed_dataset_final', mnist_path=None):
        self.img_size = img_size
        self.output_dir = output_dir
        self.mnist_path = mnist_path
        self.class_names = ['Empty_Clean'] + [str(i) for i in range(1, 10)] + ['Empty_Hints']
        self.samples_per_class = 15000
        
    def build_dataset(self):
        """ساخت دیتاست نهایی"""
        
        print("="*70)
        print("🚀 ساخت دیتاست نهایی (۱۵۰۰۰ نمونه از هر کلاس)")
        print("="*70)
        
        # ۱. بارگذاری MNIST (۵۰۰۰ از هر رقم)
        mnist_loader = MNISTLoader(mnist_path=self.mnist_path)
        X_mnist, y_mnist = mnist_loader.load_data(samples_per_digit=5000)
        
        # ۲. تولید داده‌های دستی (۱۰۰۰۰ از هر رقم)
        print("\n🔄 تولید داده‌های دستی (۱۰۰۰۰ از هر رقم)...")
        sudoku_gen = SudokuDatasetGenerator(img_size=self.img_size)
        
        X_manual = []
        y_manual = []
        
        for digit in range(1, 10):
            X_digit, y_digit = sudoku_gen.generate_digit_manual(digit, 10000)
            X_manual.append(X_digit)
            y_manual.append(y_digit)
        
        X_manual = np.concatenate(X_manual)
        y_manual = np.concatenate(y_manual)
        print(f"   داده‌های دستی: {len(X_manual)} نمونه")
        
        # ۳. تغییر اندازه MNIST به 39x39
        print("\n🔄 تغییر اندازه MNIST به 39x39...")
        X_mnist_resized = []
        for img in tqdm(X_mnist, desc="   تغییر اندازه"):
            img_resized = cv2.resize(img, (self.img_size, self.img_size), 
                                    interpolation=cv2.INTER_AREA)
            X_mnist_resized.append(img_resized.astype(np.float32))
        X_mnist_resized = np.array(X_mnist_resized)
        print(f"   MNIST تغییر اندازه داده شد: {len(X_mnist_resized)} نمونه")
        
        # ۴. تولید empty (۱۵۰۰۰ نمونه)
        print("\n🔄 تولید empty با خطوط حاشیه‌ای (۱۵۰۰۰ نمونه)...")
        X_empty, y_empty = sudoku_gen.generate_empty_with_lines(15000)
        
        # ۵. تولید empty_hints (۱۵۰۰۰ نمونه)
        print("\n🔄 تولید empty_hints روشن‌تر (۱۵۰۰۰ نمونه)...")
        X_hints, y_hints = sudoku_gen.generate_empty_hints(15000)
        
        # ۶. ترکیب ارقام (MNIST + دستی)
        print("\n🔄 ترکیب ارقام...")
        X_digits = np.concatenate([X_mnist_resized, X_manual])
        y_digits = np.concatenate([y_mnist, y_manual])
        print(f"   ارقام: {len(X_digits)} نمونه")
        
        # ۷. ترکیب نهایی همه داده‌ها
        print("\n🔄 ترکیب همه داده‌ها...")
        X_all = np.concatenate([X_digits, X_empty, X_hints])
        y_all = np.concatenate([y_digits, y_empty, y_hints])
        
        # ۸. شافل کردن کامل
        print("\n🔄 شافل کردن داده‌ها...")
        indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]
        
        print(f"\n✅ دیتاست کامل: {len(X_all)} نمونه")
        
        # ۹. نمایش توزیع
        unique, counts = np.unique(y_all, return_counts=True)
        print("\n📊 توزیع کلاس‌ها در دیتاست کامل:")
        print("-" * 50)
        total = len(X_all)
        for cls, count in zip(unique, counts):
            if cls == 0:
                name = 'Empty_Clean'
            elif cls == 10:
                name = 'Empty_Hints'
            else:
                name = f'Digit_{cls}'
            print(f"   {name:>15}: {count:>7} ({count/total*100:>5.1f}%)")
        print("-" * 50)
        print(f"   {'مجموع':>15}: {total:>7} نمونه")
        
        # ۱۰. تقسیم به train/test با stratify
        print("\n🔄 تقسیم به Train/Test (85%/15%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, 
            test_size=0.15, 
            random_state=42, 
            stratify=y_all
        )
        
        print(f"\n   Train: {len(X_train)} نمونه")
        print(f"   Test: {len(X_test)} نمونه")
        
        # ۱۱. نمایش توزیع Train
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print("\n📊 توزیع Train:")
        for cls, count in zip(unique_train, counts_train):
            name = self.class_names[int(cls)]
            print(f"   {name}: {count} نمونه")
        
        # ۱۲. نمایش توزیع Test
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print("\n📊 توزیع Test:")
        for cls, count in zip(unique_test, counts_test):
            name = self.class_names[int(cls)]
            print(f"   {name}: {count} نمونه")
        
        # ۱۳. ایجاد پوشه و ذخیره
        print(f"\n💾 ذخیره دیتاست در پوشه '{self.output_dir}'...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ذخیره فایل‌های numpy
        print("   ذخیره فایل‌های numpy...")
        np.save(os.path.join(self.output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.output_dir, 'y_test.npy'), y_test)
        
        # (اختیاری) ذخیره کل دیتاست برای استفاده‌های بعدی
        np.save(os.path.join(self.output_dir, 'X_all.npy'), X_all)
        np.save(os.path.join(self.output_dir, 'y_all.npy'), y_all)
        
        print(f"\n   ✅ X_train.npy: {X_train.shape}")
        print(f"   ✅ y_train.npy: {y_train.shape}")
        print(f"   ✅ X_test.npy: {X_test.shape}")
        print(f"   ✅ y_test.npy: {y_test.shape}")
        print(f"   ✅ X_all.npy: {X_all.shape}")
        print(f"   ✅ y_all.npy: {y_all.shape}")
        
        # ۱۴. ذخیره ۲۰۰ نمونه از هر کلاس برای نمایش (اختیاری)
        print("\n💾 ذخیره ۲۰۰ نمونه تصویر از هر کلاس...")
        self.save_samples_by_class(X_train, y_train, max_per_class=200)
        
        print("\n" + "="*70)
        print("✅ ساخت دیتاست کامل شد!")
        print(f"📁 مسیر: {os.path.abspath(self.output_dir)}")
        print("="*70)
        
        return X_train, y_train, X_test, y_test
    
    def save_samples_by_class(self, X, y, max_per_class=200):
        """ذخیره نمونه‌های تصویر از هر کلاس"""
        
        for cls in range(11):
            class_name = self.class_names[cls]
            folder_path = os.path.join(self.output_dir, class_name)
            os.makedirs(folder_path, exist_ok=True)
            
            class_indices = np.where(y == cls)[0]
            if len(class_indices) == 0:
                continue
            
            # انتخاب حداکثر max_per_class نمونه
            num_samples = min(max_per_class, len(class_indices))
            selected_indices = np.random.choice(class_indices, num_samples, replace=False)
            
            saved_count = 0
            for i, idx in enumerate(selected_indices):
                img = X[idx]
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                filename = f"{class_name}_{i:06d}.png"
                filepath = os.path.join(folder_path, filename)
                Image.fromarray(img_uint8, mode='L').save(filepath, format='PNG')
                saved_count += 1
            
            print(f"   ✅ {class_name}: {saved_count} تصویر ذخیره شد")


# ==================== اجرا ====================

if __name__ == "__main__":
    
    # مسیر MNIST
    mnist_path = r"C:\Users\T.Abdellahi\Desktop\term8\vision\proj\FINAL\Sudoku-Solver\data\MNIST\raw"
    
    if not os.path.exists(mnist_path):
        mnist_path = r"C:\Users\T.Abdellahi\Desktop\term8\vision\proj\FINAL\Sudoku-Solver\data\raw"
    
    # ساخت دیتاست
    builder = MixedDatasetBuilder(
        img_size=39, 
        output_dir='mixed_dataset_final',
        mnist_path=mnist_path
    )
    
    X_train, y_train, X_test, y_test = builder.build_dataset()
    
    print("\n🎯 دیتاست آماده است! برای آموزش مدل از فایل‌های زیر استفاده کن:")
    print(f"   📁 {os.path.abspath('mixed_dataset_final')}")
    print("   📄 X_train.npy, y_train.npy")
    print("   📄 X_test.npy, y_test.npy")