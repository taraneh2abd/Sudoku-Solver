import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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


# ==================== کلاس آموزش (اصلاح‌شده) ====================

class SudokuTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.class_names = ['Empty_Clean'] + [str(i) for i in range(1, 10)] + ['Empty_Hints']

        print(f"📱 دستگاه: {self.device}")
        print(f"📐 تعداد پارامترها: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, X_train, y_train, X_test, y_test, epochs=6, lr=0.001, patience=10, batch_size=256):
        """
        آموزش مدل با داده‌های numpy
        """

        print("\n" + "="*70)
        print("🚀 شروع آموزش مدل")
        print("="*70)

        # ۱. تبدیل به tensor + نرمال‌سازی به بازه ۰ تا ۱
        # نکته مهم: تصاویر خام مقادیر ۰ تا ۲۵۵ دارند. اگر بدون نرمال‌سازی
        # به شبکه داده شوند، به‌خاطر مقیاس بزرگ ورودی همراه با BatchNorm
        # پشت‌سرهم و lr=0.001، فعال‌سازی‌ها خیلی سریع منفجر شده و NaN می‌شوند
        # که باعث می‌شود مدل همیشه فقط کلاس ۰ (Empty_Clean) را پیش‌بینی کند.
        X_train_tensor = torch.FloatTensor(X_train / 255.0).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test / 255.0).unsqueeze(1)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # بررسی سریع برای اطمینان از نبود NaN در ورودی
        assert not torch.isnan(X_train_tensor).any(), "❌ ورودی X_train شامل NaN است!"
        assert not torch.isnan(X_test_tensor).any(), "❌ ورودی X_test شامل NaN است!"

        # ۲. ایجاد دیتاست کامل
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # ۳. تقسیم train به train/val با random_split (تصادفی!)
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # برای تکرارپذیری
        )

        # ۴. ایجاد DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"\n📊 تقسیم داده‌ها:")
        print(f"   Train: {len(train_subset)} نمونه")
        print(f"   Val: {len(val_subset)} نمونه")
        print(f"   Test: {len(test_dataset)} نمونه")
        print(f"   Batch Size: {batch_size}")

        # ۵. آموزش
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # آموزش
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for data, target in train_bar:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                # اگر با وجود نرمال‌سازی بازهم loss خراب شد، فوراً متوجه شویم
                if torch.isnan(loss):
                    print("\n❌ هشدار: Loss برابر NaN شد! آموزش متوقف می‌شود.")
                    print("   بررسی کنید داده‌های ورودی سالم باشند (بدون NaN/Inf).")
                    return best_val_acc, 0.0

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                train_bar.set_postfix({
                    'loss': f'{train_loss/(train_total/target.size(0)):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })

            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total

            # اعتبارسنجی
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            with torch.no_grad():
                for data, target in val_bar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

                    val_bar.set_postfix({
                        'loss': f'{val_loss/(val_total/target.size(0)):.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })

            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total

            # فقط چاپ دقت روی تست در این epoch (بدون تاثیر روی early stopping یا ذخیره مدل)
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
            test_acc_epoch = 100. * test_correct / test_total

            self.train_losses.append(train_loss_avg)
            self.val_losses.append(val_loss_avg)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            scheduler.step(val_loss_avg)

            print(f'\n📊 Epoch {epoch+1}/{epochs}')
            print(f'   Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'   Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'   Test Acc: {test_acc_epoch:.2f}%')
            print(f'   LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'best_sudoku_model_11classes.pth')
                print(f'   ✅ بهترین مدل ذخیره شد! (دقت: {val_acc:.2f}%)')
            else:
                patience_counter += 1
                print(f'   ⚠️ بدون بهبود ({patience_counter}/{patience})')

                if patience_counter >= patience:
                    print(f'\n⏹ Early Stopping در epoch {epoch+1}')
                    break

            print("-" * 70)

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        print(f"\n✅ آموزش کامل شد! زمان: {training_time/60:.2f} دقیقه")
        print(f"🏆 بهترین دقت اعتبارسنجی: {best_val_acc:.2f}%")

        self.plot_curves()

        # ارزیابی روی تست
        test_acc = self.evaluate(test_loader)

        return best_val_acc, test_acc

    def plot_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.train_accs, label='Train Acc', linewidth=2)
        axes[1].plot(self.val_accs, label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curves', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 نمودارها ذخیره شد: training_curves.png")

    def evaluate(self, test_loader):
        print("\n" + "="*70)
        print("🔍 ارزیابی مدل روی داده‌های تست")
        print("="*70)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='ارزیابی'):
                data = data.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = np.mean(all_preds == all_labels) * 100
        print(f"\n🎯 دقت کلی روی تست: {accuracy:.2f}%")

        print("\n📊 گزارش طبقه‌بندی:")
        print(classification_report(all_labels, all_preds,
                                   target_names=self.class_names, digits=4))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   square=True)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 ماتریس درهم‌ریختگی ذخیره شد: confusion_matrix.png")

        return accuracy


# ==================== اجرا ====================

def main():
    print("="*70)
    print("🎯 آموزش مدل تشخیص اعداد سودوکو - ۱۱ کلاس")
    print("="*70)

    # بارگذاری دیتاست
    dataset_path = 'mixed_dataset_final'

    X_train = np.load(os.path.join(dataset_path, 'X_train.npy'))
    y_train = np.load(os.path.join(dataset_path, 'y_train.npy'))
    X_test = np.load(os.path.join(dataset_path, 'X_test.npy'))
    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))

    print(f"\n📂 دیتاست بارگذاری شد:")
    print(f"   Train: {len(X_train)} نمونه")
    print(f"   Test: {len(X_test)} نمونه")

    # ساخت مدل
    model = SudokuCNN(num_classes=11)

    # آموزش
    trainer = SudokuTrainer(model, device='cuda')
    best_val_acc, test_acc = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=6,
        lr=0.001,
        patience=10,
        batch_size=256
    )

    print("\n" + "="*70)
    print("🏆 نتایج نهایی")
    print("="*70)
    print(f"   بهترین دقت اعتبارسنجی: {best_val_acc:.2f}%")
    print(f"   دقت روی تست: {test_acc:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
