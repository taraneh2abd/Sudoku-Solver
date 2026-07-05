# روند پیش‌پردازش و استخراج جدول سودوکو

## مراحل کلی اجرای برنامه

```
تصویر ورودی
      │
      ▼
preprocess.py
      │
      ▼
extract() در grid_extraction.py
      │
      ├── Primary
      ├── Fallback
      └── Full Image (Last Resort)
      │
      ▼
اصلاح گوشه‌ها (_refine_corners)
      │
      ▼
Perspective Warp
      │
      ▼
cell_extraction.py
      │
      ▼
استخراج و ذخیره ۸۱ سلول
```

---

# مرحله اول: Preprocess

در فایل `preprocess.py` تصویر فقط **یک بار** آماده‌سازی می‌شود.

مسیر اصلی:

```
Image
 │
 ▼
Gray
 │
 ▼
CLAHE
 │
 ▼
Median Blur
 │
 ▼
Gaussian Blur
 │
 ▼
Adaptive Threshold
```

همزمان نسخه‌ای مخصوص تشخیص گرید نیز ساخته می‌شود (در صورت بزرگ بودن تصویر ابتدا Resize انجام می‌شود):

```
Image
 │
 ▼
Resize (در صورت نیاز)
 │
 ▼
Gray
 │
 ▼
CLAHE
 │
 ▼
Median Blur
 │
 ▼
Gaussian Blur
 │
 ▼
Adaptive Threshold
```

در پایان این فایل خروجی‌های زیر تولید می‌شوند:

* gray
* normalized
* blurred
* binary
* detect_blur
* detect_bin
* detect_scale

---

# مرحله دوم: Primary

ابتدا از نسخه Detect استفاده می‌شود.

```
detect_blur + detect_bin
        │
        ▼
Contour Detection
        │
        ├── موفق → پایان
        │
        └── ناموفق
                │
                ▼
          Hough Transform
                │
                ├── موفق → پایان
                └── ناموفق → ورود به Fallback
```

---

# مرحله سوم: Fallback

اگر Primary موفق نباشد، از تصویر `normalized` استفاده می‌شود.

دقت شود که **Median Blur عمداً حذف شده است**.

```
normalized
      │
      ▼
Gaussian Blur
      │
      ▼
Adaptive Threshold
      │
      ▼
Contour Detection
      │
      ├── موفق → پایان
      │
      └── ناموفق
              │
              ▼
        Hough Transform
              │
              ├── موفق → پایان
              └── ناموفق → Full Image
```

---

# مرحله چهارم: Full Image

اگر هیچ روشی موفق نشود، کل تصویر به عنوان جدول سودوکو فرض می‌شود.

```
Whole Image
      │
      ▼
Perspective Warp
```

---

# دلیل استفاده از normalized در Fallback

در `preprocess` دو تصویر مهم ساخته می‌شود:

```
gray
 │
 ▼
CLAHE
 │
 ├────────► normalized
 │
 ▼
Median Blur
 │
 ▼
Gaussian Blur
 │
 ▼
blurred
```

اگر Fallback از `blurred` استفاده می‌کرد، مسیر آن دقیقاً مشابه Primary می‌شد:

```
CLAHE
 ↓
Median
 ↓
Gaussian
```

در نتیجه هیچ تفاوتی با Primary نداشت.

به همین دلیل Fallback از `normalized` استفاده می‌کند و فقط Gaussian را مجدداً اعمال می‌کند:

```
CLAHE
 ↓
Gaussian
```

در این حالت اثر Median Blur حذف می‌شود و ممکن است خطوط یا لبه‌هایی که در Primary از بین رفته‌اند دوباره قابل تشخیص باشند.

---

# ترتیب کلی تصمیم‌گیری

```
Primary
│
├── Contour
│      ├── موفق → پایان
│      └── ناموفق
│
└── Hough
       ├── موفق → پایان
       └── ناموفق
              │
              ▼
Fallback
│
├── Contour
│      ├── موفق → پایان
│      └── ناموفق
│
└── Hough
       ├── موفق → پایان
       └── ناموفق
              │
              ▼
Full Image
```

---

# تفاوت سه روش

| روش        | تصویر ورودی                            | Median Blur | Gaussian Blur | Threshold | تشخیص           |
| ---------- | -------------------------------------- | ----------- | ------------- | --------- | --------------- |
| Primary    | نسخه Detect (ممکن است Resize شده باشد) | ✓           | ✓             | ✓         | Contour → Hough |
| Fallback   | تصویر اصلی (`normalized`)              | ✗           | ✓             | ✓         | Contour → Hough |
| Full Image | کل تصویر                               | ✗           | ✗             | ✗         | بدون تشخیص      |

---

# توابعی که ممکن است دوباره اجرا شوند

اگر Primary موفق باشد، هیچ مرحله‌ای دوباره اجرا نمی‌شود.

اگر Primary شکست بخورد و Fallback اجرا شود، توابع زیر روی **نسخه متفاوتی از تصویر** دوباره اجرا می‌شوند:

* Gaussian Blur
* Adaptive Threshold
* `_locate_grid`
* `findContours`
* `approxPolyDP`
* `Hough Transform` (در صورتی که Contour موفق نباشد)

بنابراین عملیات تکراری روی **یک تصویر یکسان** انجام نمی‌شود؛ بلکه همان الگوریتم روی نسخه‌ای متفاوت از تصویر اجرا می‌شود تا احتمال موفقیت افزایش یابد.


# دیتاست های ما
https://huggingface.co/datasets/Lexski/sudoku-image-recognition/blob/main/README.md?code=true&utm_source=chatgpt.com


```bash
python -m src.train --data-dir data/processed/digits --epochs 5 --output models/new_ds.pt
python main.py C:\\Users\\T.Abdellahi\\Desktop\\term8\\vision\\proj\\FINAL\\Sudoku-Solver\\data\\train\\00026.jpg  
```

چیزهایی که داخل پایپلاین نیست و یک بار باید ران بشه دستی:
جنریت دیتاست
ترین  