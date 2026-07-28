# Retail Intelligence Agent

تحويل فيديوهات كاميرات المطاعم والمتاجر إلى مؤشرات تشغيلية قابلة للمراجعة: كثافة العملاء داخل المناطق، طول الطابور، التنبيهات، ومسارات الحركة داخل كل كاميرا.

> **حالة المشروع: الـpipeline الأساسي يعمل لكل كاميرا بصورة مستقلة.** لا توجد مطابقة هوية بين كاميرتين ضمن مخرجاته أو الـDashboard، ولا تُحسب أرقامه كعدد زوار فريد على مستوى المتجر. يوجد نوتبوك بحثي منفصل باسم `02b_Global_Identity_Demo_Mode.ipynb` لا يغذي هذا المسار، وكل ناتجه معلّم صراحةً بأنه Demo.

## ما الذي نُفّذ؟

```text
فيديو CCTV
  -> YOLO11m لاكتشاف الأشخاص + ByteTrack
  -> نقطة القدم و local_track_id لكل كاميرا
  -> Homography + مناطق المكان
  -> تحليلات حركة وطوابير داخل الكاميرا
  -> CSV + DuckDB + Streamlit Dashboard
```

| المكوّن | الحالة | التفاصيل |
| --- | --- | --- |
| اكتشاف وتتبع الأشخاص | مُنفّذ | `YOLO11m` وByteTrack؛ تُحفظ فقط الاكتشافات بثقة `0.25` فأعلى. |
| هوية التتبع | مُنفّذ ومحلي | `local_track_id` و`camera_track_uid` فريدان داخل الكاميرا فقط، وليسَا هوية شخص عبر الكاميرات. |
| إسناد المناطق | مُنفّذ | نقطة القدم تمر عبر Homography ثم Polygon zones. |
| تحليلات التشغيل | مُنفّذ | ازدحام المناطق، سجل الطابور، baseline بسيط، تنبيهات، مسافة/زمن مكوث، وانتقالات بين المناطق داخل الكاميرا. |
| التخزين والعرض | مُنفّذ | CSV وDuckDB وتطبيق Streamlit بفلاتر المكان/الكاميرا ومسارات وSankey. |
| سؤال عربي عن البيانات | مُنفّذ محليًا | اختياري عبر Ollama: يولّد SQL مقيدًا بـ`SELECT` على DuckDB ثم يجيب من النتيجة الفعلية فقط. |
| Re-ID أو Global Fusion | غير مُنفّذ عمدًا | لا ربط عبر الكاميرات في هذا الـMVP. |
| Global Identity Demo | تجريبي ومنفصل | `02b` يكتب `global_tracks_demo.csv` فقط؛ لا يستخدم ظهور الشخص وحده كهوية نهائية. |
| عامل/زبون بمنطقة عمل | مُنفّذ بإعداد يدوي | `01b` يحسب role محليًا ويكتب ملفات وفيديوهات جديدة من دون تغيير التتبع الأصلي. |

## نتيجة آخر تشغيل محفوظ

هذه الأرقام تصف الـartifacts الموجودة محليًا في **28 يوليو 2026**، وليست قياس دقة نهائيًا:

| بند | النتيجة |
| --- | --- |
| المقاطع التي عالجها Notebook 01 | مقطعان: `CAFE_place_05_camera_17_15min` و`CAFE_place_05_camera_18_15min` |
| طول كل مقطع | 900 ثانية، 5 FPS، 4,500 frame |
| صفوف التتبع المحلي | 93,362 صفًا |
| أحداث المناطق | 95,560 صفًا |
| صفوف قياسات الحركة | 187 صفًا |
| انتقالات المناطق | انتقال واحد محفوظ |

### تنبيه مهم عن حالة النتائج الحالية

تم تشغيل Notebook 01 مرة أخرى بعد إنشاء ملفات المرحلتين 02 و03؛ لذلك `local_tracking_run.json` أحدث من `zone_events.csv` وجداول التحليلات. التطبيق يرفض هذه النتائج الأقدم تلقائيًا حتى لا يعرض Dashboard بيانات لا تتوافق مع أحدث تتبع. شغّل **Notebook 02 ثم Notebook 03** (أو `05_Run_All.ipynb`) لتجديدها.

كما أن الـsnapshot المخزن يضع `store_id=default_store` رغم أن أسماء الكاميرات تشير إلى `place_05`. كود Notebook 02 الحالي يستخرج `place_05` من اسم الكاميرا؛ إعادة المرحلتين 02 و03 مطلوبة أيضًا لتجديد هذا الحقل بشكل متسق.

## حدود النسخة الحالية وما ينقصها

- قيم `floor_x` و`floor_y` الحالية بوحدة `reference_pixels`؛ لذلك المسافة **ليست مترًا** ولا تصلح لقياس مسافات حقيقية قبل معايرة أرضية فعلية.
- لا يوجد Re-ID أو OSNet في مسار التشغيل، ولا يجوز جمع `camera_track_uid` من كاميرات مختلفة باعتباره شخصًا واحدًا.
- يحتاج كل نشر فعلي إلى Homography موثق وpolygons مناطق خاصة بالموقع. الملف الحالي يحتوي معايرات لكاميرات 17–20 من `place_05`، لكن لا يغني عن تحقق ميداني.
- لا توجد حتى الآن مقارنة مع ground truth أو تقييم كمي للدقة، ولا تكامل POS أو WhatsApp/Telegram.
- توقع الطابور الحالي baseline قصير المدى مبني على آخر تغير، وليس نموذج تعلم مُدرّب.
- Ollama اختياري ومحلي؛ يحتاج تشغيل الخدمة وتنزيل النموذج، وليس بديلًا عن LLM/RAG مستضاف.

## تنظيم المشروع

```text
Retail_Intelligence_Agent/
├── Data/
│   ├── raw/                         # فيديوهات الإدخال - غير مرفوعة إلى GitHub
│   ├── config/                      # المناطق، المعايرة، وإعدادات ByteTrack
│   └── models/                      # أوزان اختيارية - غير مرفوعة
├── Notebook/
│   ├── 00_Project_Setup.ipynb
│   ├── 01_Local_Detection_Tracking.ipynb
│   ├── 01b_Staff_Customer_Zones.ipynb         # اختياري؛ لا يغيّر local_tracks.csv
│   ├── 02_Global_Fusion_and_Zones.ipynb  # الاسم تاريخي؛ تعمل محليًا لكل كاميرا
│   ├── 02b_Global_Identity_Demo_Mode.ipynb    # بحثي/Demo؛ لا يغذي Dashboard
│   ├── 03_Retail_Analytics_Agent.ipynb
│   ├── 04_Streamlit_Dashboard.ipynb
│   └── 05_Run_All.ipynb
├── Output/
│   ├── app/streamlit_app.py
│   ├── tables/                      # نواتج تشغيل محلية - غير مرفوعة
│   ├── database/                    # DuckDB محلي - غير مرفوع
│   └── videos/                      # فيديوهات مشروحة محلية - غير مرفوعة
├── tests/
├── requirements.txt
└── README.md
```

## تشغيل المشروع

### 1. المتطلبات

- Python 3.10 أو 3.11 موصى به.
- PyTorch وTorchvision متوافقان مع جهازك. استخدم حزمة PyTorch المناسبة للـCPU أو CUDA؛ لا تخلط نسخة CUDA من `torch` مع نسخة CPU من `torchvision`.
- ضع `yolo11m.pt` في جذر المشروع أو في `Notebook/`. الأوزان لا تُرفع إلى GitHub.
- ضع المقاطع المطلوبة مباشرة داخل `Data/raw/`. نوتبوك `05_Run_All.ipynb` يتجاهل المجلدات الفرعية عمدًا، حتى لا يعمل على كاميرا بلا معايرة واضحة.

في PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

إذا أردت استخدام السؤال العربي الحر داخل Dashboard، ثبّت وشغّل Ollama ثم نزّل النموذج الافتراضي:

```powershell
ollama pull qwen2.5-coder:7b
ollama serve
```

### 2. إعداد المناطق والمعايرة

1. عدّل `Data/config/store_zones.json` بحيث تتطابق الـpolygons مع خريطة الأرضية.
2. أنشئ/تحقق من `Data/config/camera_calibration.generated.json` عبر `Notebook/calibrate_homography.py`.
3. لا تستخدم identity homography إلا للكاميرا المرجعية المعلّمة صراحةً، ولا تفسر الناتج كوحدة متر.

### 3. تنفيذ الـpipeline

نفّذ النوتبوكات بهذا الترتيب:

1. `00_Project_Setup.ipynb`
2. `01_Local_Detection_Tracking.ipynb`
3. `02_Global_Fusion_and_Zones.ipynb`
4. `03_Retail_Analytics_Agent.ipynb`
5. `04_Streamlit_Dashboard.ipynb`

أو افتح `05_Run_All.ipynb` لتشغيلها بالتسلسل. بعد إعادة تشغيل Notebook 01 يجب دائمًا إعادة 02 و03.

### 4. تشغيل الـDashboard

```powershell
python -m streamlit run Output\app\streamlit_app.py
```

ثم افتح `http://localhost:8501`. لا تشغّل `streamlit_app.py` باستخدام `python` وحده.

### تجارب اختيارية لا تغيّر الـpipeline

- `01b_Staff_Customer_Zones.ipynb` يقرأ `Data/config/staff_zones.json`. ابدأ بالملف الفارغ الآمن، ثم حدّد polygon لمنطقة العمل لكل كاميرا باستخدام:

```powershell
python Notebook\pick_staff_zone.py CAFE_place_05_camera_17_15min
```

  الإحداثيات بكسل الكاميرا، ويظل أي track بلا polygon عاملًا كـcustomer افتراضيًا إلا لو وجد `manual_override` في `employee_local_ids.json`.

- `02b_Global_Identity_Demo_Mode.ipynb` ينشئ `global_tracks_demo.csv` فقط. إذا لم توجد كاميرتان في `local_tracks.csv` فالناتج الصحيح هو صفر matches؛ لا يختلق Matches. وعند غياب calibration موثق، يعتمد Demo Mode على appearance + عدم التداخل الزمني فقط ويظل `is_demo_mode=True` في كل صف.

### 5. الاختبارات

بعد تثبيت بيئة Python:

```powershell
python -m pytest -q
```

الاختبارات تتحقق من أن pipeline يظل camera-local ولا يعيد إدخال Re-ID أو دمج الهوية بين الكاميرات.

## ملفات الإخراج الرئيسية

| ملف | محتواه |
| --- | --- |
| `local_tracks.csv` | نقاط القدم وIDs المحلية من مرحلة التتبع. |
| `video_metadata.csv` | FPS، frames، ومدة كل فيديو. |
| `zone_events.csv` | نقاط المسار بعد تحويلها إلى منطقة. |
| `movement_metrics.csv` | مسافة وزمن مكوث لكل track محلي/منطقة. |
| `zone_transitions.csv` | انتقالات المناطق المتتابعة داخل الكاميرا. |
| `zone_traffic.csv`, `queue_history.csv`, `agent_actions.csv` | مؤشرات التشغيل والطوابير والتنبيهات. |
| `retail_intelligence.duckdb` | نسخة DuckDB التي يقرأ منها سؤال Ollama الآمن. |

## الخصوصية والاستخدام المسؤول

المشروع يعتمد على تتبع مجهول داخل الكاميرا ولا يخزن وجوهًا أو أسماء أشخاص. قبل أي استخدام فعلي، التزم بقوانين الخصوصية وإشعارات الكاميرات وسياسات الاحتفاظ بالفيديو الخاصة بالموقع.
