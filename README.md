# 🔥 نظام التنبؤ المبكر بالحريق والأعطال الكبرى في سنترالات الاتصالات

**مشروع تخرج — AI for Business | قطاع الاتصالات**
مستوحى من حادثة سنترال رمسيس، الهدف: التنبؤ بالخطر **قبل وقوعه بساعات/أيام** بدل الاكتفاء برصده لحظة الحدوث.

---

## 1. فكرة المشروع والمشكلة

الأعطال الكبرى (كحريق سنترال) نادرًا ما تحدث فجأة — عادة تسبقها فترة "تدهور تدريجي":
ارتفاع تدريجي في الحرارة، تذبذب الجهد، ازدياد الحمل الكهربائي، اهتزاز غير طبيعي في المعدات...
المشكلة أن هذه الإشارات مبعثرة عبر عشرات الحساسات ولا يلاحظها الإنسان في الوقت المناسب.

**الحل:** نظام AI يراقب الحساسات لحظيًا، يتعلم "بصمة ما قبل الحادثة"، ويرسل إنذارًا مبكرًا
لفريق الصيانة **قبل** أن يتحول التدهور إلى كارثة.

---

## 2. خطة العمل (Roadmap) خطوة بخطوة

| # | المرحلة | الوصف | الملف المسؤول |
|---|---------|-------|----------------|
| 1 | **جمع/توليد البيانات** | بيانات حساسات (حرارة، دخان، حمل كهرباء، اهتزاز...) + بيانات صناعية واقعية لو لا تتوفر بيانات حقيقية كافية | `src/generate_synthetic_data.py` |
| 2 | **Preprocessing** | تنظيف، حدود فيزيائية، معالجة Missing Values بالـ Interpolation الزمني، إزالة Outliers | `src/preprocessing.py` |
| 3 | **Feature Engineering** | Rolling stats (1h/4h/24h)، معدلات التغير، Lag features، Risk Score مركب، **تحويل الهدف لأفق زمني مستقبلي (Early Warning Horizon)** | `src/feature_engineering.py` |
| 4 | **نمذجة ML** | Random Forest + XGBoost لتصنيف "هل سيحدث خطر خلال H ساعة قادمة؟" مع معالجة عدم التوازن (SMOTE) | `src/train_ml_models.py` |
| 5 | **نمذجة DL** | LSTM يعمل على نوافذ زمنية متسلسلة لالتقاط الأنماط التراكمية | `src/train_lstm.py` |
| 6 | **نظام الإنذار** | تصنيف 3 مستويات خطر (طبيعي/تحذير/حرج) + إرسال Email فوري عند الخطر الحرج | `src/alert_system.py` |
| 7 | **الداشبورد** | مراقبة لحظية، Gauge للخطر، رسوم بيانية، تفسير النموذج (Feature Importance)، سجل التنبيهات | `app.py` (Streamlit) |
| 8 | **Deployment** | تشغيل محلي أو نشر سحابي (Streamlit Cloud / Docker) | القسم 6 بالأسفل |

---

## 3. لماذا هذا التصميم؟ (نقاط تستخدمها في المناقشة)

- **Time-based Split بدل Random Split:** لأن بيانات السلاسل الزمنية تتطلب تدريب على الماضي واختبار على المستقبل، وإلا يحدث Data Leakage ويبدو النموذج أدق مما هو عليه فعليًا في الإنتاج.
- **Early Warning Horizon:** لم نجعل الهدف "هل يوجد عطل الآن؟" بل "هل سيحدث عطل خلال الـ 4 ساعات القادمة؟" — وهذا هو جوهر مشروع "الإنذار المبكر" الحقيقي.
- **SMOTE على بيانات التدريب فقط:** الأعطال نادرة (Class Imbalance)، لذلك نعالجها بدون تلويث بيانات الاختبار.
- **ML + DL معًا:** الـ ML (RF/XGBoost) سريع، قابل للتفسير (Feature Importance)، ومناسب كخط أساس (Baseline) قوي. الـ LSTM يضيف فهمًا للتسلسل الزمني الذي قد تفوته النماذج الشجرية.
- **F2-Score بدل F1** عند اختيار الـ Threshold: لأن تفويت حريق حقيقي (False Negative) أخطر بكثير من إنذار كاذب (False Positive)، فنعطي وزنًا أكبر للـ Recall.
- **3 مستويات خطر** (Normal/Warning/Critical) بدل تصنيف ثنائي فقط: يعطي فريق الصيانة وقتًا للاستجابة المتدرجة بدل مفاجأتهم بإنذار واحد فقط.

---

## 4. هيكل المشروع

```
fire_prediction_system/
├── app.py                      # الداشبورد الرئيسي (Streamlit)
├── run_pipeline.py             # تشغيل كل شيء بأمر واحد
├── requirements.txt
├── .env.example                # قالب إعدادات البريد
├── src/
│   ├── generate_synthetic_data.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_ml_models.py
│   ├── train_lstm.py
│   └── alert_system.py
├── data/                       # يتولد تلقائيًا
└── models/                     # النماذج المدربة (.pkl / .keras)
```

---

## 5. طريقة التشغيل (محليًا)

### أ) التثبيت
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### ب) تشغيل الـ Pipeline كاملاً (بيانات + تدريب)
```bash
# باستخدام بيانات صناعية واقعية (لو لا تملك بيانات حقيقية بعد)
python run_pipeline.py

# لو عندك بيانات حقيقية بنفس الأعمدة المذكورة:
python run_pipeline.py --raw-data data/your_real_data.csv

# لتضمين تدريب LSTM أيضًا (أبطأ، يحتاج TensorFlow):
python run_pipeline.py --with-lstm
```

### ج) إعداد الإيميل (اختياري لكن مطلوب للعرض الحي)
```bash
cp .env.example .env
# افتح .env واملأ:
#  - SENDER_EMAIL: إيميل Gmail مخصص للنظام
#  - SENDER_APP_PASSWORD: App Password (وليس كلمة مرور Gmail العادية)
#    يُنشأ من: Google Account → Security → 2-Step Verification → App Passwords
#  - RECEIVER_EMAILS: إيميلات فريق الصيانة (مفصولة بفاصلة)
```

### د) تشغيل الداشبورد
```bash
streamlit run app.py
```
افتح المتصفح على `http://localhost:8501`

---

## 6. النشر (Deployment)

### الخيار 1: Streamlit Community Cloud (الأسرع للعرض/التقديم)
1. ارفع المشروع على GitHub (تأكد أن `.env` **غير** مرفوع — موجود في `.gitignore`).
2. اذهب إلى share.streamlit.io وسجّل دخول بحساب GitHub.
3. اختر الـ Repo → حدد `app.py` كملف رئيسي → Deploy.
4. أضف أسرار البريد من `Settings → Secrets` بنفس صيغة `.env`.

### الخيار 2: Docker (لبيئة شركة/سيرفر خاص)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t fire-warning-system .
docker run -p 8501:8501 --env-file .env fire-warning-system
```

### الخيار 3: ربط ببيانات حساسات حقيقية (IoT) في الإنتاج
- استبدل قراءة CSV بـ Consumer من MQTT / Kafka يستقبل قراءات الحساسات لحظيًا.
- شغّل `model.predict_proba()` على كل دفعة قراءات جديدة كل X دقائق (Cron Job / Scheduler).
- استخدم نفس دالة `send_email_alert()` — يمكن لاحقًا إضافة SMS أو Webhook لـ Slack/Teams.

---

## 7. تقييم النموذج والمقاييس المستخدمة

- **Precision / Recall / F1** لكل فئة (Normal vs Risk)
- **ROC-AUC** لقياس القدرة التمييزية الكلية
- **Confusion Matrix** لفهم الإنذارات الكاذبة مقابل الحوادث المُفوّتة
- **Feature Importance** لتفسير قرار النموذج (شفافية = ثقة أكبر من فريق الهندسة)

> ملاحظة: النتائج المعروضة في هذا الـ README الآن مبنية على بيانات صناعية تجريبية.
> عند ربط بيانات حقيقية من سنترالات فعلية، يجب إعادة التدريب وضبط الـ Thresholds
> بالتعاون مع فريق الصيانة الهندسي بناءً على تكلفة False Positive مقابل False Negative الفعلية.

---

## 8. أفكار للتوسع المستقبلي (Future Work) — تُفيدك في سؤال "خطط التطوير"

- إضافة Computer Vision لتحليل كاميرات المراقبة الحرارية (Thermal Cameras) كمصدر بيانات إضافي.
- نموذج Anomaly Detection غير مُشرف (Isolation Forest / Autoencoder) لاكتشاف أنماط فشل لم تُشاهد من قبل.
- تطبيق موبايل لفريق الصيانة بدل البريد فقط (Push Notifications).
- ربط النظام بنظام Ticketing لفتح تذكرة صيانة تلقائيًا عند إنذار حرج.
- Model Monitoring لرصد Data/Concept Drift مع الزمن وإعادة التدريب الدوري.
