# Shorts Factory

يحوّل فيديو يوتيوب طويل (لايف / بودكاست) إلى شورتات عمودية 9:16 تتبع المتكلم، ويرسلها لتليجرام للمراجعة.

## الخط
1. **تحميل الصوت** (yt-dlp)
2. **تفريغ** مع توقيت كل كلمة (Groq Whisper)
3. **اختيار أقوى اللحظات** + هوك + عنوان + وصف + هاشتاقات (Gemini)
4. **الكاميرامان الافتراضي** (`virtual_cam.py`): يتتبع المتكلم ويقص 9:16
5. **إرسال معاينة لتليجرام**، والنسخة الكاملة الجودة في Artifacts

## الإعداد
1. أنشئ مستودع GitHub. (العام له وقت تشغيل أكبر بكثير من الخاص.)
2. ارفع الملفات بنفس هذا الهيكل:
   ```
   .github/workflows/shorts.yml
   src/config.py, download.py, transcribe.py, select_clips.py,
       virtual_cam.py, telegram_send.py, pipeline.py
   requirements.txt
   ```
3. من Settings ← Secrets and variables ← Actions أضف:
   - `GROQ_API_KEY` من console.groq.com
   - `GEMINI_API_KEY` من aistudio.google.com
   - `TELEGRAM_BOT_TOKEN` من @BotFather
   - `TELEGRAM_CHAT_ID`: راسل البوت ثم افتح `https://api.telegram.org/bot<TOKEN>/getUpdates` وخذ قيمة `chat.id`
   - `YT_COOKIES` (اختياري): محتوى ملف cookies.txt من **حساب يوتيوب جانبي**، إذا حظر يوتيوب التحميل من GitHub
4. من تبويب Actions اختر **Shorts Factory** ← Run workflow ← الصق الرابط واللغة (`ar` / `en` / `auto`).

## ضبط التتبع
في أعلى `virtual_cam.py`:
- `MIN_SCORE`: ارفعه إذا اعتبر المستمع متكلماً، واخفضه إذا فوّت المتكلم.
- `SWITCH_HOLD` و`MIN_SHOT`: يتحكمان في سرعة الانتقال بين المتكلمين.
- شغّله محلياً مع `--debug` لترى المربعات (الأخضر = المتكلم الحالي):
  `python src/virtual_cam.py in.mp4 out.mp4 --start 60 --end 110 --debug`

## ملاحظات
- أسماء النماذج (`WHISPER_MODEL`, `GEMINI_MODEL` في config.py) تتغير مع الوقت: راجع وثائق Groq وGoogle وعدّلها عبر متغيرات البيئة إن لزم.
- الطبقات المجانية لها حدود (Groq تحدّ ساعات الصوت في الساعة)، فاللايف الطويل جداً قد ينتظر بين الأجزاء تلقائياً.
- أول تشغيل ينزّل نموذج الوجوه (`models/face_landmarker.task`) تلقائياً.
- استخدم محتوى تملكه أو لديك إذن به.
