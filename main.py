import os
import sys
import requests
from analyzer import get_youtube_transcript, analyze_podcast_with_gemini
from processor import download_and_cut_clip

# جلب المتغيرات السرية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
YOUTUBE_URL = os.environ.get("YOUTUBE_URL")

def send_telegram_video(video_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    with open(video_path, 'rb') as video_file:
        files = {'video': video_file}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'supports_streaming': True}
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("تم إرسال الفيديو بنجاح إلى تيليجرام!")
        else:
            print(f"فشل إرسال الفيديو: {response.text}")

if __name__ == "__main__":
    if not YOUTUBE_URL:
        print("خطأ: لم يتم إدخال رابط يوتيوب.")
        sys.exit(1)
        
    print(f"جاري جلب نص الفيديو: {YOUTUBE_URL}")
    text, _ = get_youtube_transcript(YOUTUBE_URL)
    
    if not text:
        print("عذراً، لم نتمكن من جلب نص الفيديو.")
        sys.exit(1)
        
    print("جاري تحليل النص واستخراج أفضل مقطع عبر Gemini...")
    clips = analyze_podcast_with_gemini(text)
    
    if not clips:
        print("لم يتم العثور على مقاطع مناسبة.")
        sys.exit(1)
        
    # نأخذ أول مقطع مقترح كبداية لتوفير الوقت
    best_clip = clips[0]
    start = best_clip.get("start_time")
    end = best_clip.get("end_time")
    title = best_clip.get("title", "شورت بودكاست جديد")
    
    print(, f"المقطع المختار: من {start} إلى {end} - العنوان: {title}")
    
    output_file = download_and_cut_clip(YOUTUBE_URL, start, end)
    
    if output_file and os.path.exists(output_file):
        caption = f"🔥 {title}\n\nتم توليد هذا الشورت تلقائياً عبر نظامك الذكي 🚀"
        send_telegram_video(output_file, caption)
        
        # تنظيف الملف المحلي
        os.remove(output_file)
    else:
        print("فشل معالجة الفيديو.")
