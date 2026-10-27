import os
import sys
import requests
from analyzer import analyze_podcast_from_url
from processor import download_and_cut_clip

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
YOUTUBE_URL = (os.environ.get("YOUTUBE_URL") or "").strip()


def send_telegram_video(video_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > 49:
        print(f"حجم الفيديو {size_mb:.1f}MB ويتجاوز حد تيليجرام.")
        return False
    with open(video_path, "rb") as video_file:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption[:1000],
                "supports_streaming": "true",
                "width": 1080,
                "height": 1920,
            },
            files={"video": video_file},
            timeout=600,
        )
    if response.status_code == 200:
        print("تم إرسال الفيديو بنجاح إلى تيليجرام!")
        return True
    print(f"فشل إرسال الفيديو: {response.text}")
    return False


if __name__ == "__main__":
    if not YOUTUBE_URL:
        print("خطأ: لم يتم إدخال رابط يوتيوب.")
        sys.exit(1)

    print(f"جاري بدء النظام لرابط يوتيوب: {YOUTUBE_URL}")
    clips = analyze_podcast_from_url(YOUTUBE_URL)
    if not clips:
        print("لم يتم العثور على مقاطع مناسبة أو حدث خطأ في التحليل.")
        sys.exit(1)

    best_clip = clips[0]
    start = best_clip["start_time"]
    end = best_clip["end_time"]
    title = best_clip["title"]
    print(f"المقطع المختار: من {start} إلى {end} - العنوان: {title}")

    output_file = download_and_cut_clip(YOUTUBE_URL, start, end)
    if output_file and os.path.exists(output_file):
        caption = f"🔥 {title}\n\nتم توليد هذا الشورت تلقائياً عبر نظامك الذكي 🚀"
        sent = send_telegram_video(output_file, caption)
        os.remove(output_file)
        if not sent:
            sys.exit(1)
    else:
        print("فشل معالجة الفيديو.")
        sys.exit(1)
