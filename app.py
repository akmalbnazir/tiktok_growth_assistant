from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from utils.video_analyzer import analyze_video
from utils.caption_generator import suggest_caption
import time

load_dotenv()

app = Flask(__name__)

def safe_remove(filepath, retries=5, delay=0.5):
    for _ in range(retries):
        try:
            os.remove(filepath)
            break
        except PermissionError:
            time.sleep(delay)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        video_path = os.path.join('static', video.filename)
        video.save(video_path)

        try:
            analysis = analyze_video(video_path)
            new_caption = suggest_caption(analysis)
        finally:
            # Always remove the uploaded file after processing
            if os.path.exists(video_path):
                safe_remove(video_path)

        return render_template('result.html', analysis=analysis, caption=new_caption)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
