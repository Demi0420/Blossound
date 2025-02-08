import os
import uuid
import subprocess

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'figures'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return "Hello, this is the backend!"

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    unique_id = str(uuid.uuid4())
    upload_subdir = os.path.join(UPLOAD_FOLDER, unique_id)
    output_subdir = os.path.join(OUTPUT_FOLDER, unique_id)
    os.makedirs(upload_subdir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)

    filename = secure_filename(file.filename)
    input_path = os.path.join(upload_subdir, filename)
    file.save(input_path)

    # 调用 main.py
    try:
        subprocess.check_call(["python", "main.py", input_path])
    except subprocess.CalledProcessError as e:
        print(e)
        return jsonify({"success": False, "error": "Processing failed"})

    # 收集输出文件
    png_files = []
    mp3_file = None
    if os.path.exists(output_subdir):
        for f in os.listdir(output_subdir):
            fpath = os.path.join(output_subdir, f)
            if f.endswith(".png"):
                png_files.append(fpath)
            elif f.endswith(".mp3"):
                mp3_file = fpath

    png_file_urls = []
    # Railway 等云平台会自动给我们一个域名
    # 这里先直接原样返回文件名，后续再决定如何访问
    for fp in sorted(png_files):
        png_file_urls.append(fp)

    return jsonify({
        "success": True,
        "pngFiles": png_file_urls,
        "mp3File": mp3_file
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)