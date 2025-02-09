from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import subprocess

app = Flask(__name__)
# 允许所有来源访问所有路由
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'figures'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------
# 1. 新增用于提供静态文件访问的路由
# 新的静态文件路由：用于访问 outputs/<unique_id>/<image_name>/<filename>
@app.route("/outputs/<unique_id>/<image_name>/<filename>")
def serve_output_file(unique_id, image_name, filename):
    return send_from_directory(
        directory=os.path.join(OUTPUT_FOLDER, unique_id, image_name),
        path=filename
    )
# ---------------------------------------


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

    # 生成 unique_id, 用于隔离不同用户/不同次上传的数据
    unique_id = str(uuid.uuid4())
    # 对上传的文件名进行安全处理，并提取 image_name（不带扩展名）
    orig_filename = secure_filename(file.filename)
    image_name, _ = os.path.splitext(orig_filename)

    # 构造上传目录和输出目录，输出目录为 outputs/<unique_id>/<image_name>/
    upload_subdir = os.path.join(UPLOAD_FOLDER, unique_id)
    output_subdir = os.path.join(OUTPUT_FOLDER, unique_id, image_name)
    os.makedirs(upload_subdir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)

    # 保存上传的图片文件
    input_path = os.path.join(upload_subdir, orig_filename)
    file.save(input_path)

    # 调用你的 main.py, 例如: python main.py figures/<unique_id>/<filename>
    try:
        subprocess.check_call(["python3", "main.py", input_path])
    except subprocess.CalledProcessError as e:
        print("Error executing main.py:", e)
        return jsonify({"success": False, "error": "Processing failed: " + str(e)})

    # 现在 main.py 应该会在 output_subdir 中输出结果
    # 比如 XXX-score-new.png / -0.png / -1.png / .mp3 等
    png_files = []
    mp3_files = []

    if os.path.exists(output_subdir):
        for f in sorted(os.listdir(output_subdir)):
            print(f)
            if f.lower().endswith(".png"):
                png_files.append(f)
            elif f.lower().endswith(".mp3"):
                mp3_files.append(f)
    groups = {}
    for filename in png_files:
        prefix = filename.rsplit('-', 1)[0]
        groups.setdefault(prefix, []).append(filename)

    grouped_list = list(groups.values())
    print(png_files)
    print(mp3_files)

    # ---------------------------------------
    # 2. 把本地文件名, 转换为 "可访问" 的 URL
    #    例如: https://your-backend.up.railway.app/outputs/<unique_id>/<filename>
    # 构造可访问的 URL。使用 request.host_url（如 "https://blossound-production.up.railway.app/"）
    # 结合新的静态文件路由： /outputs/<unique_id>/<image_name>/<filename>
    base_url = request.host_url
    # png_file_urls = [f"{base_url}outputs/{unique_id}/{image_name}/{fname}" for fname in png_files]
    png_file_urls = [
        [f"{base_url}outputs/{unique_id}/{image_name}/{fname}" for fname in group]
        for group in grouped_list
    ]
    mp3_file_urls = [f"{base_url}outputs/{unique_id}/{image_name}/{fname}" for fname in mp3_files]

    return jsonify({
        "success": True,
        "pngFiles": png_file_urls,
        "mp3Files": mp3_file_urls
    })


if __name__ == "__main__":
    # 你可以使用 5000 或其他端口
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)