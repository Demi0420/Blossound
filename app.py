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
# /outputs/<unique_id>/<filename> => 读取 outputs/unique_id/filename 并返回
@app.route("/outputs/<unique_id>/<filename>")
def serve_output_file(unique_id, filename):
    """
    这个路由专门用来返回某个 unique_id 子目录下的文件
    """
    return send_from_directory(
        directory=os.path.join(OUTPUT_FOLDER, unique_id),
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
    upload_subdir = os.path.join(UPLOAD_FOLDER, unique_id)
    output_subdir = os.path.join(OUTPUT_FOLDER, unique_id)
    os.makedirs(upload_subdir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)

    # 保存上传的文件
    filename = secure_filename(file.filename)
    input_path = os.path.join(upload_subdir, filename)
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
    mp3_file = None

    if os.path.exists(output_subdir):
        for f in sorted(os.listdir(output_subdir)):
            # 构造文件的完整路径
            fpath = os.path.join(output_subdir, f)
            # 判断后缀
            if f.endswith(".png"):
                png_files.append(f)
            elif f.endswith(".mp3"):
                mp3_file = f

    # ---------------------------------------
    # 2. 把本地文件名, 转换为 "可访问" 的 URL
    #    例如: https://your-backend.up.railway.app/outputs/<unique_id>/<filename>
    base_url = request.host_url  # eg. "https://xxx.up.railway.app/"

    png_file_urls = []
    for fname in png_files:
        # fname 是文件名, 构造成 /outputs/<unique_id>/<fname>
        file_url = f"{base_url}outputs/{unique_id}/{fname}"
        png_file_urls.append(file_url)

    mp3_file_url = None
    if mp3_file:
        mp3_file_url = f"{base_url}outputs/{unique_id}/{mp3_file}"
    # ---------------------------------------

    return jsonify({
        "success": True,
        "pngFiles": png_file_urls,
        "mp3File": mp3_file_url
    })


if __name__ == "__main__":
    # 你可以使用 5000 或其他端口
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)