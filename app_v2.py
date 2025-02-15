from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'figures'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------
# 静态文件路由，用于访问 /outputs/<unique_id>/<image_name>/<filename>
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
    """
    接收前端上传的图片文件以及相关参数，并调用 music_gen_deeplearning_advanced.py 生成音乐和可视化文件。
    前端可通过 formData 发送:
      file: 图片文件
      length: 小节数 (默认24)
      method: "dual" 或 "pattern"
      pattern_name: (默认 "alberti_4_4")
      left_program_index: (默认 32)
      right_program_index: (默认 0)
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    # 读取其他参数（如果没有，就用默认值）
    length = request.form.get("length", "24")
    method = request.form.get("method", "dual")  # "dual" or "pattern"
    pattern_name = request.form.get("pattern_name", "alberti_4_4")
    left_program_index = request.form.get("left_program_index", "32")
    right_program_index = request.form.get("right_program_index", "0")

    # 生成 unique_id, 用于隔离不同用户/不同次上传的数据
    unique_id = str(uuid.uuid4())
    # 对上传的文件名进行安全处理，并提取 image_name（不带扩展名）
    orig_filename = secure_filename(file.filename)
    image_name, ext = os.path.splitext(orig_filename)

    # 构造上传目录和输出目录
    upload_subdir = os.path.join(UPLOAD_FOLDER, unique_id)
    output_subdir = os.path.join(OUTPUT_FOLDER, unique_id, image_name)
    os.makedirs(upload_subdir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)

    # 保存上传的图片文件
    input_path = os.path.join(upload_subdir, orig_filename)
    file.save(input_path)

    # 调用你的 music_gen_deeplearning_advanced.py
    # 注意：需要在同目录或指定绝对路径，下面示例用相对路径
    try:
        subprocess.check_call([
            "python3", "deeplr_music.py",
            "--img_path", input_path,
            "--length", str(length),
            "--method", method,
            "--pattern_name", pattern_name,
            "--left_program_index", str(left_program_index),
            "--right_program_index", str(right_program_index),
            "--out_midi", os.path.join(output_subdir, f"{image_name}.mid"),
            "--out_ly", os.path.join(output_subdir, f"{image_name}.ly"),
            "--out_pdf_dir", output_subdir
        ])
    except subprocess.CalledProcessError as e:
        print("Error executing music_gen_deeplearning_advanced.py:", e)
        return jsonify({"success": False, "error": "Processing failed: " + str(e)})

    # 根据你在 music_gen_deeplearning_advanced.py 中的逻辑，
    # 输出文件会写到 out_midi, out_ly, 以及 PDF -> output_subdir
    # 如果还生成了 PNG 或 MP3 等其他文件，也可以在下方检索

    # 收集输出目录下的资源
    png_files = []
    mp3_files = []
    pdf_files = []
    if os.path.exists(output_subdir):
        for f in sorted(os.listdir(output_subdir)):
            ext_lower = f.lower()
            if ext_lower.endswith(".png"):
                png_files.append(f)
            elif ext_lower.endswith(".mp3"):
                mp3_files.append(f)
            elif ext_lower.endswith(".pdf"):
                pdf_files.append(f)

    # 分组（如果有多张 PNG 需要分组）
    groups = {}
    for filename in png_files:
        prefix = filename.rsplit('-', 1)[0]
        groups.setdefault(prefix, []).append(filename)
    grouped_list = list(groups.values())

    # 将本地文件名转换为可访问的URL
    base_url = request.host_url
    png_file_urls = [
        [f"{base_url}outputs/{unique_id}/{image_name}/{fname}" for fname in group]
        for group in grouped_list
    ]
    mp3_file_urls = [
        f"{base_url}outputs/{unique_id}/{image_name}/{fname}"
        for fname in mp3_files
    ]
    pdf_file_urls = [
        f"{base_url}outputs/{unique_id}/{image_name}/{fname}"
        for fname in pdf_files
    ]

    return jsonify({
        "success": True,
        "pngFiles": png_file_urls,
        "mp3Files": mp3_file_urls,
        "pdfFiles": pdf_file_urls
    })


if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)