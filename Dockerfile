# 使用官方 Python 3.9 slim 作为基础镜像
FROM python:3.9-slim

# 更新 apt-get 并安装必要的系统依赖：libgl1-mesa-glx（解决 OpenCV 的 libGL 问题）
# 以及 libglib2.0-0（提供 libgthread-2.0.so.0）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    lilypond \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 复制到工作目录，并安装 Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 复制项目所有文件到容器内
COPY . /app

# 暴露端口（此处仅作说明，实际端口由 Railway 环境变量控制）
EXPOSE 8080

# 使用环境变量 PORT 启动应用
CMD ["python3", "app.py"]