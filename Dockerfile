# 使用官方的 Python 3.9 slim 版本作为基础镜像
FROM python:3.9-slim

# 安装系统依赖，包括 libGL.so.1 所需的包
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 拷贝到工作目录，并安装 Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 将当前项目的所有文件复制到容器的 /app 目录中
COPY . /app

# 暴露端口（Railway 会自动设置端口环境变量，这里只是说明）
EXPOSE 5001

# 启动命令，注意这里使用 python3 来确保使用正确的解释器
CMD ["python3", "app.py"]