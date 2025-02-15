FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    lilypond \
    fluidsynth \
    fluid-soundfont-gm \
    imagemagick \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/g' /etc/ImageMagick-6/policy.xml


WORKDIR /app_v2

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -c "import torch; from torchvision import models; models.mobilenet_v2(weights='DEFAULT')"


COPY . /app_v2

EXPOSE 8080

CMD ["python3", "app_v2.py"]