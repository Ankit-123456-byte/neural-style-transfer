FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY model.py dataset.py utils/ utils/
COPY app.py .

# Optional: mount weights at runtime via -v instead of baking them in
# COPY vgg_normalised.pth .

EXPOSE 7860

ENV VGG_PATH=vgg_normalised.pth
ENV DECODER_PATH=experiment/experiment1/decoder_final.pth
ENV PORT=7860

CMD ["python", "app.py"]
