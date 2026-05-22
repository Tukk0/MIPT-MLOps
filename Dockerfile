FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    telebot opencv-python-headless \
    requests numpy

WORKDIR /app
COPY main.py .
ENV TELEGRAM_BOT_TOKEN=""
ENV TRITON_ENDPOINT="http://triton:8000/v2/models/digit_recognition/infer"

EXPOSE 5000

CMD ["python", "main.py"]
