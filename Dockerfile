# PyTorch image for training, inference and conversion
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    pytorch-lightning \
    hydra-core fire \
    onnx onnxruntime onnxsim \
    scikit-learn \
    opencv-python-headless \
    ruff \
    pytest \
    torchmetrics torchinfo \
    tensorboard && \
    rm -rf /root/.cache/pip

WORKDIR /app
COPY . .

# Install telebot only if needed for bot mode
RUN pip install --no-cache-dir telebot pyTelegramBotAPI requests

# Make run.sh executable
RUN chmod +x run.sh entrypoint.sh

# Default command is the one-command pipeline
CMD ["./run.sh"]

# Dockerfile allows overriding CMD for other modes:
#   docker run --rm digit-recognition ./run.sh train 10     # train 10 epochs
#   docker run --rm digit-recognition ./run.sh test         # run pytest
#   docker run --rm digit-recognition ./run.sh lint         # ruff check
#   docker run --rm digit-recognition ./run.sh onnx         # export to ONNX + verify
