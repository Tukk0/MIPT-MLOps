"""Telegram bot for MNIST digit recognition via Triton Inference Server."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TRITON_ENDPOINT = os.getenv(
    "TRITON_ENDPOINT",
    "http://triton:8000/v2/models/digit_recognition/infer",
)


if TELEGRAM_BOT_TOKEN is None:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var first.")


def _preprocess(photo_path: str) -> np.ndarray:
    img = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image from {photo_path}")
    img = cv2.resize(img, (28, 28))
    return img.astype(np.float32) / 255.0


def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    raw = bot.download_file(file_info.file_path)
    tmp_path = Path(tempfile.mkdtemp()) / "digit.png"
    tmp_path.write_bytes(raw)
    img = _preprocess(str(tmp_path))
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 28, 28],
                "datatype": "FP32",
                "data": img.flatten().tolist(),
            }
        ]
    }
    response = requests.post(TRITON_ENDPOINT, json=payload, timeout=10)
    data = response.json()
    probs = np.array(data["outputs"][0]["data"]).reshape(10)
    label = int(np.argmax(probs))
    confidence = float(probs[label])
    bot.send_message(
        message.chat.id,
        f"Digit: {label}  (confidence: {confidence:.0%})",
        parse_mode="HTML",
    )


def handle_text(message):
    text = message.text.strip().lower()
    if text == "/start":
        bot.send_message(
            message.chat.id,
            "Start the bot and send a photo of a handwritten digit!",
            parse_mode="HTML",
        )
    elif text == "/help":
        bot.send_message(
            message.chat.id,
            "/help — show this message",
            parse_mode="HTML",
        )
    else:
        bot.send_message(message.chat.id, "Send a photo of a handwritten digit.")


# Import inside function to avoid crashing module-level on missing telebot
def _get_bot():
    from telebot import TeleBot

    return TeleBot(TELEGRAM_BOT_TOKEN)


bot = _get_bot()


def _register_handlers():
    bot.message_handler(content_types=["photo"])(handle_photo)
    bot.message_handler(content_types=["text"])(handle_text)


_register_handlers()


if __name__ == "__main__":
    print("Starting Telegram bot...")
    bot.infinity_polling()
