# Handwritten Digit Recognition with Deep Learning
**Листов Тихон Андреевич, МФТИ, MLOps, весна 2026**

## 1. Постановка задачи

Система классификации рукописных цифр (0–9) на растровых изображениях. На вход подаётся чёрно-белое изображение размером 28×28 пикселей, содержащее одну цифру, написанную от руки. На выходе — предсказанный класс (цифра 0–9) с уверенностью модели.

Проект включает полный MLOps-пайплайн: загрузка и управление данными через DVC, обучение через PyTorch Lightning, логирование через MLFlow/TensorBoard, экспорт модели в ONNX, развёртывание inference-сервера через NVIDIA Triton.

## 2. Входные и выходные данные

**Вход:** изображение в форматах PNG/JPG, grayscale, размер 28×28 пикселей. Возможна подача фото произвольного размера — предобработка масштабирует и нормализует его автоматически.

**Выход:**
- Предсказанный класс (int: 0–9).
- softmax-вероятности для каждого из 10 классов.

## 3. Метрики

Основная метрика — **accuracy**. Дополнительно:
- **Per-class recall и precision** — для анализа баланса между классами.
- **Macro-F1 score** — гармоническое среднее precision и recall по всем классам.
- **Confusion matrix** — визуализация ошибок.

Планируемый target — **accuracy ≥ 98.5%** на test set.

## 4. Разделение данных

Стратифицированное разделение 60/20/20 (train/validation/test).

## 5. Датасет

**Основной датасет:** MNIST.
- **Источник:** `https://huggingface.co/datasets/ylecun/mnist`
- **Объём:** 60 000 изображений для тренировки, 10 000 для тестирования — ~70 МБ (gzipped).
- **Формат:** grayscale PNG/IDX, 28×28 пикселей, интенсивность 0–255.
- **Дата публикации:** 1998 г.
- **Особенности:** цифры написаны посетителями Американского бюро переписи населения и студентами.

## 6. Моделирование

### Базовый бейзлайн

**Логистическая регрессия** с softmax. ~92% accuracy.

### Основная модель

**LeNet-5 модифицированная** (BatchNorm, dropout, AdamW):
```
Conv2d(1, 32, 3) → BatchNorm → ReLU → MaxPool
Conv2d(32, 64, 3) → BatchNorm → ReLU → MaxPool
Flatten → Linear(3136, 128) → ReLU → Dropout(0.5) → Linear(128, 10)
```

### Тренировочный пайплайн (PyTorch Lightning):
- **Предобработка:** torchvision.transforms — нормализация (mean=0.1307, std=0.3081), аргументация.
- **Обучение:** PyTorch Lightning — модуль, DataModule, Trainer с EarlyStopping, MLFlow/WandB.
- **Оптимизатор:** AdamW с weight decay=1e-2, LR scheduling (CosineAnnealingLR).
- **Loss:** CrossEntropyLoss.

## 7. Внедрение

### Production preparation
- **ONNX** — универсальный инференсный формат.
- **TensorRT** — оптимизированное runtime-исполнение на NVIDIA GPU.

### Inference server
**NVIDIA Triton Inference Server** — ONNX модель загружается в Triton.

### Telegram-бот
Telegram-бот получает изображение, передаёт его в Triton inference-сервер и возвращает предсказание.

## 8. Setup

```bash
git clone <repo-url> && cd digit-recognition

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# или
# poetry install
# poetry shell

pre-commit install
pre-commit run -a
```

## 9. Training

```bash
source .venv/bin/activate
python train.py
```

Перевычисление с кастомными параметрами:
```bash
python train.py training.max_epochs=50 training.optimizer.lr=0.0005
```

Логи пишутся в:
- **MLFlow** на `http://127.0.0.1:8080` (метрики, параметры, git commit).
- **TensorBoard** в `logs/run_<timestamp>/`.
- **Checkpoints модели** сохраняются в `checkpoints/`.

## 10. Inference

```bash
python infer.py predict --model-path checkpoints/best.pt --image test.png
python infer.py onnx --model checkpoints/best.pt --onnx model.onnx
tritonserver --model-repo models/digit_recognition --http-port 8000 --grpc-port 8001
```

## 11. Конфигурация

Все гиперпараметры лежат в `configs/` — иерархические YAML-файлы Hydra:

```yaml
├── configs/
│   ├── data.yaml          # batch_size, workers, augmentation params
│   ├── model.yaml         # архитектура, слои, dropout, batch_norm
│   ├── training.yaml      # optimizer, scheduler, early_stopping, devices
│   ├── logging.yaml       # mlflow_uri, wandb_project, mlflow_experiment_name
│   ├── inference.yaml     # model_path, image_size, device, onnx/trt paths
│   ├── train.yaml         # defaults group для тренировки
│   └── infer.yaml         # defaults group для инференса
```

Переопределение через CLI:
```bash
python train.py training.max_epochs=50 training.optimizer.lr=0.0005
```

## 12. Production preparation

```bash
python infer.py onnx --model checkpoints/best.pt --onnx model.onnx

# TensorRT (требуется NVIDIA GPU + tensorrt installed)
./deploy_tensorrt.sh model.onnx model.trt 1
```

## 13. Telegram-бот

```bash
cp .env.local .env.local.bak
# В .env.local.bak: export TELEGRAM_BOT_TOKEN=<your_token>
python main.py --triton-endpoint http://localhost:8000/v2/models/digit_recognition/infer
```

## 14. Docker

```bash
docker compose up --build

# Только Triton
docker run -d --name triton \
  -v $(pwd)/models:/models \
  -p 8000:8000 -p 8001:8001 \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  --model-repo /models --http-port 8000 --grpc-port 8001
```

## 15. Структура проекта

```
digit-recognition/
├── configs/
│   ├── data.yaml          # параметры загрузки и структуры dataloader
│   ├── model.yaml         # архитектура модели
│   ├── training.yaml      # гиперпараметры обучения (optimizer, lr, batch_size)
│   ├── logging.yaml       # MLFlow/WandB конфигурация
│   ├── inference.yaml     # параметры инференса и конвертации
│   ├── train.yaml         # defaults group для тренировки
│   └── infer.yaml         # defaults group для инференса
├── src/
│   ├── __init__.py
│   ├── data_module.py     # LightningDataModule для загрузки данных
│   ├── model.py           # PyTorch Lightning модуль, архитектура CNN
│   ├── preprocessing.py   # transforms и нормализация
│   ├── inference.py       # инференс (load, predict, ONNX)
│   └── download_data.py   # download MNIST via DVC Python API
├── models/
│   └── digit_recognition/ # Triton Model Repository
│       ├── 1/
│       │   └── model.onnx
│       └── config.pbtxt
├── train.py               # точка входа в тренировку (Hydra + Lightning)
├── infer.py               # точка входа в инференс (public API, CLI)
├── convert.py             # конвертация модели в ONNX
├── deploy_tensorrt.sh     # shell-скрипт для TensorRT конвертации
├── main.py                # Telegram-бот (входная точка)
├── pyproject.toml         # зависимости (Poetry)
├── poetry.lock
├── .gitignore
├── .pre-commit-config.yaml
├── .env.local             # переменные окружения (Telegram Token)
├── docker-compose.yaml    # оркестрация Triton + Telegram-бот
├── Dockerfile             # контейнеризация Telegram-бота
├── dvc.yaml               # DVC pipeline
├── .dvc/
│   ├── config             # хранилища данных и моделей
│   └── ...
└── data/                  # данные под управлением DVC (не в git)
```
