# Aesthetic-LoRA

Demo for **"Explainable Pairwise Aesthetic Preference with a LoRA-Adaptive Vision–Language Model"** (IJCAI).

Upload two images and the app uses a fine-tuned Qwen2-VL model (with optional LoRA weights) to decide which image is aesthetically superior and explain why.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (the default config targets `cuda:0`)

---

## Installation

```bash
git clone https://github.com/Vsevolod-pl/Aesthetic-LoRA.git
cd Aesthetic-LoRA

pip install -r requirenments.txt
```

Upload weights from https://disk.yandex.ru/d/OM2JSNbxEAc8VA ad copy them to weights/

---

## Model Weights

The LoRA model variant requires a pretrained weights file. Place it at:

```
./weights/LoRA_Qwen2_VL.pth
```

The base Qwen2-VL model (`Qwen/Qwen2-VL-2B-Instruct`) is downloaded automatically from Hugging Face on first run.

---

## Configuration

All model and prompt settings live in `config.yaml`. Key fields:

| Field | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen2-VL-2B-Instruct` | HuggingFace model to load |
| `device` | `cuda:0` | Device for inference |
| `models.lora_qwen.weights_path` | `./weights/LoRA_Qwen2_VL.pth` | Path to LoRA checkpoint |

To run on CPU (slower), change `device` to `cpu`.

---

## Running the App

```bash
python app.py
```

The server starts on **http://0.0.0.0:8032**. Open **http://localhost:8032** in your browser.

Alternatively, launch via uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8032
```

---

## Usage

1. Open **http://localhost:8032** in your browser.
2. Upload two images (e.g. two edits of the same photo).
3. Select a model — **LoRA Fine-tuned Qwen2-VL** or **Base Qwen2-VL**.
4. Optionally customise the describe prompt.
5. Click **Compare** — the app returns which image wins and a detailed per-region explanation.
