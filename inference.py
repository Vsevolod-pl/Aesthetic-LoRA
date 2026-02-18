import torch
import yaml
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

_processor = None
_models = {}
_config = None


def init_models(config_path: str = "config.yaml"):
    global _processor, _models, _config

    with open(config_path) as f:
        _config = yaml.safe_load(f)

    model_id = _config["model_id"]
    device = _config["device"]

    _processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    for key, model_cfg in _config["models"].items():
        if model_cfg["type"] == "lora":
            from VLM_models import LoRA_Qwen

            classifier = LoRA_Qwen(device=device)
            stdct = torch.load(
                model_cfg["weights_path"],
                weights_only=False,
                map_location=device,
            )
            classifier.load_state_dict(stdct["model"])
            _models[key] = classifier.model

        elif model_cfg["type"] == "base":
            _models[key] = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.float32, device_map=device
            )

        else:
            raise ValueError(f"Unknown model type: {model_cfg['type']}")

    print(f"Loaded {len(_models)} models: {list(_models.keys())}")


def get_model_choices() -> list[dict]:
    return [
        {"key": key, "label": cfg["label"]}
        for key, cfg in _config["models"].items()
    ]


def get_default_describe_prompt() -> str:
    return _config["prompts"]["describe"]


def _build_conversation(
    prompt_template: str,
    img1_path: str,
    img2_path: str,
    winner: str | None = None,
) -> list:
    text = prompt_template
    if winner is not None:
        text = text.replace("{winner}", winner)

    parts = text.split("{image1}")
    before_img1 = parts[0]
    after_img1 = parts[1]

    parts2 = after_img1.split("{image2}")
    between = parts2[0]
    after_img2 = parts2[1]

    content = []
    if before_img1.strip():
        content.append({"type": "text", "text": before_img1.strip()})
    content.append({"type": "image", "url": img1_path})
    if between.strip():
        content.append({"type": "text", "text": between.strip()})
    content.append({"type": "image", "url": img2_path})
    if after_img2.strip():
        content.append({"type": "text", "text": after_img2.strip()})

    return [{"role": "user", "content": content}]


def _generate(model, conversation: list, max_new_tokens: int) -> str:
    inputs = _processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)
    ]
    text = _processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    return text


@torch.no_grad()
def run_inference(
    model_key: str,
    img1_path: str,
    img2_path: str,
    describe_prompt: str | None = None,
) -> dict:
    model = _models[model_key]
    vote_template = _config["prompts"]["vote"]
    desc_template = describe_prompt or _config["prompts"]["describe"]
    vote_max = _config["generation"]["vote_max_tokens"]
    desc_max = _config["generation"]["describe_max_tokens"]

    # Step 1: Vote
    conversation = _build_conversation(vote_template, img1_path, img2_path)
    vote_text = _generate(model, conversation, vote_max)

    # Extract "1" or "2" from response
    winner = vote_text
    for char in vote_text:
        if char in ("1", "2"):
            winner = char
            break

    # Step 2: Describe
    conversation = _build_conversation(
        desc_template, img1_path, img2_path, winner=winner
    )
    description = _generate(model, conversation, desc_max)

    return {
        "vote_raw": vote_text,
        "winner": winner,
        "description": description,
    }
