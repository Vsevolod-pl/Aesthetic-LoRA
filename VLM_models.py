import torch.nn as nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class LoRA_Qwen(nn.Module):
    def __init__(self, MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct", lora_config_params=None, device='cpu'):
        super().__init__()
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, device_map=device)
        if lora_config_params is None:
            lora_config_params = {
                'r':16,
                'lora_alpha': 16,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.1,
                'bias': "none"
            }
        config = LoraConfig(**lora_config_params)
        self.model = get_peft_model(model, config)
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        self.token_first_id = processor.tokenizer.encode("1", add_special_tokens=False)[0]
        self.token_second_id = processor.tokenizer.encode("2", add_special_tokens=False)[0]

    def forward(self, x):
        logits = self.model(**x).logits
        return F.softmax(logits[:, -1, [self.token_first_id, self.token_second_id]], dim=-1)[:, 0]

models = {
    'LoRA_Qwen2_VL': LoRA_Qwen
}