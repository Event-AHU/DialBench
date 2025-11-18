# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import re
from typing import List, Dict, Any, Tuple, Union

import io
import requests
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Gemma3ForConditionalGeneration, PaliGemmaForConditionalGeneration,Qwen2VLForConditionalGeneration
from transformers.image_utils import load_image as hf_load_image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,Glm4vForConditionalGeneration,Gemma3nForConditionalGeneration
# ==== Qwen2.5-VL 依赖 ====
# 重要：给 ModelScope 的 AutoProcessor 起别名，避免与 HF 的 AutoProcessor 混淆
from modelscope import AutoModel as MS_AutoModel, AutoProcessor as MS_AutoProcessor
from modelscope import Qwen2_5_VLForConditionalGeneration
# 新增：为 modelscope 版 Llava 起别名（用于 pixtral-12b）
#from modelscope import LlavaForConditionalGeneration as MS_LlavaForConditionalGeneration

# HF 侧：为避免与 modelscope 冲突，起别名
from transformers import LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration, MllamaForConditionalGeneration
# 尝试导入，如果不存在则忽略（仅Qwen需要）
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


# ==== InternVL 依赖 ====
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor as HF_AutoProcessor, AutoModelForImageTextToText

# ==== 可选：PEFT ====
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# =========================
# 公共工具 & 指标方法
# =========================
def disable_torch_init():
    import torch as _torch
    setattr(_torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(_torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def extract_last_number(text: str) -> float:
    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if matches:
        return float(matches[-1])
    return 0.0


def calculate_metrics(
    absolute_errors: List[float],
    ranges: List[float],
    valid_theta_samples: List[float],
    correct_count_theta: int,
    nonzero_gt_count_theta: int,
    accuracy_epsilon_correct_count: int,
    processed_samples_count: int,
) -> Dict[str, float]:
    metrics: Dict[str, Any] = {}
    if len(valid_theta_samples) > 0:
        theta_ratios = np.array(valid_theta_samples)
        theta_ratios = np.clip(theta_ratios, None, 100)
        metrics["theta"] = float(np.mean(theta_ratios))
    else:
        metrics["theta"] = None

    if len(absolute_errors) > 0 and len(ranges) > 0:
        gamma_ratios = np.array(absolute_errors) / np.array(ranges)
        gamma_ratios = np.clip(gamma_ratios, None, 10)
        metrics["gamma"] = float(np.mean(gamma_ratios))
    else:
        metrics["gamma"] = None

    metrics["accuracy_theta"] = (
        correct_count_theta / nonzero_gt_count_theta if nonzero_gt_count_theta > 0 else None
    )
    metrics["accuracy_epsilon"] = (
        accuracy_epsilon_correct_count / processed_samples_count if processed_samples_count > 0 else None
    )
    return metrics


def print_metrics(metrics: Dict[str, float], batch: bool = False):
    prefix = "批次 " if batch else "最终 "
    print(f"\n--- {prefix}评估结果 ---")
    if metrics["theta"] is not None:
        print(f" Theta (Θ̂): {metrics['theta']:.4f}")
    if metrics["gamma"] is not None:
        print(f" Gamma (Γ̂): {metrics['gamma']:.4f}")
    if metrics["accuracy_theta"] is not None:
        print(f" Accuracy (Theta < 0.05): {metrics['accuracy_theta']:.4f}")
    if metrics["accuracy_epsilon"] is not None:
        print(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}")
    print("------------------------\n")


def create_results_file(model_name: str, save_dir: str) -> str:
    safe_model_name = model_name.replace('/', '_')
    results_file_path = os.path.join(save_dir, f"evaluation_results_{safe_model_name}.txt")
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w', encoding="utf-8") as f:
            f.write(f"Evaluation Results for {model_name}\n")
            f.write("=" * 50 + "\n")
    return results_file_path


def save_metrics_to_file(metrics: Dict[str, float], file_path: str):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- Evaluation Metrics ---\n")
        if metrics["theta"] is not None:
            f.write(f" Theta (Θ̂): {metrics['theta']:.4f}\n")
        if metrics["gamma"] is not None:
            f.write(f" Gamma (Γ̂): {metrics['gamma']:.4f}\n")
        if metrics["accuracy_theta"] is not None:
            f.write(f" Accuracy (Theta < 0.05): {metrics['accuracy_theta']:.4f}\n")
        if metrics["accuracy_epsilon"] is not None:
            f.write(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}\n")
        f.write("-" * 50 + "\n")


def load_data(file_path: str, image_prefix: str):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    image_paths = [os.path.join(image_prefix, item['image']) for item in data]
    questions = [item['query'] for item in data]
    ground_truths = [extract_last_number(str(item['reading'])) for item in data]
    ranges = [float(item['range']) for item in data]
    return image_paths, questions, ground_truths, ranges


# =========================
# 数据预加载（与模型无关）
# =========================
def preload_images(image_paths: List[str], questions: List[str]) -> List[Dict[str, Any]]:
    prepared = []
    for i in tqdm(range(len(image_paths)), desc="预加载图像"):
        p = image_paths[i]
        q = questions[i]
        try:
            img = Image.open(p).convert('RGB')
            prepared.append({
                "original_image": img,
                "original_path": p,
                "question": q,
                "original_index": i
            })
        except Exception as e:
            print(f"加载图像失败 {p}: {e}")
    return prepared


# =========================
# 适配器接口 & 实现
# =========================
class BaseVLMAdapter:
    def __init__(self, model_id: str, model_path: str, lora_path: str, device: str, max_new_tokens: int = 128):
        self.model_id = model_id
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        raise NotImplementedError

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        raise NotImplementedError
    
    
# ---- LLaVA-NEXT 适配器 ----
class LLaVANextAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[LLaVA-NEXT] 从 {load_from} 加载模型与分词器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        self.processor = LlavaNextProcessor.from_pretrained(load_from)
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            load_from, 
            torch_dtype=self.fp_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        ).eval().to(device_id)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs = []
        # 当前处理器实现可能不直接支持批处理图像+文本，采用循环处理确保正确性
        for item in batch_items:
            image = item["original_image"]
            question = item["question"]

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            # 解码并去除输入部分
            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self.processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
            outputs.append(generated_text.strip())
        
        return outputs
    
    

# ---- GLM4.1V 适配器 ----
class GLM41VAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[GLM4.1V] 从 {load_from} 加载模型与分词器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        self.processor = AutoProcessor.from_pretrained(load_from)
        
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=load_from,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).to(device_id)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs = []
        # 当前处理器实现可能不直接支持批处理图像+文本，采用循环处理确保正确性
        for item in batch_items:
            image = item["original_image"]
            question = item["question"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ],
                }
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            # 解码并去除输入部分
            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self.processor.decode(output_ids[0, prompt_len:], skip_special_tokens=False)
            outputs.append(generated_text.strip())
        
        return outputs
# ---- KeyeVL 适配器 ----
class KeyeVLAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.float16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Keye-VL] Loading model and processor from {load_from}...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        
        # Loading model and processor
        self.processor = AutoProcessor.from_pretrained(load_from)
        self.model = AutoModel.from_pretrained(
            load_from, 
            torch_dtype=self.fp_dtype,
            device_map=device_map
        ).to(device_id)
        
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs: List[str] = []

        for item in batch_items:
            raw_image = item["original_image"]
            question = item["question"]

            # Prepare messages as per the input format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},  # The image
                        {"type": "text", "text": question}  # The associated text
                    ]
                }
            ]

            # Generate the input text for the processor
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            # Process input (image and text)
            inputs = self.processor(
                text=input_text,
                images=raw_image,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self.processor.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

            outputs.append(generated_text.strip())

        return outputs
    
#---- MiniCPM 适配器 ----#
class MiniCPMA4Vdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.float16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[MiniCPM] Loading model and tokenizer from {load_from}...")
        device_id = _parse_device(self.device)
        
        # Load the model with SDPA attention
        self.model = AutoModel.from_pretrained(
            load_from,
            attn_implementation="sdpa",
            torch_dtype=self.fp_dtype,
            trust_remote_code=True
        ).eval().to(device_id)
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs: List[str] = []

        for item in batch_items:
            raw_image = item["original_image"]
            question = item["question"]

            # Prepare messages for the first round
            msgs = [{'role': 'user', 'content': [raw_image, question]}]

            # Perform first round of chat (question-answer)
            answer = self.model.chat(
                msgs=msgs,
                image=raw_image,
                tokenizer=self.tokenizer
            )
            outputs.append(answer.strip())

        return outputs    
    
# ---- PaliGemma2-10B 适配器 ----
class PaliGemmaMSAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Gemma3] 从 {load_from} 加载模型与分词器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            load_from, 
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval().to(device_id)
        self.processor = AutoProcessor.from_pretrained(load_from)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "请先调用 load() 方法加载模型"
        outputs: List[str] = []

        for it in batch_items:
            raw_image = it["original_image"]
            question = it["question"]

            # 消息结构：text + image
            prompt = question
            model_inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(torch.bfloat16).to(self.model.device)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                outputs.append(decoded.strip())
        return outputs
    
# ---- Gemma3 -12B 适配器 ----
import torch
from typing import List, Dict, Any
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

class Gemma3Adapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Gemma3] 从 {load_from} 加载模型与分词器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            load_from, 
            device_map=device_map,
            torch_dtype=self.fp_dtype
        ).to(device_id)
        self.processor = AutoProcessor.from_pretrained(load_from)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "请先调用 load() 方法加载模型"
        outputs: List[str] = []

        for it in batch_items:
            raw_image = it["original_image"]
            question = it["question"]

            # 消息结构：text + image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": raw_image},  # 图片 URL
                        {"type": "text", "text": question}   # 问题文本
                    ]
                }
            ]

            # 处理消息并将其转化为输入
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            # 生成回答
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]

            # 解码生成的文本
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            outputs.append(decoded.strip())

        return outputs



# ---- MINICPMV2-6 适配器 ----
class MiniCPMAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[MiniCPM] 从 {load_from} 加载模型与分词器...")
        device_id = _parse_device(self.device)

        self.model = AutoModel.from_pretrained(
            load_from,
            trust_remote_code=True,
            attn_implementation="sdpa",   # 也可以换成 "flash_attention_2"
            torch_dtype=self.fp_dtype
        ).eval().to(device_id)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            trust_remote_code=True
        )

        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "请先调用 load() 方法加载模型"
        outputs: List[str] = []

        for it in batch_items:
            raw_image: Image.Image = it["original_image"]
            question: str = it["question"]

            # MiniCPM 的消息结构：msgs 里直接放 [image, text]
            msgs = [{"role": "user", "content": [raw_image, question]}]

            # 推理
            res = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
            )
            outputs.append(res.strip())

        return outputs

#---- Llama-3.2-Vision (ModelScope) 适配器 ----
class Llama32VisionMSAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, lora_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, lora_path, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Llama-3.2-Vision] 从 {load_from} 加载模型与处理器 (modelscope)...")

        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            load_from,
            torch_dtype=self.fp_dtype,
            device_map=device_map,  # Llama3.2V 推荐使用 auto device map
        ).to(device_id)
        self.processor = MS_AutoProcessor.from_pretrained(load_from)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs = []
        # 当前处理器实现可能不直接支持批处理图像+文本，采用循环处理确保正确性
        for item in batch_items:
            image = item["original_image"]
            question = item["question"]

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False, # 根据官方示例
                return_tensors="pt"
            ).to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            # 解码并去除输入部分
            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self.processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
            outputs.append(generated_text.strip())
        
        return outputs


# ---- LLaVA-1.5 (ModelScope/HF) 适配器 ----
def _parse_device(device: str | int) -> int | str:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        if device.startswith("cuda:"):
            return int(device.split(":")[1])
        if device == "cuda":
            return 0
        if device == "cpu":
            return "cpu"
    return device

class Llava15MSAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, lora_path: str, device: str, max_new_tokens: int = 200,
                 do_sample: bool = False, fp_dtype=torch.float16):
        super().__init__(model_id, model_path, lora_path, device, max_new_tokens)
        self.do_sample = do_sample
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[LLaVA-MS] 从 {load_from} 加载模型与处理器 (modelscope/HF)...")
        
        device_id = _parse_device(self.device)
        
        self.model = HF_LlavaForConditionalGeneration.from_pretrained(
            load_from,
            torch_dtype=self.fp_dtype,
            low_cpu_mem_usage=True,
        ).to(device_id)
        
        self.processor = MS_AutoProcessor.from_pretrained(load_from)

        self.pad_token_id = (
            self.model.config.pad_token_id
            if getattr(self.model.config, "pad_token_id", None) is not None
            else self.model.config.eos_token_id
        )
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs: List[str] = []
        for it in batch_items:
            raw_image = it["original_image"]
            question = it['question']
            
            conversation = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]}]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(self.model.device, self.fp_dtype)

            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, pad_token_id=self.pad_token_id)
            
            text = self.processor.decode(out_ids[0][2:], skip_special_tokens=True)
            outputs.append(text.strip())
            
        return outputs


#---- Qwen2.5-VL 适配器 ----
class Qwen25VLAdapter(BaseVLMAdapter):
    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Qwen2.5-VL] 从 {load_from} 加载模型与处理器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            load_from, torch_dtype="auto", device_map=device_map
        ).to(device_id)
        self.processor = MS_AutoProcessor.from_pretrained(load_from, trust_remote_code=True)

        if self.lora_path:
            if PeftModel is None: raise ImportError("未安装 peft；请执行 pip install peft")
            print(f"[Qwen2.5-VL] 应用 LoRA: {self.lora_path}")
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model
        
        self.model.eval()
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        if process_vision_info is None: 
            raise ImportError("使用 Qwen 后端需要 qwen_vl_utils.py。")

        # 1. 构建消息批次
        messages_list = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": item["original_image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }] for item in batch_items
        ]

        # 2. 批量处理文本和视觉信息
        texts_for_batch = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        images_for_batch, _ = process_vision_info(messages_list)

        # 3. 将批量的文本和图像送入处理器
        inputs = self.processor(
            text=texts_for_batch,
            images=images_for_batch,
            videos=None,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(self.model.device)

        # 4. 批量推理
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # 5. 批量解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return [o.strip() for o in outputs]

#---- Qwen2-VL 适配器 ----
class Qwen2VLAdapter(BaseVLMAdapter):
    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Qwen2-VL] 从 {load_from} 加载模型与处理器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            load_from, torch_dtype="auto", device_map=device_map
        ).to(device_id)
        self.processor = MS_AutoProcessor.from_pretrained(load_from, trust_remote_code=True)

        if self.lora_path:
            if PeftModel is None: raise ImportError("未安装 peft；请执行 pip install peft")
            print(f"[Qwen2.5-VL] 应用 LoRA: {self.lora_path}")
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model
        
        self.model.eval()
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."

        # 1. 构建消息批次
        messages_list = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": item["original_image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }] for item in batch_items
        ]

        # 2. 批量处理文本和视觉信息
        texts_for_batch = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        images_for_batch, _ = process_vision_info(messages_list)

        # 3. 将批量的文本和图像送入处理器
        inputs = self.processor(
            text=texts_for_batch,
            images=images_for_batch,
            videos=None,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(self.model.device)

        # 4. 批量推理
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # 5. 批量解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return [o.strip() for o in outputs]

# ---- InternVL3 适配器 ----
def _device_map_from_device_str(device: str):
    if not device or device == "auto": return "auto"
    d = device.lower()
    if d.startswith("cuda"):
        if ":" in d:
            try: return {"": int(d.split(":")[1])}
            except Exception: return "auto"
        return "auto"
    if d == "cpu": return {"": "cpu"}
    return "auto"

class InternVL3HFAdapter(BaseVLMAdapter):
    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[InternVL3-HF] 从 {load_from} 加载模型与处理器（transformers）...")
        self.processor = HF_AutoProcessor.from_pretrained(load_from)
        device_map = _device_map_from_device_str(self.device)
        self.model = AutoModelForImageTextToText.from_pretrained(
            load_from, device_map=device_map, torch_dtype=torch.bfloat16).eval()
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs: List[str] = []
        for it in batch_items:
            messages = [{"role": "user", "content": [{"type": "image", "image": it["original_image"]}, {"type": "text", "text": it["question"]}]}]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            
            inputs_on_device = {}
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    v = v.to(self.model.device)
                    if torch.is_floating_point(v): v = v.to(torch.bfloat16)
                    inputs_on_device[k] = v

            generate_ids = self.model.generate(**inputs_on_device, max_new_tokens=self.max_new_tokens)
            prompt_len = inputs_on_device["input_ids"].shape[1]
            decoded = self.processor.decode(generate_ids[0, prompt_len:], skip_special_tokens=True)
            outputs.append(decoded.strip())

        return outputs

#---- Gemma3-4B 适配器 ----
class GemmaE4BITHFAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.bfloat16):
        super().__init__(model_id, model_path, None, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[GEMMA3n] 从 {load_from} 加载模型与分词器...")
        device_id = _parse_device(self.device)
        device_map = _device_map_from_device_str(self.device)
        self.processor = AutoProcessor.from_pretrained(load_from)
        
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=load_from,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).to(device_id)
        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs: List[str] = []
        for it in batch_items:
            raw_image = it["original_image"]
            question = it["question"]

            # pixtral 官方示例消息结构（text 节点使用 "content" 键）
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "content": question},
                    ]
                }
            ]
            # 与示例一致：text=prompt, images=[<PIL 或 URL>]
            inputs = self.processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            
            # pixtral 示例直接 batch_decode，不裁剪 prompt
            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            outputs.append(decoded.strip())
        return outputs
    
#---- Pixtral-12B (ModelScope) 适配器 ----
class PixtralMSAdapter(BaseVLMAdapter):
    def __init__(self, model_id: str, model_path: str, lora_path: str, device: str, max_new_tokens: int = 128, fp_dtype=torch.float16):
        super().__init__(model_id, model_path, lora_path, device, max_new_tokens)
        self.fp_dtype = fp_dtype

    def load(self):
        load_from = self.model_path if self.model_path else self.model_id
        print(f"[Pixtral-MS] 从 {load_from} 加载模型与处理器 (modelscope)...")
        device_id = _parse_device(self.device)

        self.model = HF_LlavaForConditionalGeneration.from_pretrained(
            load_from
        ).to(device_id)

        # 注意：pixtral 示例使用 modelscope 的 AutoProcessor
        self.processor = MS_AutoProcessor.from_pretrained(load_from)

        self._loaded = True
        return self

    @torch.no_grad()
    def generate_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        assert self._loaded, "Call load() first."
        outputs: List[str] = []
        for it in batch_items:
            raw_image = it["original_image"]
            question = it["question"]

            # pixtral 官方示例消息结构（text 节点使用 "content" 键）
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": question},
                        {"type": "image"},
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(chat)

            # 与示例一致：text=prompt, images=[<PIL 或 URL>]
            inputs = self.processor(
                text=prompt,
                images=[raw_image],
                return_tensors="pt"
            ).to(self.model.device)

            generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            # pixtral 示例直接 batch_decode，不裁剪 prompt
            output = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            outputs.append(output.strip())

        return outputs


# =========================
# 主流程
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="多模态 VLM 统一评估脚本")
    parser.add_argument("--backend", type=str, required=True,
                        choices=["internvl", "qwen2.5vl","qwen2vl", "llava_ms", "llama32v_ms", "pixtral_ms", "minicpmv2_6", "gemma3", "pali_gemma2_10b", "llava_next","keyevl","minicpm4v","glm4.1v", "gemma3n"],
                        help="选择评测后端")
    parser.add_argument("--model_name", type=str, default=None, help="HF / ModelScope 模型 ID")
    parser.add_argument("--model_path", type=str, default=None, help="本地模型路径（优先于 model_name)")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 适配器路径（可选，仅部分后端支持）")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备 (例如 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--data_path", type=str, required=True, help=".json 数据路径")
    parser.add_argument("--image_prefix", type=str, required=True, help="JSON 图像字段的前缀目录")
    parser.add_argument("--save_path", type=str, required=True, help="保存结果和失败样本的目录")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    return args


def build_adapter(args) -> BaseVLMAdapter:
    # LLaVA-NEXT
    if args.backend == "llava_next":
        model_id = args.model_name or "LLaVA-Next/LLaVA-Next-7B"
        adapter = LLaVANextAdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    if args.backend == "gemma3n":
        model_id = args.model_name or "google/gemma-3n-e4b-it"
        adapter = GemmaE4BITHFAdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    if args.backend == "glm4.1v":
        model_id = args.model_name or "GLM4.1V"
        adapter = GLM41VAdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    # keyevl
    if args.backend == "keyevl":
        model_id = args.model_name or "keye-vl/keye-vl-7b"
        adapter = KeyeVLAdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    if args.backend == "minicpm4v":
        model_id = args.model_name or "Minicpm/Minicpm-v-4"
        adapter = MiniCPMA4Vdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    # PaliGemma2-10B
    if args.backend == "pali_gemma2_10b":
        model_id = args.model_name or "PaliGemma/PaliGemma2-10B"
        adapter = PaliGemmaMSAdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    # Gemma3-12B
    if args.backend == "gemma3":
        model_id = args.model_name or "Gemma3/Gemma3-12B"
        adapter = Gemma3Adapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    
    # Minicpm-v-2-6
    if args.backend == "minicpmv2_6":
        model_id = args.model_name or "Minicpm/Minicpm-v-2-6"
        adapter = MiniCPMAdapter(
            model_id=model_id,
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,  # 推荐使用 bfloat16
        )
        return adapter.load()
    
    #Llama-3.2-Vision (ModelScope)
    if args.backend == "llama32v_ms":
        model_id = args.model_name or "LLM-Research/Llama-3.2-11B-Vision-Instruct"
        adapter = Llama32VisionMSAdapter(
            model_id=model_id,
            model_path=args.model_path,
            lora_path=None,  # 当前脚本不支持为此模型加载LoRA
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fp_dtype=torch.bfloat16,
        )
        return adapter.load()

    #LLaVA (HF + ModelScope Processor)
    if args.backend == "llava_ms":
        model_id = args.model_name or "swift/llava-1.5-7b-hf"
        adapter = Llava15MSAdapter(
            model_id=model_id, model_path=args.model_path, lora_path=None,
            device=args.device, max_new_tokens=args.max_new_tokens,
            do_sample=False, fp_dtype=torch.float16,
        )
        return adapter.load()

    #Qwen2.5-VL
    if args.backend == "qwen2.5vl":
        model_id = args.model_name or "Qwen/Qwen2.5-VL-7B-Instruct"
        adapter = Qwen25VLAdapter(
            model_id=model_id, model_path=args.model_path, lora_path=args.lora_path,
            device=args.device, max_new_tokens=args.max_new_tokens,
        )
        return adapter.load()
    
    if args.backend == "qwen2vl":
        model_id = args.model_name or "Qwen/Qwen2-VL-7B-Instruct"
        adapter = Qwen2VLAdapter(
            model_id=model_id, model_path=args.model_path, lora_path=args.lora_path,
            device=args.device, max_new_tokens=args.max_new_tokens,
        )
        return adapter.load()

    #InternVL3 (HF)
    if args.backend == "internvl":
        model_id = args.model_name or "OpenGVLab/InternVL3-8B"
        adapter = InternVL3HFAdapter(
            model_id=model_id, model_path=args.model_path, lora_path=None,
            device=args.device, max_new_tokens=args.max_new_tokens,
        )
        return adapter.load()

    #Pixtral-12B (ModelScope)
    if args.backend == "pixtral_ms":
        model_id = args.model_name or "AI-ModelScope/pixtral-12b"
        adapter = PixtralMSAdapter(
            model_id=model_id, model_path=args.model_path, lora_path=None,
            device=args.device, max_new_tokens=args.max_new_tokens, fp_dtype=torch.float16
        )
        return adapter.load()

    raise ValueError(f"未知 backend: {args.backend}")


def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    disable_torch_init()

    os.makedirs(args.save_path, exist_ok=True)
    print(f"保存目录: {args.save_path}")

    model_tag = args.model_path if args.model_path else (args.model_name or args.backend)
    results_file = create_results_file(str(model_tag), args.save_path)

    adapter = build_adapter(args)

    image_paths, questions, gts, ranges = load_data(args.data_path, args.image_prefix)
    dataset = preload_images(image_paths, questions)

    absolute_errors, processed_ranges, valid_theta_samples = [], [], []
    correct_count_theta = nonzero_gt_count_theta = accuracy_epsilon_correct_count = processed_samples_count = 0

    batch_size = max(1, int(args.batch_size))
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for bidx in tqdm(range(num_batches), desc="评估进度"):
        s = bidx * batch_size
        e = min((bidx + 1) * batch_size, num_samples)
        batch = dataset[s:e]
        if not batch: continue

        try:
            preds = adapter.generate_batch(batch)
            if len(preds) != len(batch):
                print(f"[警告] 生成数量与输入不一致：{len(preds)} vs {len(batch)}")
                preds = preds[:len(batch)]
        except Exception as ex:
            print(f"[错误] 批次 {bidx} 推理失败: {ex}")
            import traceback; traceback.print_exc()
            continue

        for i, pred_text in enumerate(preds):
            item = batch[i]
            idx, rg = item["original_index"], ranges[item["original_index"]]
            gt = gts[idx]
            err = abs(extract_last_number(pred_text) - gt)

            absolute_errors.append(err)
            processed_ranges.append(rg)
            if err <= 0.01 * rg: accuracy_epsilon_correct_count += 1

            if gt != 0:
                nonzero_gt_count_theta += 1
                theta_sample = err / gt
                valid_theta_samples.append(theta_sample)
                if theta_sample < 0.05:
                    correct_count_theta += 1
                else:
                    save_name = f"sample_{idx+1:05d}.jpg"
                    save_path = os.path.join(args.save_path, save_name)
                    try: item["original_image"].save(save_path)
                    except Exception as se: print(f"[警告] 保存图像失败 {item['original_path']} -> {save_path}: {se}")

            processed_samples_count += 1

        if e > 0 and (e % 100 < batch_size or e == num_samples) and processed_samples_count > 0:
            batch_metrics = calculate_metrics(
                absolute_errors, processed_ranges, valid_theta_samples,
                correct_count_theta, nonzero_gt_count_theta,
                accuracy_epsilon_correct_count, processed_samples_count
            )
            print(f"\n处理完成 {e}/{num_samples} 样本")
            print_metrics(batch_metrics, batch=True)

    final_metrics = calculate_metrics(
        absolute_errors, processed_ranges, valid_theta_samples,
        correct_count_theta, nonzero_gt_count_theta,
        accuracy_epsilon_correct_count, processed_samples_count
    )
    print_metrics(final_metrics, batch=False)
    save_metrics_to_file(final_metrics, results_file)

if __name__ == "__main__":
    main()
