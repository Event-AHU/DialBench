import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import json
import os
import re

# 导入适配 Qwen-VL 的特定库
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor,Qwen2VLForConditionalGeneration,AutoModel
from qwen_vl_utils import process_vision_info

# 导入 PEFT 库，用于加载 LoRA 权重
try:
    from peft import PeftModel
except ImportError:
    raise ImportError("请先安装peft库: 'pip install peft'")


def disable_torch_init():
    """
    禁用冗余的 torch 默认初始化以加速模型创建。
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def parse_args():
    """
    从命令行解析参数。
    """
    parser = argparse.ArgumentParser(description="用于模型评估的参数")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="要使用的 Hugging Face 基础模型标识符 (例如 'Qwen/Qwen2.5-VL-7B-Instruct')。")
    parser.add_argument("--model_path", type=str, default=None,
                        help="本地基础模型文件的路径。如果提供此参数，将优先于 --model_name。")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="可选：包含 LoRA 适配器权重的本地文件夹路径。")
    parser.add_argument("--device", type=str, default="cuda", help="指定要使用的设备（例如 'cuda', 'cpu'）。")
    parser.add_argument("--data_path", type=str, required=True, help="包含图像路径、问题和答案的 .json 文件路径。")
    parser.add_argument("--image_prefix", type=str, required=True, help="要添加到 JSON 文件中图像路径的前缀。")
    parser.add_argument("--save_path", type=str, required=True, help="用于保存 Theta > 0.05 的图像的目录。")
    parser.add_argument("--batch_size", type=int, default=1, help="评估的批处理大小。")
    args = parser.parse_args()
    return args


def load_model_and_processor(model_name, model_path, lora_path, device):
    """
    使用 modelscope 和 transformers 库加载 Qwen2.5-VL 模型和对应的处理器。
    """
    load_location = model_path if model_path else model_name
    print(f"正在从 '{load_location}' 加载基础模型和处理器...")
    try:
        processor = AutoProcessor.from_pretrained(load_location, trust_remote_code=True)
        
        # 1. 移除 device_map="auto"
        base_model = AutoModel.from_pretrained(
            load_location, 
            torch_dtype="auto", 
            trust_remote_code=True
        )
        print("基础模型和处理器加载完成。")
    except Exception as e:
        print(f"从 '{load_location}' 加载基础模型失败: {e}")
        raise

    if lora_path:
        print(f"正在从路径 '{lora_path}' 加载并应用 LoRA 权重...")
        try:
            model = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA 权重应用成功。")
        except Exception as e:
            print(f"加载 LoRA 权重失败: {e}")
            raise
    else:
        model = base_model
        
    # 2. 重新启用这一行，将模型手动移动到指定设备
    print(f"正在将模型移动到设备: {device}...")
    model.to(device)
    print("模型移动完成。")
    
    return model, processor


def preload_and_prepare_data(image_paths, questions):
    """
    一次性加载所有图像并准备好用于模型输入的数据结构。
    *** 此处新增图像缩放逻辑 ***
    """
    print("开始预加载和准备所有数据（图像将缩放至 224x224）...")
    prepared_data = []
    for i in tqdm(range(len(image_paths)), desc="预处理数据"):
        img_path = image_paths[i]
        question = questions[i]
        try:
            # 1. 加载原始图像
            original_image = Image.open(img_path).convert('RGB')
            
            # 2. 创建一个缩放后的版本用于模型推理
            # 使用 LANCZOS 滤镜以获得高质量的缩放结果
            inference_image = original_image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 3. 构建 messages，使用缩放后的图像
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": inference_image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            # 4. 存储处理好的数据，同时保留原始图像用于保存
            prepared_data.append({
                'messages': messages,
                'original_image': original_image, # 保留原始图像
                'original_path': img_path,
                'original_index': i
            })
        except Exception as e:
            print(f"预处理图像 {img_path} 时出错: {e}")
    
    print(f"数据预加载和准备完成。共处理 {len(prepared_data)} 条数据。")
    return prepared_data


def eval_batch(model, processor, batch_prepared_data):
    """
    使用 Qwen2.5-VL 模型对一个批次的预处理数据进行推理。
    此函数现在只关注模型推理。
    """
    device = model.device
    
    texts_for_batch = []
    images_for_batch = []
    
    # 1. 从预处理好的数据中提取 messages
    for item in batch_prepared_data:
        messages = item['messages']
        
        # 2. 使用模板生成文本，并使用辅助函数提取图像
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        texts_for_batch.append(text)
        images_for_batch.extend(image_inputs)

    # 3. 将整个批次的文本和图像传递给 processor 进行张量转换
    inputs = processor(
        text=texts_for_batch,
        images=images_for_batch,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 4. 使用模型进行推理生成
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # 5. 从生成结果中裁剪掉输入的 token 部分，只保留回答
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # 6. 解码生成结果
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return [pred.strip() for pred in output_text]


def extract_last_number(text):
    """
    从字符串中提取最后一个数字（整数或浮点数）。
    """
    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if matches:
        return float(matches[-1])
    else:
        print(f"在文本中未找到数字: {text}")
        return 0.0


def load_data(file_path, image_prefix):
    """
    从 .json 文件加载图像路径、问题、真实答案和范围。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    image_paths = [os.path.join(image_prefix, item['image']) for item in data]
    questions = [item['query'] for item in data]
    ground_truths = [extract_last_number(str(item['reading'])) for item in data]
    ranges = [float(item['range']) for item in data]

    return image_paths, questions, ground_truths, ranges


def calculate_metrics(absolute_errors, ranges,
                      valid_theta_samples,
                      correct_count_theta, nonzero_gt_count_theta,
                      accuracy_epsilon_correct_count, processed_samples_count):
    """
    根据当前数据计算并返回指标。
    """
    metrics = {}

    if len(valid_theta_samples) > 0:
        theta_ratios = np.array(valid_theta_samples)
        theta_ratios = np.clip(theta_ratios, None, 100)
        metrics["theta"] = np.mean(theta_ratios)
    else:
        metrics["theta"] = None

    if len(absolute_errors) > 0 and len(ranges) > 0:
        gamma_ratios = np.array(absolute_errors) / np.array(ranges)
        gamma_ratios = np.clip(gamma_ratios, None, 10)
        metrics["gamma"] = np.mean(gamma_ratios)
    else:
        metrics["gamma"] = None

    if nonzero_gt_count_theta > 0:
        metrics["accuracy_theta"] = correct_count_theta / nonzero_gt_count_theta
    else:
        metrics["accuracy_theta"] = None

    if processed_samples_count > 0:
        metrics["accuracy_epsilon"] = accuracy_epsilon_correct_count / processed_samples_count
    else:
        metrics["accuracy_epsilon"] = None

    return metrics


def print_metrics(metrics, batch=False):
    """
    格式化打印指标。
    """
    prefix = "批次 " if batch else "最终 "
    print(f"\n--- {prefix}评估结果 ---")
    if metrics["theta"] is not None: print(f" Theta (Θ̂): {metrics['theta']:.4f}")
    if metrics["gamma"] is not None: print(f" Gamma (Γ̂): {metrics['gamma']:.4f}")
    if metrics["accuracy_theta"] is not None: print(f" Accuracy (Theta < 0.05): {metrics['accuracy_theta']:.4f}")
    if metrics["accuracy_epsilon"] is not None: print(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}")
    print("------------------------\n")


def create_results_file(model_name, save_dir):
    """
    创建一个文件，根据模型名称保存评估结果。
    """
    # 替换模型名称中的斜杠为下划线，以便创建合法的文件名
    safe_model_name = model_name.replace('/', '_')
    results_file_path = os.path.join(save_dir, f"evaluation_results_{safe_model_name}.txt")
    
    # 如果文件不存在，创建一个新的文件
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w') as f:
            f.write(f"Evaluation Results for {model_name}\n")
            f.write("=" * 50 + "\n")
    return results_file_path


def save_metrics_to_file(metrics, file_path):
    """
    将计算得到的指标保存到文件中。
    """
    with open(file_path, 'a', encoding='utf-8') as f:  # Explicitly specify utf-8 encoding
        f.write(f"\n--- Evaluation Metrics ---\n")
        if metrics["theta"] is not None: f.write(f" Theta (Θ̂): {metrics['theta']:.4f}\n")
        if metrics["gamma"] is not None: f.write(f" Gamma (Γ̂): {metrics['gamma']:.4f}\n")
        if metrics["accuracy_theta"] is not None: f.write(f" Accuracy (Theta < 0.05): {metrics['accuracy_theta']:.4f}\n")
        if metrics["accuracy_epsilon"] is not None: f.write(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}\n")
        f.write("-" * 50 + "\n")



def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    disable_torch_init()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"已创建用于保存图像的目录: {args.save_path}")

    # Create the results file based on the model name
    results_file_path = create_results_file(args.model_name, args.save_path)

    model, processor = load_model_and_processor(
        args.model_name, args.model_path, args.lora_path, args.device
    )
    model.eval()

    image_paths, questions, ground_truths, ranges = load_data(args.data_path, args.image_prefix)
    
    # --- 核心改动：在这里预处理所有数据 ---
    prepared_dataset = preload_and_prepare_data(image_paths, questions)
    # ------------------------------------

    absolute_errors, processed_ranges, valid_theta_samples = [], [], []
    correct_count_theta, nonzero_gt_count_theta, accuracy_epsilon_correct_count, processed_samples_count = 0, 0, 0, 0

    batch_size = args.batch_size
    num_samples = len(prepared_dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="评估进度"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        # 从预处理好的数据集中获取批次
        batch_data = prepared_dataset[start_idx:end_idx]
        if not batch_data:
            continue

        try:
            with torch.no_grad():
                # eval_batch 现在接收预处理好的数据
                predictions = eval_batch(model, processor, batch_data)

            for i, pred_text in enumerate(predictions):
                # 从 batch_data 中获取元信息
                item_info = batch_data[i]
                original_idx = item_info['original_index']
                
                ground_truth = ground_truths[original_idx]
                range_value = ranges[original_idx]

                pred_number = extract_last_number(pred_text)
                error = abs(pred_number - ground_truth)

                absolute_errors.append(error)
                processed_ranges.append(range_value)

                epsilon = 0.01 * range_value
                if error <= epsilon:
                    accuracy_epsilon_correct_count += 1

                if ground_truth != 0:
                    nonzero_gt_count_theta += 1
                    theta_sample = error / ground_truth
                    valid_theta_samples.append(theta_sample)

                    if theta_sample < 0.05:
                        correct_count_theta += 1
                    else:
                        # 保存 theta > 0.05 的图像
                        filename = f"sample_{original_idx+1:05d}.jpg"
                        save_path = os.path.join(args.save_path, filename)
                        try:
                            item_info['original_image'].save(save_path)
                        except Exception as e:
                            print(f"保存图像 {item_info['original_path']} 到 {save_path} 时出错: {e}")
                
                processed_samples_count += 1

        except Exception as e:
            print(f"批次 {batch_idx} 推理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        # 定期打印和保存指标
        if end_idx % 100 < batch_size and end_idx > 0 and end_idx != num_samples:
            batch_metrics = calculate_metrics(
                absolute_errors, processed_ranges, valid_theta_samples,
                correct_count_theta, nonzero_gt_count_theta,
                accuracy_epsilon_correct_count, processed_samples_count
            )
            print(f"\n处理完成 {end_idx}/{num_samples} 样本")
            print_metrics(batch_metrics, batch=True)
            # 保存指标到文件
            save_metrics_to_file(batch_metrics, results_file_path)

    # 最终指标
    final_metrics = calculate_metrics(
        absolute_errors, processed_ranges, valid_theta_samples,
        correct_count_theta, nonzero_gt_count_theta,
        accuracy_epsilon_correct_count, processed_samples_count
    )
    print_metrics(final_metrics, batch=False)
    # 保存最终指标到文件
    save_metrics_to_file(final_metrics, results_file_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)