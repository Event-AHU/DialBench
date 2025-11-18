import argparse
import numpy as np
from PIL import Image
import io
import base64
from openai import OpenAI
import torch
from tqdm import tqdm
import json
import os
import re
import sys
import time
import random

# ---------- helpers ----------
def disable_torch_init():
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation with GPT-5 (multimodal) via domestic relay")
    parser.add_argument("--data_path", type=str, required=True, help="JSON with items: {image, query, reading, range}")
    parser.add_argument("--image_prefix", type=str, required=True, help="Prefix for image paths in JSON")
    parser.add_argument("--save_path", type=str, required=True, help="Dir to save images with Theta>0.05")
    parser.add_argument("--responses_json", type=str, required=True, help="Path to save responses (image filename as key)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--openai_api_key", type=str, default='xx', help="If omitted, use env OPENAI_API_KEY")
    parser.add_argument("--api_base", type=str, required=True, help="Your relay base URL, e.g. https://api.your-proxy.com/v1")
    return parser.parse_args()

def pil_to_data_url(img: Image.Image, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def load_or_init_json(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] Failed to read existing responses_json: {e}; starting fresh.", file=sys.stderr)
    return {}

def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def chat_once_with_retry(client: OpenAI, model: str, image_url: str, question: str,
                         max_tokens: int, temperature: float, max_retries: int = 4) -> str:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content":
                        "You are a vision model. Read the image and answer the question. "
                    },
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            jitter = random.uniform(0, 0.5)
            time.sleep(delay + jitter)
            delay *= 2
    return ""

def eval_batch(images, questions, client: OpenAI, model: str, max_tokens: int, temperature: float):
    responses = []
    for img, q in zip(images, questions):
        if isinstance(img, Image.Image):
            image_url = pil_to_data_url(img, fmt="JPEG")
        elif isinstance(img, str):
            _im = Image.open(img).convert("RGB")
            image_url = pil_to_data_url(_im, fmt="JPEG")
        else:
            raise ValueError("Unsupported image type in eval_batch.")
        text = chat_once_with_retry(client, model, image_url, q, max_tokens, temperature)
        responses.append(text)
    return responses

def extract_last_number(text):
    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if matches:
        return float(matches[-1])
    print(f"No number found in text: {text}")
    return 0.0

def load_data(file_path, image_prefix):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_paths = [os.path.join(image_prefix, item['image']) for item in data]
    questions = [item['query'] for item in data]
    ground_truths = [extract_last_number(str(item['reading'])) for item in data]
    ranges = [float(item['range']) for item in data]
    return image_paths, questions, ground_truths, ranges

def calculate_metrics(absolute_errors, ranges,
                      valid_errors_for_theta, valid_ground_truths_for_theta,
                      correct_count_theta, nonzero_gt_count_theta,
                      accuracy_epsilon_correct_count, processed_samples_count):
    metrics = {}
    if len(valid_ground_truths_for_theta) > 0:
        theta_ratios = np.array(valid_errors_for_theta) / np.array(valid_ground_truths_for_theta)
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
    prefix = "Batch " if batch else "Final "
    print(f"\n--- {prefix}Evaluation Results ---")
    print(f" Theta (Θ̂): {metrics['theta']:.4f}" if metrics["theta"] is not None else "No valid data for Theta calculation.")
    print(f" Gamma (Γ̂): {metrics['gamma']:.4f}" if metrics["gamma"] is not None else "No valid data for Gamma calculation.")
    print(f" Accuracy (Theta < 0.05): {metrics['accuracy_theta']:.4f}" if metrics["accuracy_theta"] is not None else "No valid samples for accuracy (Theta < 0.05) calculation.")
    print(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}" if metrics["accuracy_epsilon"] is not None else "No samples were processed for Accuracy@ε calculation.")
    print("------------------------\n")

# ---------- main ----------
def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    disable_torch_init()

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Saving images with Theta>0.05 to: {args.save_path}")

    # OpenAI client via domestic relay
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API key missing: pass --openai_api_key or set OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=args.api_base.rstrip("/"))

    # load / init response json
    resp_store = load_or_init_json(args.responses_json)

    image_paths, questions, ground_truths, ranges = load_data(args.data_path, args.image_prefix)

    absolute_errors = []
    processed_ranges = []
    valid_errors_for_theta = []
    valid_ground_truths_for_theta = []
    correct_count_theta = 0
    nonzero_gt_count_theta = 0
    accuracy_epsilon_correct_count = 0
    processed_samples_count = 0

    batch_size = args.batch_size
    num_samples = len(image_paths)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_indices = list(range(start_idx, end_idx))

        batch_images = []
        batch_questions = []
        batch_triplets = []

        for idx in batch_indices:
            img_path = image_paths[idx]
            try:
                original_image = Image.open(img_path).convert('RGB')
                batch_images.append(original_image)
                batch_questions.append(questions[idx])
                batch_triplets.append((idx, img_path, original_image))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        if not batch_images:
            continue

        predictions = eval_batch(
            batch_images, batch_questions,
            client=client, model=args.model,
            max_tokens=args.max_tokens, temperature=args.temperature
        )

        for i, (idx, img_path, original_image) in enumerate(batch_triplets):
            if i >= len(predictions):
                continue
            pred = predictions[i]
            img_key = os.path.basename(img_path)

            # —— 保存 response（以图片名为主键） ——
            resp_store[img_key] = {"response": pred}
            atomic_write_json(args.responses_json, resp_store)

            ground_truth = ground_truths[idx]
            range_value = ranges[idx]
            pred_number = extract_last_number(pred)
            error = abs(pred_number - ground_truth)

            absolute_errors.append(error)
            processed_ranges.append(range_value)

            epsilon = 0.01 * range_value
            if error <= epsilon:
                accuracy_epsilon_correct_count += 1

            if ground_truth != 0:
                nonzero_gt_count_theta += 1
                theta_sample = error / ground_truth

                if theta_sample < 0.05:
                    correct_count_theta += 1

                capped_theta = min(theta_sample, 100.0)
                valid_errors_for_theta.append(capped_theta * ground_truth)
                valid_ground_truths_for_theta.append(ground_truth)

                if theta_sample > 0.05:
                    filename = f"sample_{idx+1:05d}.jpg"
                    save_path = os.path.join(args.save_path, filename)
                    try:
                        original_image.save(save_path)
                    except Exception as e:
                        print(f"Error saving image {img_path} to {save_path}: {e}")
            else:
                print(f"Ground truth is 0 for image {img_path}. Skipping Theta calculation.")

            processed_samples_count += 1

        if end_idx % 100 <= batch_size and end_idx != num_samples:
            batch_metrics = calculate_metrics(
                absolute_errors, processed_ranges,
                valid_errors_for_theta, valid_ground_truths_for_theta,
                correct_count_theta, nonzero_gt_count_theta,
                accuracy_epsilon_correct_count, processed_samples_count
            )
            print(f"\n处理完成 {end_idx}/{len(image_paths)} 样本")
            print_metrics(batch_metrics, batch=True)

    final_metrics = calculate_metrics(
        absolute_errors, processed_ranges,
        valid_errors_for_theta, valid_ground_truths_for_theta,
        correct_count_theta, nonzero_gt_count_theta,
        accuracy_epsilon_correct_count, processed_samples_count
    )
    print_metrics(final_metrics, batch=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
