import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bliva.models import load_model_and_preprocess
import torch
from tqdm import tqdm
import json
import os
import re

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for Evaluation")
    parser.add_argument("--model_name", type=str, default="bliva_vicuna", help="Name of the model to use.")
    parser.add_argument("--device", type=str, default="cuda:5", help="Specify which GPU device to use.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the .json file containing image paths, questions, and ground truth answers.")
    parser.add_argument("--image_prefix", type=str, required=True, help="Prefix to add to image paths in the JSON file.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save images with Theta < 0.05.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation.")
    args = parser.parse_args()
    return args

def eval_batch(images, questions, model):
    """
    Evaluate a batch of questions with the model
    """
    outputs = model.generate({"image": images, "prompt": questions})
    return outputs

def extract_last_number(text):
    """
    Extract the last number (integer or float) found in a string.
    """
    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if matches:
        return float(matches[-1])
    else:
        print(f"No number found in text: {text}")
        return 0.0

def load_data(file_path, image_prefix):
    """
    Load image paths, questions, ground truth answers, and ranges from a .json file.
    Add a prefix to each image path.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    image_paths = [os.path.join(image_prefix, item['image']) for item in data]
    questions = [item['query'] for item in data]
    ground_truths = [extract_last_number(str(item['reading'])) for item in data]
    ranges = [float(item['range']) for item in data]
    meter_types = [item['meter_type'] for item in data]
    environment_conditions = [item['environment_conditions'].split(',') for item in data]  # Splitting multiple conditions

    return image_paths, questions, ground_truths, ranges, meter_types, environment_conditions

def calculate_metrics(absolute_errors, ranges,
                      valid_errors_for_theta, valid_ground_truths_for_theta, 
                      correct_count_theta, nonzero_gt_count_theta, 
                      accuracy_epsilon_correct_count, processed_samples_count):
    """
    Calculate and return metrics based on current data
    """
    metrics = {}
    
    # Theta (capped at 100)
    if len(valid_ground_truths_for_theta) > 0:
        theta_ratios = np.array(valid_errors_for_theta) / np.array(valid_ground_truths_for_theta)
        theta_ratios = np.clip(theta_ratios, None, 100)
        metrics["theta"] = np.mean(theta_ratios)
    else:
        metrics["theta"] = None
    
    # Gamma (capped at 10)
    if len(absolute_errors) > 0 and len(ranges) > 0:
        gamma_ratios = np.array(absolute_errors) / np.array(ranges)
        gamma_ratios = np.clip(gamma_ratios, None, 10)
        metrics["gamma"] = np.mean(gamma_ratios)
    else:
        metrics["gamma"] = None
    
    # Accuracy (Theta < 0.05)
    if nonzero_gt_count_theta > 0:
        metrics["accuracy_theta"] = correct_count_theta / nonzero_gt_count_theta
    else:
        metrics["accuracy_theta"] = None
    
    # Accuracy@ε
    if processed_samples_count > 0:
        metrics["accuracy_epsilon"] = accuracy_epsilon_correct_count / processed_samples_count
    else:
        metrics["accuracy_epsilon"] = None
    
    return metrics

def print_metrics(metrics, batch=False):
    """
    Print metrics in a formatted way
    """
    prefix = "Batch " if batch else "Final "
    
    print(f"\n--- {prefix}Evaluation Results ---")
    
    if metrics["theta"] is not None:
        print(f" Theta (Θ̂): {metrics['theta']:.4f}")
    else:
        print("No valid data for Theta calculation.")
    
    if metrics["gamma"] is not None:
        print(f" Gamma (Γ̂): {metrics['gamma']:.4f}")
    else:
        print("No valid data for Gamma calculation.")
    
    if metrics["accuracy_theta"] is not None:
        print(f" Accuracy (Theta < 0.05): {metrics['accuracy_theta']:.4f}")
    else:
        print("No valid samples for accuracy (Theta < 0.05) calculation.")
    
    if metrics["accuracy_epsilon"] is not None:
        print(f" Accuracy@ε (ε=0.01*range): {metrics['accuracy_epsilon']:.4f}")
    else:
        print("No samples were processed for Accuracy@ε calculation.")
    
    print("------------------------\n")

def save_image_with_predictions(image, save_path, ground_truth, pred_number):
    """
    Save the image with the ground truth (green) and prediction (red) values written on it,
    using a larger, bold font if available, with black stroke for readability.
    """
    draw = ImageDraw.Draw(image)
    # Try a common bold TTF; fallback to default if unavailable
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=36)
    except Exception:
        # load_default 没有 size 参数，但至少保证可用
        font = ImageFont.load_default()

    # Text content
    gt_text = f"GT: {ground_truth:.2f}"
    pred_text = f"Pred: {pred_number:.2f}"

    # Positions
    x, y = 10, 10
    line_gap = 10
    # If truetype font, use font.size；否则用一个保守的行高
    line_height = getattr(font, "size", 20) + line_gap

    # Draw with stroke to enhance visibility
    draw.text((x, y), gt_text, font=font, fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((x, y + line_height), pred_text, font=font, fill=(255, 0, 0), stroke_width=2, stroke_fill=(0, 0, 0))

    image.save(save_path)

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    disable_torch_init()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"Created directory for saving images: {args.save_path}")
    else:
        print(f"Saving images to existing directory: {args.save_path}")

    if args.model_name == "bliva_vicuna":
        model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="vicuna7b", is_eval=True, device=args.device)
    elif args.model_name == "bliva_vicuna_lora":
        model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="vicuna7b", is_eval=True, device=args.device)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    vis_processor = vis_processors["eval"]

    image_paths, questions, ground_truths, ranges, meter_types, environment_conditions = load_data(args.data_path, args.image_prefix)

    absolute_errors = []
    processed_ranges = []
    valid_errors_for_theta = []
    valid_ground_truths_for_theta = []
    correct_count_theta = 0
    nonzero_gt_count_theta = 0
    accuracy_epsilon_correct_count = 0
    processed_samples_count = 0

    # Tracking accuracies per meter_type and environment_condition
    # 扩展结构：同时统计 theta 与 epsilon
    meter_type_accuracy = {}  # {mt: {"theta_correct": int, "epsilon_correct": int, "total": int}}
    environment_condition_accuracy = {}  # {cond: {"theta_correct": int, "epsilon_correct": int, "total": int}}

    # Process in batches
    batch_size = args.batch_size
    num_samples = len(image_paths)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Prepare batch data
        batch_images = []
        batch_questions = []
        batch_img_paths = []
        
        for idx in batch_indices:
            img_path = image_paths[idx]
            try:
                original_image = Image.open(img_path).convert('RGB')
                processed_image = vis_processor(original_image).to(args.device)
                batch_images.append(processed_image)
                batch_questions.append(questions[idx])
                batch_img_paths.append((idx, img_path, original_image))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        if not batch_images:  # Skip if no valid images in batch
            continue
            
        # Stack images for batch processing
        batch_images_tensor = torch.stack(batch_images)
        
        try:
            # Get predictions for the batch
            predictions = eval_batch(batch_images_tensor, batch_questions, model)
            
            # Process each prediction in the batch
            for i, (idx, img_path, original_image) in enumerate(batch_img_paths):
                if i >= len(predictions):  # Safety check
                    continue
                
                pred = predictions[i]
                ground_truth = ground_truths[idx]
                range_value = ranges[idx]
                meter_type = meter_types[idx]
                env_conditions = environment_conditions[idx]

                pred_number = extract_last_number(pred)
                error = abs(pred_number - ground_truth)
                
                absolute_errors.append(error)
                processed_ranges.append(range_value)

                # Accuracy@ε Calculation (per-sample)
                epsilon = 0.01 * range_value
                epsilon_correct = (error <= epsilon)
                if epsilon_correct:
                    accuracy_epsilon_correct_count += 1

                # Theta calculations
                theta_condition = False
                if ground_truth != 0:
                    nonzero_gt_count_theta += 1
                    theta_sample = error / ground_truth
                    theta_condition = (theta_sample < 0.05)

                    if theta_condition:
                        correct_count_theta += 1
                        # Save image with ground truth and prediction
                        filename = f"sample_{idx+1:05d}.jpg"
                        save_path = os.path.join(args.save_path, filename)
                        try:
                            save_image_with_predictions(original_image, save_path, ground_truth, pred_number)
                        except Exception as e:
                            print(f"Error saving image {img_path} to {save_path}: {e}")
                    
                    # Cap theta at 100 for metric calculation
                    capped_theta = min(theta_sample, 100.0)
                    valid_errors_for_theta.append(capped_theta * ground_truth)
                    valid_ground_truths_for_theta.append(ground_truth)

                # Track meter_type accuracy (theta & epsilon)
                if meter_type not in meter_type_accuracy:
                    meter_type_accuracy[meter_type] = {"theta_correct": 0, "epsilon_correct": 0, "total": 0}
                meter_type_accuracy[meter_type]["total"] += 1
                if theta_condition:
                    meter_type_accuracy[meter_type]["theta_correct"] += 1
                if epsilon_correct:
                    meter_type_accuracy[meter_type]["epsilon_correct"] += 1
                
                # Track environment_condition accuracy (theta & epsilon)
                for condition in env_conditions:
                    condition = condition.strip()
                    if condition not in environment_condition_accuracy:
                        environment_condition_accuracy[condition] = {"theta_correct": 0, "epsilon_correct": 0, "total": 0}
                    environment_condition_accuracy[condition]["total"] += 1
                    if theta_condition:
                        environment_condition_accuracy[condition]["theta_correct"] += 1
                    if epsilon_correct:
                        environment_condition_accuracy[condition]["epsilon_correct"] += 1
                
                processed_samples_count += 1  # Update count after processing a sample

        except Exception as e:
            print(f"Error during batch prediction: {e}")

    # Calculate and output final metrics
    final_metrics = calculate_metrics(
        absolute_errors, processed_ranges,
        valid_errors_for_theta, valid_ground_truths_for_theta,
        correct_count_theta, nonzero_gt_count_theta,
        accuracy_epsilon_correct_count, processed_samples_count
    )
    print_metrics(final_metrics, batch=False)

    # Print accuracy per meter_type
    print("\nMeter Type Accuracy:")
    for meter_type, acc in meter_type_accuracy.items():
        theta_acc = acc["theta_correct"] / acc["total"] if acc["total"] > 0 else 0.0
        eps_acc = acc["epsilon_correct"] / acc["total"] if acc["total"] > 0 else 0.0
        print(f"{meter_type}: Theta<0.05={theta_acc:.4f} | Accuracy@ε={eps_acc:.4f}")

    # Print accuracy per environment_condition
    print("\nEnvironment Condition Accuracy:")
    for condition, acc in environment_condition_accuracy.items():
        theta_acc = acc["theta_correct"] / acc["total"] if acc["total"] > 0 else 0.0
        eps_acc = acc["epsilon_correct"] / acc["total"] if acc["total"] > 0 else 0.0
        print(f"{condition}: Theta<0.05={theta_acc:.4f} | Accuracy@ε={eps_acc:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
