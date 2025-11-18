#!/bin/bash

# 设置日志文件路径
LOG_FILE="fineLr.log"
LOSS_FILE="loss.txt"
LOSS_PLOT="loss_plot.png"

# 设置 wandb 仅本地记录
export WANDB_MODE=offline  # 3 = 仅本地记录，不上传云端
export CUDA_VISIBLE_DEVICES=4
# 训练命令
TRAIN_CMD="torchrun --rdzv_endpoint=localhost:29527  --nnodes=1 --nproc_per_node=1 \
    train.py --cfg-path /rydata/wengchaoliu/BLIVA/train_configs/finetune_bliva_vicuna.yaml"

# 运行训练并记录输出
echo "开始训练..."
$TRAIN_CMD | tee $LOG_FILE

# 提取 loss 信息 (假设 wandb 日志中有 "Loss: 0.1234" 形式的输出)
grep -oE "Loss: [0-9]+\.[0-9]+" $LOG_FILE | awk '{print $2}' > $LOSS_FILE

# 生成 loss 变化图
python - <<EOF
import matplotlib.pyplot as plt

# 读取 loss 数据
try:
    with open("$LOSS_FILE", "r") as f:
        losses = [float(line.strip()) for line in f if line.strip()]

    if losses:
        plt.plot(losses, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.savefig("$LOSS_PLOT")
        print(f"Loss 曲线已保存为 {LOSS_PLOT}")
    else:
        print("未找到 loss 数据，无法绘制曲线。")
except Exception as e:
    print(f"绘制 loss 过程出错: {e}")
EOF

echo "训练完成，loss 曲线已生成！"
