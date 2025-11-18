#!/bin/bash

# =========================
# 配置参数
# =========================
LOG_FILE="eval.log"             # 标准输出和日志保存路径
LOSS_FILE="loss.txt"            # 如果评估过程输出 loss，这里可以单独保存
LOSS_PLOT="loss_plot.png"       # 可选：后续画 loss 曲线的图片

# 模型与数据路径
MODEL_NAME="OpenGVLab/InternVL3-8B-hf"
MODEL_PATH="/wangx_nas/WCL/checkpoint/saves/InternVL3-8B-hf/full/train_2025-08-08-14-12-01/checkpoint-16380"
DATA_PATH="/rydata/wengchaoliu/BLIVA/bliva/data/meter_reading_dataset_test_correct_2000_copy.json"
IMAGE_PREFIX="/rydata/wengchaoliu/BLIVA/bliva/data/img_correct"
SAVE_PATH="./output_images"

# Python 脚本及运行参数
PY_SCRIPT="benchmark-internvl.py"
DEVICE="cuda:5"
BATCH_SIZE=4
MAX_NEW_TOKENS=256
BACKEND="internvl"  # internvl 或 qwen

# =========================
# 执行评估
# =========================
echo "开始评估 ${MODEL_NAME} ..."
python ${PY_SCRIPT} \
  --backend ${BACKEND} \
  --device ${DEVICE} \
  --model_name ${MODEL_NAME} \
  --model_path ${MODEL_PATH} \
  --data_path ${DATA_PATH} \
  --image_prefix ${IMAGE_PREFIX} \
  --save_path ${SAVE_PATH} \
  --batch_size ${BATCH_SIZE} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  2>&1 | tee ${LOG_FILE}

echo "评估完成！日志已保存到 ${LOG_FILE}"
