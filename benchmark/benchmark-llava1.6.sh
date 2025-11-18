
#!/bin/bash
# =========================
# 配置参数
# =========================

# 模型与数据路径
MODEL_NAME="LLaVA-Next/LLaVA-Next-7B"
MODEL_PATH="/rydata/wengchaoliu/LLAVA-NEXT-7B-HF/checkpoint-16368"
DATA_PATH="/rydata/wengchaoliu/BLIVA/bliva/data/meter_reading_dataset_test_correct_2000.json"
IMAGE_PREFIX="/rydata/wengchaoliu/BLIVA/bliva/data/img_correct"
SAVE_PATH="./output_images"

# Python 脚本及运行参数
PY_SCRIPT="benchmark-internvl.py"
DEVICE="cuda:0"
BATCH_SIZE=4
MAX_NEW_TOKENS=256
BACKEND="llava_next"  # internvl 或 qwen

# 日志保存目录
LOG_DIR="./benchmark-logs"
mkdir -p "${LOG_DIR}"  # 创建日志目录（如果不存在）

# 时间戳（格式：YYYYmmdd_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 将 MODEL_NAME 中的斜杠等特殊字符替换为下划线
MODEL_NAME_SAFE=$(echo "${MODEL_NAME}" | tr '/:' '_')

# 动态生成日志文件名（放到 logs 文件夹里）
LOG_FILE="${LOG_DIR}/${MODEL_NAME_SAFE}_${TIMESTAMP}.log"
LOSS_FILE="${LOG_DIR}/${MODEL_NAME_SAFE}_${TIMESTAMP}_loss.txt"
LOSS_PLOT="${LOG_DIR}/${MODEL_NAME_SAFE}_${TIMESTAMP}_loss_plot.png"

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
  2>&1 | tee "${LOG_FILE}"

echo "评估完成！日志已保存到 ${LOG_FILE}"
