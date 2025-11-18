#!/bin/bash

# 设置日志文件路径
LOG_FILE="test.log"
LOSS_FILE="loss.txt"
LOSS_PLOT="loss_plot.png"

# 训练命令
TRAIN_CMD="python evaluate_copy.py \
            --model_name bliva_vicuna \
            --device cuda:2 \
            --data_path /rydata/wengchaoliu/BLIVA/bliva/data/environment_meter_reading_dataset_test_correct_2000_copy_meter_type.json \
            --image_prefix /rydata/wengchaoliu/BLIVA/bliva/data/img_correct \
            --save_path ./withoutkfm/output_images"

# 运行训练并记录输出
echo "开始测试..."
$TRAIN_CMD | tee $LOG_FILE


echo "测试完成！"

