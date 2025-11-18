#!/bin/bash

# 设置日志文件路径
LOG_FILE="test.log"
LOSS_FILE="loss.txt"
LOSS_PLOT="loss_plot.png"

# 训练命令
TRAIN_CMD="python benchmark.py --device cuda:5 --model_name OpenGVLab/InternVL3-8B --model_path /wangx_nas/WCL/checkpoint/saves/InternVL3-8B-hf/full/train_2025-08-08-14-12-01/checkpoint-16380  --data_path /rydata/wengchaoliu/BLIVA/bliva/data/meter_reading_dataset_test_correct_2000_copy.json --image_prefix /rydata/wengchaoliu/BLIVA/bliva/data/img_correct --save_path ./output_images"

# 运行训练并记录输出
echo "开始测试..."
$TRAIN_CMD | tee $LOG_FILE


echo "测试完成！"
