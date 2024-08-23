#!/bin/bash

# 参数检查
#if [ "$#" -ne 4 ]; then
#    echo "Usage: $0 <dataset_name> <gpu_list> <label_ratio_list>"
#    echo "Example: $0 InsectWingbeatSound '3 4 5' '0.4 0.2 0.1'"
#    bash run.sh XJTU '3 4 5' '0.4 0.2 0.1'
#    bash run.sh MFPT '1 2' '0.2 0.1'
#    bash run.sh EpilepticSeizure '3' '0.4'
# bash run.sh CricketX '1 2' '0.2 0.4'
# bash run.sh InsectWingbeatSound '3 4 5 6' '0.1 0.2 0.4 1.0'
#    exit 1
#fi

# 获取参数
DATASET_NAME=$1
GPU_LIST=($2)  # 将GPU列表转换为数组
LABEL_RATIO_LIST=($3)  # 将label_ratio列表转换为数组

# 检查 GPU 数量与 label_ratio 数量是否匹配
if [ "${#GPU_LIST[@]}" -ne "${#LABEL_RATIO_LIST[@]}" ]; then
    echo "Error: The number of GPUs and label ratios must match."
    exit 1
fi

# 检查是否存在log目录，如果不存在则创建
if [ ! -d "log" ]; then
  mkdir log
fi

# 启动实验
for i in "${!GPU_LIST[@]}"; do
  GPU=${GPU_LIST[$i]}
  LABEL_RATIO=${LABEL_RATIO_LIST[$i]}
  LOG_FILE="log/${DATASET_NAME}_gpu${GPU}.log"

  nohup python mainOurs.py --dataset_name=${DATASET_NAME} --gpu=${GPU} --label_ratio=${LABEL_RATIO} > ${LOG_FILE} 2>&1 &

  echo "Started process on GPU ${GPU} with label ratio ${LABEL_RATIO}. Check ${LOG_FILE} for output."
done