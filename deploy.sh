#!/bin/bash

# 检查是否提供了目标节点参数
if [ -z "$1" ]; then
  echo "Usage: $0 <target_node>"
  exit 1
fi

# 目标节点
TARGET_NODE=$1

SOURCE_DIR1="/data/qingyi/semitime/"
SOURCE_DIR2="/data/qingyi/miniconda3/"

TARGET_DIR1="/data/qingyi/semitime/"
TARGET_DIR2="/data/qingyi/miniconda3/"

scp -r $SOURCE_DIR1 qingyi@$TARGET_NODE:$TARGET_DIR1
scp -r $SOURCE_DIR2 qingyi@$TARGET_NODE:$TARGET_DIR2

ssh qingyi@$TARGET_NODE << 'ENDSSH'
if [ -d "/data/qingyi/miniconda3" ]; then
  echo "miniconda3 directory exists on target node. Deleting it..."
  rm -rf /data/qingyi/miniconda3
fi

echo "Creating symbolic link for miniconda3..."
ln -s /data/qingyi/miniconda3 /home/qingyi/

export WANDB_API_KEY="78aa9312108a04888f4d78df9a5871282832b0ba"
wandb login --relogin $WANDB_API_KEY

ENDSSH

git clone
cd vat
git checkout version1
ln -s /data/qingyi/semitime/datasets/datasets/ ./

echo "Operations completed successfully."
scp -r /home/qingyi/mini