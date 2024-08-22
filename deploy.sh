#!/bin/bash

# 检查是否提供了目标节点参数
if [ -z "$1" ]; then
  echo "Usage: $0 <target_node>"
  exit 1
fi

# 目标节点
TARGET_NODE=$1

# 定义要拷贝的目录路径
SOURCE_DIR1="/data/qingyi/semitime/"
SOURCE_DIR2="/data/qingyi/miniconda3/"

# 目标路径（假设目标路径与源路径相同）
TARGET_DIR1="/data/qingyi/semitime/"
TARGET_DIR2="/data/qingyi/miniconda3/"

# 执行 SCP 拷贝
scp -r $SOURCE_DIR1 qingyi@$TARGET_NODE:$TARGET_DIR1
scp -r $SOURCE_DIR2 qingyi@$TARGET_NODE:$TARGET_DIR2

# 检查并处理目标节点上的 miniconda3 目录
ssh qingyi@$TARGET_NODE << 'ENDSSH'
if [ -d "/data/qingyi/miniconda3" ]; then
  echo "miniconda3 directory exists on target node. Deleting it..."
  rm -rf /data/qingyi/miniconda3
fi

echo "Creating symbolic link for miniconda3..."
ln -s /data/qingyi/miniconda3 /home/qingyi/
ENDSSH

# 克隆 Git 仓库并切换分支
git clone git@github.com:pqy000/vat.git
cd vat
git checkout version1

echo "Operations completed successfully."