#!/bin/bash

# 设置你要拉取的远程和分支
REMOTE="origin"
BRANCH="main"

# 尝试从远程拉取代码
git pull $REMOTE $BRANCH

# 检查是否有冲突
if [ $? -ne 0 ]; then
    echo "Pull failed due to conflicts. Reverting local changes and trying again."

    # 回退到上一个提交版本
    git reset --hard HEAD~1

    # 再次尝试从远程拉取代码
    git pull $REMOTE $BRANCH

    if [ $? -eq 0 ]; then
        echo "Successfully pulled from remote after reverting."
    else
        echo "Failed to pull from remote after reverting. Please check for issues."
        exit 1
    fi
else
    echo "Successfully pulled from remote."
fi