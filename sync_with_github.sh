#!/bin/bash

REMOTE="origin"
BRANCH="version1"
LOG_DIR="/data/qingyi/log"
UNTRACKED_LOG_DIR="/data/qingyi/untracked_logs"

# 确保日志目录和未跟踪文件目录存在
mkdir -p $LOG_DIR
mkdir -p $UNTRACKED_LOG_DIR

function move_conflicted_files {
    git diff --name-only --diff-filter=U | while read file; do
        if [ -f "$file" ]; then
            echo "Moving conflicted file $file to $LOG_DIR"
            mv "$file" "$LOG_DIR/"
        fi
    done
}

function move_untracked_files {
    git ls-files --others --exclude-standard | while read file; do
        if [ -f "$file" ]; then
            echo "Moving untracked file $file to $UNTRACKED_LOG_DIR"
            mv "$file" "$UNTRACKED_LOG_DIR/"
        fi
    done
}

move_untracked_files

git pull $REMOTE $BRANCH

if [ $? -ne 0 ]; then
    echo "Pull failed due to conflicts. Attempting to resolve."

    move_conflicted_files

    git reset --hard HEAD@{1}

    git pull $REMOTE $BRANCH

    while [ $? -ne 0 ]; do
        echo "Pull failed again. Moving conflicted files and trying to revert further."

        move_conflicted_files

        git reset --hard HEAD@{1}

        git pull $REMOTE $BRANCH
    done

    if [ $? -eq 0 ]; then
        echo "Successfully pulled from remote after reverting."
    else
        echo "Failed to pull from remote after multiple attempts. Please check for issues."
        exit 1
    fi
else
    echo "Successfully pulled from remote."
fi