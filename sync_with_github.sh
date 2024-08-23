#!/bin/bash

REMOTE="origin"
BRANCH="version1"

git pull $REMOTE $BRANCH

if [ $? -ne 0 ]; then
    echo "Pull failed due to conflicts. Reverting local changes and trying again."

    git reset --hard HEAD~1

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