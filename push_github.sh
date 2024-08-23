#!/bin/bash

REMOTE="origin"
BRANCH="version1"

git add .

if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    git commit -m "new update"
fi

git push $REMOTE $BRANCH --force

if [ $? -eq 0 ]; then
    echo "Successfully force pushed to $REMOTE/$BRANCH."
else
    echo "Failed to force push to $REMOTE/$BRANCH. Please check for issues."
    exit 1
fi