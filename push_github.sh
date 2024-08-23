#!/bin/bash
REMOTE="origin"
BRANCH="version1"
git add .
git commit -m "new update"
git push $REMOTE $BRANCH --force