#!/bin/bash
REMOTE="origin"
BRANCH="version1"
git remote add origin git@github.com:pqy000/vat.git
git add .
git commit -m "new update"
git push $REMOTE $BRANCH --force