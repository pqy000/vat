#!/bin/bash

DATASETS=$1
shift
# bash new_deploy.sh "Heartbeat NATOPS SelfRegulationSCP2" g19
TARGET_PATH=""

for NODE in "$@"
do
    for DATASET in $DATASETS
    do
        echo "Copying $DATASET to $NODE:$TARGET_PATH"
        scp -r "$TARGET_PATH$DATASET" "qingyi@$NODE:$TARGET_PATH"
        if [ $? -eq 0 ]; then
            echo "Successfully copied $DATASET to $NODE"
        else
            echo "Failed to copy $DATASET to $NODE"
        fi
    done
done