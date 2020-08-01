#!/usr/bin/env bash
# Created by Chun-Ming Su
SLACK_TOKEN=xoxb-163244810354-721307881920-SsWNhr70KgKfVr1ru8c7e98o
SLACK_ID="sokunmin"
SEND_TO_SLACK=true
DELETE_OLD_CKPT=false
SEED=0
DETERMINISTIC="--deterministic"
WORK_DIR="runs"

# Internal Field Separator
IFS="|"
ARG_COUNT=3
CONFIGS=(
# > "net            | loss | epochs "
# e.g.
    "mobilenetv2    | ce     | 100  |"
    "mobilenetv2    | focal1 | 100  |"
    "mobilenetv2    | focal2 | 100  |"
    "mobilenetv2    | focal3 | 100  |"
)


for CONFIG in "${CONFIGS[@]}" ;
do
    # > remove white spaces and split configs
    CFG="${CONFIG// /}"
    IFS_COUNT=$(tr -dc '|' <<< ${CFG} | wc -c)
    echo "IFS_COUNT: ${IFS_COUNT}"
    if [[ "${IFS_COUNT}" -ne "${ARG_COUNT}" ]]; then
        echo "> Invalid arguments = ${CFG}"
        continue
    fi

    # > parse configs
    set -- "$CFG"
    declare -a CFG=($*)
    MODEL=${CFG[0]}
    LOSS=${CFG[1]}
    EPOCHS=${CFG[2]}
    echo "> MODEL= ${MODEL}"
    echo "> LOSS = ${LOSS}"
    echo "> EPOCHS = ${EPOCHS}"
    echo ""

    # > delete old checkpoints
    if ${DELETE_OLD_CKPT} ; then
        echo "> <<DELETE>> old checkpoints"
        rm -rf "checkpoint/${MODEL}/*.pth"
    fi

    # > run training script
    sleep 5
    python train.py -net ${MODEL} -gpu -loss ${LOSS} -epoch ${EPOCHS}

    sleep 5
    if ${SEND_TO_SLACK} ; then
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} "> ${MODEL}-${LOSS}-${EPOCHS} training is DONE"
        import -window root -delay 1000 screenshot.png
        sleep 5
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} -f screenshot.png
    fi
    echo ""
    sleep 25

    echo ""
    echo "=============================== [end of training/testing] ======================================"
    echo ""
done
