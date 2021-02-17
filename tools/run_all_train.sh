#!/usr/bin/env bash
# Created by Chun-Ming Su
CONFIG_ROOT="configs"

# Internal Field Separator
IFS="|"
ARG_COUNT=6
CONFIGS=(
# mode: (train, test), delete_old: true or false
# " seed | config_dir | config_py              | checkpoint  | deterministic   | no_valididate | delete_old"
  "      | ttfnet     | ttfnet_r18_10x_pose.py |             | --deterministic |               | true    "
)
NUM_GPUS=2
PYTHON=${PYTHON:-"python"}
PORT=${PORT:-29500}

for CONFIG in "${CONFIGS[@]}" ;
do
    # > remove white spaces and split configs
    CFG="${CONFIG// /}"
    IFS_COUNT=$(tr -dc '|' <<< ${CFG} | wc -c)
    if [[ "${IFS_COUNT}" -ne "${ARG_COUNT}" ]]; then
        echo "> Invalid arguments = ${CFG}"
        continue
    fi
    # > parse configs
    set -- "$CFG"
    declare -a CFG=($*)

    SEED="${CFG[0]}"
    if [[ -z "$SEED" ]] ; then
        SEED=$RANDOM
    fi
    CONFIG_DIR="${CONFIG_ROOT}/${CFG[1]}"
    CONFIG_NAME=$(basename ${CFG[2]} .py)
    CONFIG_PY="${CFG[2]}"
    READ_CONFIG_PY=${CONFIG_DIR}/${CONFIG_PY}
    WORK_DIR="work_dirs/${CONFIG_NAME}"
    RESUME_CKPT=${CFG[3]}
    DETERMINISTIC=${CFG[4]}
    NO_VALID=${CFG[5]}
    DELETE_OLD=${CFG[6]}
    echo "> SEED = ${SEED}"
    echo "> READ_CONFIG_PY = ${READ_CONFIG_PY}"
    echo "> CONFIG_DIR = ${CONFIG_DIR}"
    echo "> WORK_DIR = ${WORK_DIR}"
    echo "> CONFIG_NAME = ${CONFIG_NAME}"
    echo "> CONFIG_PY = ${CONFIG_PY}"
    echo "> DETERMINISTIC = ${DETERMINISTIC}"
    echo "> NO_VALID = ${NO_VALID}"
    echo "> DELETE_OLD = ${DELETE_OLD}"

    sleep 10

    # > delete old checkpoints
    if [[ "${DELETE_OLD}" == "true" ]] ; then
        echo "> <<DELETE>> old checkpoints"
        rm -rf ${WORK_DIR}/*.pth
        rm -rf ${WORK_DIR}/tf_logs
    fi

    # > append resume checkpoint argument
    if [[ -z "$RESUME_CKPT" ]] ; then
        RESUME_FROM=""
        echo "> RESUME_CHECKPOINT = None"
    else
        RESUME_FROM="--resume-from ${WORK_DIR}/${RESUME_CKPT}"
        echo "> RESUME_CHECKPOINT = ${RESUME_CKPT}"
    fi

    # > run training script
    sleep 5
    echo "python tools/dist_train.sh ${READ_CONFIG_PY} ${NUM_GPUS} --seed ${SEED} ${DETERMINISTIC} ${RESUME_FROM}"
    bash tools/dist_train.sh ${READ_CONFIG_PY} ${NUM_GPUS} --seed ${SEED} ${DETERMINISTIC} ${RESUME_FROM} ${NO_VALIDATE}

    sleep 5
    if [[ "${SLACK_TOKEN}" && "${SLACK_ID}" ]] ; then
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} "> DONE <<TRAINING>> with <<${CONFIG_PY}>>"
    fi
    echo ""
    sleep 25

    echo ""
    echo "=============================== [end of training] ======================================"
    echo ""
done
