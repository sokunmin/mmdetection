#!/usr/bin/env bash
# Created by Chun-Ming Su
CONFIG_ROOT="configs"

# Internal Field Separator
IFS="|"
ARG_COUNT=5
CONFIGS=(
# " config_dir | config_py                  | checkpoint           | multitask   | eval_metrics    | ckpt_module"
  " centernet  | centernet_dla34_3x_pose.py | centerpose_dla34.pth | --multitask | bbox, keypoints | centerpose"
)
NUM_GPUS=1
PYTHON=${PYTHON:-"python"}
PORT=${PORT:-29500}

for CONFIG in "${CONFIGS[@]}" ;
do
    # > remove white spaces and split configs
    CFG="${CONFIG// /}"
    IFS_COUNT=$(tr -dc '|' <<< ${CFG} | wc -c)
    if [[ "${IFS_COUNT}" -ne "${ARG_COUNT}" ]]; then
        echo "> Invalid arguments = ${CFG}"
        continueq
    fi
    # > parse configs
    set -- "$CFG"
    declare -a CFG=($*)

    CONFIG_DIR="${CONFIG_ROOT}/${CFG[0]}"
    CONFIG_NAME=$(basename ${CFG[1]} .py)
    CONFIG_PY="${CFG[1]}"
    READ_CONFIG_PY=${CONFIG_DIR}/${CONFIG_PY}
    WORK_DIR="work_dirs/${CONFIG_NAME}"
    MULTITASK=${CFG[3]}
    echo "> READ_CONFIG_PY = ${READ_CONFIG_PY}"
    echo "> CONFIG_DIR = ${CONFIG_DIR}"
    echo "> WORK_DIR = ${WORK_DIR}"
    echo "> CONFIG_NAME = ${CONFIG_NAME}"
    echo "> CONFIG_PY = ${CONFIG_PY}"
    IFS=',' read -a EVAL_METRICS <<< "${CFG[4]}"
    READ_CKPT=${CFG[2]}
    if [[ -z "${READ_CKPT}" ]] ; then
        READ_CKPT="latest.pth"
    fi
    CHECKPOINT="${WORK_DIR}/${READ_CKPT}"
    CONFIG_NAME=${CONFIG_PY%.py}
    LOG_TXT="${CONFIG_NAME}-${READ_CKPT}.txt"
    CKPT_MODULE=${CFG[5]}
    echo "> READ_CHECKPOINT = ${CHECKPOINT}"
    echo "> MULTITASK = ${MULTITASK}"
    echo "> EVAL_METRICS = ${EVAL_METRICS[@]}"
    if [[ -z "${CKPT_MODULE}" ]] ; then
        CKPT_MODULE="mmcv"
    fi
    echo "> CKPT_MODULE = ${CKPT_MODULE}"
    sleep 5

    if [[ ! -d "$WORK_DIR" ]]; then
        echo "$WORK_DIR not found"
        exit 0
    fi

    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "$CHECKPOINT not found!"
        exit 0
    fi

    OUTPUT_LOG_TXT=${WORK_DIR}/${LOG_TXT}
    rm -rf ${OUTPUT_LOG_TXT}

    if [[ ${NUM_GPUS} -eq 1 ]] ; then
        echo "> Evaluating with single GPU ..."
        echo "python tools/test.py ${READ_CONFIG_PY} ${CHECKPOINT} --ckpt-module ${CKPT_MODULE} ${MULTITASK} --out output.pkl --eval ${EVAL_METRICS[@]} > ${OUTPUT_LOG_TXT}"
        echo ""
        $PYTHON tools/test.py ${READ_CONFIG_PY} ${CHECKPOINT} --ckpt-module ${CKPT_MODULE} ${MULTITASK} --out output.pkl --eval ${EVAL_METRICS[@]} > ${OUTPUT_LOG_TXT}
    else
        echo "> Evaluating with ${NUM_GPUS} GPUs ..."
        echo "bash tools/dist_test.sh ${READ_CONFIG_PY} ${CHECKPOINT} ${NUM_GPUS} --ckpt-module ${CKPT_MODULE} ${MULTITASK} --out result.pkl --eval ${EVAL_METRICS[@]} > ${OUTPUT_LOG_TXT}"
        echo ""
        bash tools/dist_test.sh ${READ_CONFIG_PY} ${CHECKPOINT} ${NUM_GPUS} --ckpt-module ${CKPT_MODULE} ${MULTITASK} --out result.pkl --eval ${EVAL_METRICS[@]} > ${OUTPUT_LOG_TXT}
    fi
    echo ""
    echo "> Output results to <<${OUTPUT_LOG_TXT}>>"
    tail -n 15 ${OUTPUT_LOG_TXT}
    sleep 5

    if [[ "${SLACK_TOKEN}" && "${SLACK_ID}" ]] ; then
        import -window root -delay 1000 screenshot.png
        sleep 5
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} -f screenshot.png
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} -f ${OUTPUT_LOG_TXT}
        echo "> <<SEND>> log to <<${SLACK_ID}>>"
    fi
    sleep 25

    echo ""
    echo "=============================== [end of testing] ======================================"
    echo ""
done
