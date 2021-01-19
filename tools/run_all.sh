#!/usr/bin/env bash
# Created by Chun-Ming Su
SLACK_TOKEN="xoxb-163244810354-721307881920-0wek05R5ddIpWnlvKEbtPONb"
SLACK_ID="sokunmin"
SEND_TO_SLACK=true
CONFIG_ROOT="configs"
DELETE_OLD_CKPT=false
NO_VALIDATE=""#"--no-validate"

# Internal Field Separator
IFS="|"
ARG_COUNT=6
CONFIGS=(
# mode: ([train, traineval], test)
# > "mode  | config_dir  | config_py                 | resume_ckpt  | eval_metrics    | test_ckpt | deterministic"
# e.g.
    "train | centernet   | centernet_r18_3x_pose.py  |              | bbox, keypoints | --deterministic |"
)
TRAIN_GPUS=2
TEST_GPUS=1
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
    IFS=',' read -a EXEC_MODE <<< "${CFG[0]}"
    CONFIG_DIR="${CONFIG_ROOT}/${CFG[1]}"
    CONFIG_NAME=$(basename ${CFG[2]} .py)
    CONFIG_PY="${CFG[2]}"
    READ_CONFIG_PY=${CONFIG_DIR}/${CONFIG_PY}
    WORK_DIR="work_dirs/${CONFIG_NAME}"
    SEED=0
#    SEED=$RANDOM
    echo "> SEED = ${SEED}"
    echo "> READ_CONFIG_PY = ${READ_CONFIG_PY}"
    echo "> EXEC_MODE_1 = ${EXEC_MODE[0]}"
    echo "> EXEC_MODE_2 = ${EXEC_MODE[1]}"
    echo "> CONFIG_DIR = ${CONFIG_DIR}"
    echo "> WORK_DIR = ${WORK_DIR}"
    echo "> CONFIG_NAME = ${CONFIG_NAME}"
    echo "> CONFIG_PY = ${CONFIG_PY}"
    echo "> DETERMINISTIC = ${CFG[6]}"
    echo ""

    if [[ "${EXEC_MODE[0]}" == "train" ]] ; then
        echo "-------- [TRAINING] --------"
        RESUME_CKPT=${CFG[3]}
        DETERMINISTIC=${CFG[6]}
        # > create work_dir and copy config to the folder
#        mkdir -p ${WORK_DIR}
#        cp -f ${CONFIG_DIR}/${CONFIG_PY} ${WORK_DIR}/${CONFIG_PY}

        # > delete old checkpoints
        if ${DELETE_OLD_CKPT} ; then
            echo "> <<DELETE>> old checkpoints"
            rm -rf ${WORK_DIR}/*.pth
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
        echo "python tools/dist_train.sh ${READ_CONFIG_PY} ${TRAIN_GPUS} --seed ${SEED} ${DETERMINISTIC} ${RESUME_FROM}"
        bash tools/dist_train.sh ${READ_CONFIG_PY} ${TRAIN_GPUS} --seed ${SEED} ${DETERMINISTIC} ${RESUME_FROM} ${NO_VALIDATE}

        sleep 5
        if ${SEND_TO_SLACK} ; then
            slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} "> DONE <<TRAINING>> with <<${CONFIG_PY}>>"
        fi
        echo ""
        sleep 25
    fi

    if [[ "${EXEC_MODE[0]}" == "test" || "${EXEC_MODE[1]}" == "test" ]] ; then
        echo "-------- [TESTING] --------"
        IFS=',' read -a EVAL_METRICS <<< "${CFG[4]}"
        READ_CKPT=${CFG[5]}
        if [[ -z "${READ_CKPT}" ]] ; then
            READ_CKPT="latest.pth"
        fi
        CHECKPOINT="${WORK_DIR}/${READ_CKPT}"
        CONFIG_NAME=${CONFIG_PY%.py}
        LOG_TXT="${CONFIG_NAME}-${READ_CKPT}.txt"
        echo "> EVAL_METRICS = ${EVAL_METRICS[@]}"
        echo "> READ_CHECKPOINT = ${CHECKPOINT}"
        sleep 5

        if [[ ${TEST_GPUS} -eq 1 ]] ; then
            echo "> <<EVALUATE>> w/ single GPU"
            $PYTHON tools/test.py ${WORK_DIR}/${CONFIG_PY} ${CHECKPOINT} --out output.pkl --eval ${EVAL_METRICS[@]} >> ${WORK_DIR}/${LOG_TXT}
        else
            echo "> <<EVALUATE>> w/ ${TEST_GPUS} GPUs"
            bash tools/dist_test.sh ${WORK_DIR}/${CONFIG_PY} ${CHECKPOINT} ${TEST_GPUS} --out result.pkl --eval ${EVAL_METRICS[@]} >> ${WORK_DIR}/${LOG_TXT}
        fi
        echo "> <<OUTPUT>> results to <<${LOG_TXT}>>"
        tail -n 15 ${WORK_DIR}/${LOG_TXT}
        sleep 5

        if ${SEND_TO_SLACK} ; then
            import -window root -delay 1000 screenshot.png
            sleep 5
            slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} -f screenshot.png
            slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} -f ${WORK_DIR}/${LOG_TXT}
            echo "> <<SEND>> log to <<${SLACK_ID}>>"
        fi
        sleep 25
    fi
    echo ""
    echo "=============================== [end of training/testing] ======================================"
    echo ""
done
