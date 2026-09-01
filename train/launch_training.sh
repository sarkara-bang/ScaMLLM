#!/bin/bash

BASE_DIR="/root/autodl-tmp/multimodel"
MODEL_PATH="${BASE_DIR}/models/DrugAssist-7B"
IMAGEMOL_CHECKPOINT="${BASE_DIR}/models/ImageMol/ImageMol.pth.tar"
TRAIN_DATA="${BASE_DIR}/data/train.json"
VAL_DATA="${BASE_DIR}/data/val.json"
IMAGE_DIR="${BASE_DIR}/datasets"
SCAFFOLD_MAPPING="${BASE_DIR}/data/scaffold_mapping.json"
OUTPUT_DIR="${BASE_DIR}/models/output"

NUM_IMAGE_TOKENS=4
PROJECTION_GAIN=1.0
USE_CROSS_ATTENTION=true
IMAGE_FEATURE_WEIGHT=1.5

USE_SCAFFOLD_LOSS=true
SCAFFOLD_LOSS_WEIGHT=0.3
SUBSTRUCTURE_WEIGHT=0.7
SCAFFOLD_LOSS_FREQ=10
START_STEP=1000

BATCH_SIZE=4
GRADIENT_ACCUM_STEPS=4
LEARNING_RATE=2e-4
NUM_EPOCHS=4
SAVE_STEPS=1000
EVAL_STEPS=1000
LOGGING_STEPS=10

echo "Architecture:"
echo "  - Image tokens: ${NUM_IMAGE_TOKENS}"
echo "  - Projection gain: ${PROJECTION_GAIN}"
echo "  - Cross-attention: ${USE_CROSS_ATTENTION}"
echo "  - Image weight: ${IMAGE_FEATURE_WEIGHT}"
echo "
echo "  - Enabled: ${USE_SCAFFOLD_LOSS}"
echo "  - Weight: ${SCAFFOLD_LOSS_WEIGHT}"
echo "  - Substructure weight: ${SUBSTRUCTURE_WEIGHT}"
echo "  - Frequency: every ${SCAFFOLD_LOSS_FREQ} steps"
echo "  - Start step: ${START_STEP}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "========================================"
echo ""

mkdir -p ${OUTPUT_DIR}

cd ${BASE_DIR}
python train/run_training_reinforce.py \
    --model_name_or_path ${MODEL_PATH} \
    --tokenizer_name_or_path ${MODEL_PATH} \
    --imagemol_checkpoint_path ${IMAGEMOL_CHECKPOINT} \
    --train_file ${TRAIN_DATA} \
    --validation_file ${VAL_DATA} \
    --image_dir ${IMAGE_DIR} \
    --scaffold_mapping_file ${SCAFFOLD_MAPPING} \
    --num_image_tokens ${NUM_IMAGE_TOKENS} \
    --projection_gain ${PROJECTION_GAIN} \
    --use_cross_attention ${USE_CROSS_ATTENTION} \
    --image_feature_weight ${IMAGE_FEATURE_WEIGHT} \
    --use_scaffold_loss ${USE_SCAFFOLD_LOSS} \
    --scaffold_loss_weight ${SCAFFOLD_LOSS_WEIGHT} \
    --substructure_weight ${SUBSTRUCTURE_WEIGHT} \
    --scaffold_loss_freq ${SCAFFOLD_LOSS_FREQ} \
    --scaffold_loss_start_step ${SCAFFOLD_LOSS_START_STEP} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --load_best_model_at_end true \
    --metric_for_best_model loss \
    --greater_is_better false \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --tf32 true \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --report_to tensorboard \
    --logging_dir ${OUTPUT_DIR}/logs \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "========================================"
echo "Model saved to: ${OUTPUT_DIR}"
echo "========================================"
