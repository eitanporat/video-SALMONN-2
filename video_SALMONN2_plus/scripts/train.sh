cd "$(cd $(dirname $0); pwd)/.."

DATASET=/home/user/structured_captions_dataset.json
EVAL_DATASET=/home/user/eval_structured_captions_dataset.json
MODEL=/home/user/Video-Salmonn
MODEL_BASE=/home/user/Video-Salmonn
LR=2e-5
BS=1
ACCUM_STEPS=1
RUN_NAME="train"
DEEPSPEED=2
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
TRAIN_AUDIO=False
TRAIN_QFORMER=False
EPOCH=1
MAX_PIXELS=61250
MIN_PIXELS=784
SAVE_STEPS=500
MIN_FRAMES=10
MAX_FRAMES=256
INTERVAL=0.1
USE_LORA=True
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
LORA_CKPT=tsinghua-ee/video-SALMONN-2_plus_7B
TRAIN_TYPE=sft
NUM_WORKER=0
NO_AUDIO=False

export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=a8ce9d9a7d252ccd3c58a35220ab899e076371f9
export WANDB_ENTITY="lightricks"
export WANDB_PROJECT="txt2img"


mkdir -p output
mkdir -p dataset
mkdir -p eval_dataset

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --eval_dataset) EVAL_DATASET="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --train_llm) TRAIN_LLM=True ;;
        --train_proj) TRAIN_PROJ=True ;;
        --train_enc) TRAIN_ENC=True ;;
        --train_audio) TRAIN_AUDIO=True ;;
        --train_qformer) TRAIN_QFORMER=True ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --epoch) EPOCH="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --use_lora) USE_LORA=True ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --train_type) TRAIN_TYPE="$2"; shift ;;
        --num_worker) NUM_WORKER="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export MAX_STEPS=20000

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12396 \
    qwenvl/train/train_qwen.py \
        --deepspeed scripts/zero${DEEPSPEED}.json \
        --model_name_or_path "$MODEL" \
        --dataset_use $DATASET \
        --eval_dataset_use $EVAL_DATASET \
        --tune_mm_vision $TRAIN_ENC \
        --tune_mm_mlp $TRAIN_PROJ \
        --tune_mm_llm $TRAIN_LLM \
        --bf16 \
        --output_dir output/$RUN_NAME \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --max_pixels $MAX_PIXELS \
        --min_pixels $MIN_PIXELS \
        --max_steps $MAX_STEPS \
	--video_max_frame_pixels $MAX_PIXELS \
        --video_min_frame_pixels $MIN_PIXELS \
        --eval_strategy "steps" \
        --save_strategy "steps" \
        --save_steps $SAVE_STEPS \
        --eval_steps 25 \
        --save_total_limit 5 \
        --learning_rate $LR \
        --weight_decay 0 \
        --warmup_steps 1000 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 131072 \
        --gradient_checkpointing True \
        --dataloader_num_workers $NUM_WORKER \
        --run_name $RUN_NAME \
        --report_to wandb \
        --video_min_frames $MIN_FRAMES \
        --video_max_frames $MAX_FRAMES \
        --base_interval $INTERVAL \
        --model_base $MODEL_BASE \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_ckpt $LORA_CKPT \
        --train_type $TRAIN_TYPE \
        --tune_mm_audio $TRAIN_AUDIO \
        --tune_mm_qformer $TRAIN_QFORMER \
        --no_audio $NO_AUDIO 
