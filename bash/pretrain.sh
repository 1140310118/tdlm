while getopts ':c:l:p:e:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        l)
        lr="$OPTARG" ;;
        p)
        position_embedding_type="$OPTARG" ;;
        e)
        encoder_layer="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done




gradient_clip_val=1
warmup_steps=1000
weight_decay=0.01
precision=16
max_seq_length=64
batch_size=128
data_dir=/home/zhangyice/2022/enwiki/


# --distributed_backend=ddp \

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python tdlm_pretrain.py \
  --accelerator='ddp' \
  --gpus=1 \
  --precision=${precision} \
  --data_dir ${data_dir} \
  --model_name_or_path bert-base-uncased \
  --output_dir ../output/tdlm/pretrain/enwiki/8_512_${encoder_layer}_${position_embedding_type}_lr=${lr}/ \
  --learning_rate ${lr}e-5 \
  --train_batch_size ${batch_size} \
  --eval_batch_size ${batch_size} \
  --seed 42 \
  --warmup_steps ${warmup_steps} \
  --lr_scheduler linear \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --max_steps 20_000 \
  --val_check_interval 1_000 \
  --accumulate_grad_batches 16 \
  --num_workers 8 \
  --position_embedding_type ${position_embedding_type} \
  --encoder_layer ${encoder_layer}


# val_check_interval // accumulate_grad_batches
# max_steps // 1
