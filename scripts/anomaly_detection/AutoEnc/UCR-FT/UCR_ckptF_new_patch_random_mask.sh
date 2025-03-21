export CUDA_VISIBLE_DEVICES=1

model_name=Timer
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
seq_len=768
d_model=1024
d_ff=2048
e_layers=8
patch_len=96
subset_rand_ratio=1
dataset_dir="./dataset/UCR_Anomaly_FullData"
counter=0

# ergodic datasets
for mask_rate in 0.25 0.5 0.75
do
for file_path in "$dataset_dir"/*
do
data_file=$(basename "$file_path")
((counter++))
echo $counter
python -u run.py \
  --task_name anomaly_detection_AE \
  --is_training 1 \
  --is_finetuning 1 \
  --root_path ./dataset/UCR_Anomaly_FullData \
  --data_path $data_file \
  --model_id UCRA_$data_file \
  --ckpt_path $ckpt_path \
  --model $model_name \
  --data UCRA \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --train_test 0 \
  --batch_size 1024 \
  --subset_rand_ratio $subset_rand_ratio \
  --train_epochs 20 \
  --use_gpu True \
  --use_mask \
  --mask_rate $mask_rate \
  --mask_type patch_random_mask \
  --date_record \


if ((counter>4)); then
  break
fi
done
done