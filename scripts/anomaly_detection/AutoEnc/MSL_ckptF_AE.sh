export CUDA_VISIBLE_DEVICES=1

model_name=Timer
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
#ckpt_path=checkpoints/timer-base.ckpt
seq_len=768
d_model=1024
d_ff=2048
e_layers=8
patch_len=96
subset_rand_ratio=0.1

python -u run.py \
  --task_name anomaly_detection_AE \
  --is_training 1 \
  --is_finetuning 1 \
  --root_path ../tslib/dataset/MSL \
  --model_id MSL_$data_file \
  --ckpt_path $ckpt_path \
  --model $model_name \
  --data MSL \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --train_test 0 \
  --batch_size 64 \
  --subset_rand_ratio $subset_rand_ratio \
  --train_epochs 10 \
  --date_record