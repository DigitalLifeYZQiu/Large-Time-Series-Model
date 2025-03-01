#export CUDA_VISIBLE_DEVICES=1
model_name=Timer-LOTSA
seq_len=672
label_len=576
#pred_len=96
#output_len=96
patch_len=96
ckpt_path=/data/qiuyunzhong/ts_adaptive_inference/Timer/ckpt/Large_timegpt_d1024_l8_p96_n64_new_full.ckpt

for subset in 500 1000 1500 2000;do
for pred_len in 24 48 96 192;do
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path  $ckpt_path\
  --root_path ../tslib/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --model_id ETTh1_full_shot \
  --model $model_name \
  --features S \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $pred_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 512 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --itr 1 \
  --finetune_epochs 20 \
  --subset $subset
done
done