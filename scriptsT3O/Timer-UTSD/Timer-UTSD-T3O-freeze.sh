export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_name=Timer
seq_len=672
label_len=576
#pred_len=96
#output_len=96
patch_len=96
ckpt_path=../CKPT/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt

# 参数列表
lr_list=(1e-4 5e-5 3e-5 1e-5 5e-6 3e-6 1e-6)
subset_list=(500)
pred_len_list=(24 48 96 192)

# 最大并行任务数（根据 GPU 显存调整）
max_jobs=8
job_counter=0  # 任务计数器（用于 GPU 轮询分配）
GPUS=(0 1 2 3 4 5 6 7)

log_dir=logs/T3O-train-scaler/Timer-UTSD-freeze
mkdir -p ${log_dir}

for lr in "${lr_list[@]}";do
  for subset in "${subset_list[@]}"; do
    for pred_len in "${pred_len_list[@]}";do
      gpu="${GPUS[$((job_counter % ${#GPUS[@]}))]}"

      # 生成对应的日志文件名
      log_file="lr${lr}_subset${subset}_predlen${pred_len}_gpu${gpu_id}.log"


      # 清空日志文件
      > "${log_dir}/${log_file}"

      echo "Initialize task: lr:${lr}, subset:${subset}, pred_len:${pred_len} log:${log_dir}/${log_file} ==== gpu:${gpu}"
      (
        export CUDA_VISIBLE_DEVICES=$gpu
        python -u run.py \
          --task_name forecast \
          --is_training 1 \
          --is_finetuning 1 \
          --seed 1 \
          --ckpt_path  $ckpt_path\
          --root_path ../DATA/electricity/ \
          --data_path electricity.csv \
          --data T3O \
          --model_id T3O_few_shot \
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
          --batch_size 2048 \
          --learning_rate $lr \
          --num_workers 10 \
          --patch_len $patch_len \
          --train_test 0 \
          --itr 1 \
          --gpu 0 \
          --finetune_epochs 20 \
          --subset $subset \
          --freeze_decoder \
          --date_record \
          > "${log_dir}/${log_file}" 2>&1
      ) &

      # 更新任务计数器
      ((job_counter++))

      # 控制并行度：如果后台任务数 ≥ max_jobs，等待
      while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
        sleep 5
      done

    done
  done
done