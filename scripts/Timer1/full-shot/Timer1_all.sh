export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 所有可用 GPU

# 定义要执行的脚本列表（按顺序执行）
SCRIPTS=(
    "scripts/Timer1/Timer1_ECL.sh"
    "scripts/Timer1/Timer1_ETTh1.sh"
    "scripts/Timer1/Timer1_ETTh2.sh"
    "scripts/Timer1/Timer1_ETTm1.sh"
    "scripts/Timer1/Timer1_ETTm2.sh"
    "scripts/Timer1/Timer1_Exchange.sh"
    "scripts/Timer1/Timer1_Traffic.sh"
    "scripts/Timer1/Timer1_Weather.sh"
)

# 定义每个脚本使用的显卡ID
GPUS=(0 1 2 3 4 5 6 7)
# 最大并行任务数（根据 GPU 显存调整）
max_jobs=8
job_counter=0  # 任务计数器（用于 GPU 轮询分配）

log_dir=logs/Timer1
mkdir -p ${log_dir}  # 创建日志目录

# 遍历并执行每个脚本
for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    gpu="${GPUS[$i]}"

    # 生成对应的日志文件名
    log_file="${script%.*}.log"   # 将脚本的 .sh 后缀替换为 .log
    log_file=(${log_file//// })   # 将脚本按'/'符号切分为数组（切分不同级别路径）
    log_file=${log_file[-1]}      # 取脚本路径中的最后一级路径

    # 清空日志文件
    > "${log_dir}/${log_file}"

    if [[ -x "$script" ]]; then  # 判断脚本是否有执行权限
        echo "========== Running $script on GPU $gpu =========="
        (
          export CUDA_VISIBLE_DEVICES=$gpu
          bash "$script" > "${log_dir}/${log_file}" 2>&1
        ) &
#        echo "========== Finished $script on GPU $gpu =========="
    else
        echo "⚠️  $script is not executable or does not exist."
    fi
    # 更新任务计数器
    ((job_counter++))

    # 控制并行度：如果后台任务数 ≥ max_jobs，等待
    while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
      sleep 5
    done
done

# 等待所有后台任务完成
wait

echo "✅ All scripts finished. Each script's log is saved in its respective .log file."