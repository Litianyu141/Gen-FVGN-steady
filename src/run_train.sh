#!/bin/bash

# 获取脚本所在的目录
BASE_TRAIN_DIR=$(dirname "$0")

LOG_FILE="$BASE_TRAIN_DIR/train_log.txt"
exec > $BASE_TRAIN_DIR/train_handler.txt 2>&1

# training set
on_gpu=0
batch_size=2
dataset_size=100
lr=0.001
load_date=""
load_index=""
ncn_smooth=false
conserved_form=false
prevent_reverse_flow=false
integrator="imex"
pressure_open_bc=1
average_sequence_length=500
equation_state="unsteady_only"
residual_tolerance=1e-7
max_inner_step=50
loss_cont=10
lr_scheduler="fix"
dataset_type="tf"
dataset_dir="/mnt/d/dataset-work2/Unsteady-pipe-cylinder-mesh-at-middle-center/converted_dataset/tf"
# "/mnt/c/Users/DOOMDUKE2-lab/Desktop/Dataset-wallfarfield/converted_dataset/tf"
# "/mnt/c/Users/DOOMDUKE2-lab/Desktop/Dataset-GEP-FVGN-steady-with-poly-without-farfield/converted_dataset/tf"
# "/mnt/d/dataset-work2/Dataset-GEP-FVGN-steady-with-poly-without-farfield/converted_dataset/tf"
# "/lvm_data/litianyu/dataset/FVNN-work2/Dataset-GEP-FVGN-farfield-only/converted_dataset/tf"
# "/mnt/d/dataset-work2/Dataset-GEP-FVGN-unsteady-cylinder-and-airfoil/converted_dataset/tf"
# case parameters
ff_dt=0.005
pf_dt=0.01
cf_dt=0.01


echo "Starting training..."

# 构造基本命令
CMD="python -u $BASE_TRAIN_DIR/train.py --on_gpu=$on_gpu --batch_size=$batch_size --lr=$lr --dataset_size=$dataset_size --ncn_smooth=$ncn_smooth --integrator=$integrator --average_sequence_lengt=$average_sequence_length --dataset_type=$dataset_type --dataset_dir=$dataset_dir --ff_dt=$ff_dt --equation_state=$equation_state --cf_dt=$cf_dt --pf_dt=$pf_dt --loss_cont=$loss_cont --lr_scheduler=$lr_scheduler --conserved_form=$conserved_form --prevent_reverse_flow=$prevent_reverse_flow --residual_tolerance=$residual_tolerance --max_inner_step=$max_inner_step --pressure_open_bc=$pressure_open_bc"

# 根据load_date和load_index的值添加对应的参数
if [ -n "$load_date" ]; then
    CMD="$CMD --load_date=$load_date"
fi

if [ -n "$load_index" ]; then
    CMD="$CMD --load_index=$load_index"
fi

# 运行CMD并将输出保存到 LOG_FILE
$CMD > $LOG_FILE 2>&1

echo "Starting detecting OOM error..."

# 检查是否有 OOM 错误
if grep -q "CUDA out of memory." $LOG_FILE; then
    echo "Detected OOM error, preparing to restart..."

    # 从日志中提取路径
    EXTRACTED_PATH=$(grep -o "saved tecplot file at .*" $LOG_FILE | tail -1 | sed -n 's|saved tecplot file at \(.*\)|\1|p')
    
    # 提取日期
    load_date=$(echo "$EXTRACTED_PATH" | awk -F'/' '{print $(NF-2)}')
    
    # 构建STATE_DIR路径
    STATE_DIR=$(echo "$EXTRACTED_PATH" | awk -F'/' 'BEGIN{OFS=FS} {$NF=""; $(NF-1)="states"; print $0}')
    
    # 检索最大的epoch文件
    MAX_EPOCH_FILE=$(ls -v "$STATE_DIR"*.state | tail -1)
    load_index=$(basename "$MAX_EPOCH_FILE" .state)

    echo "Loading from date: $load_date and epoch: $load_index"
    
    continue
else
    echo "Not detected Training OOM error"
    exit 0
fi
