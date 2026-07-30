#### paired
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path> \
#  --save_dir <save directory>

# #### unpaired
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path> \
#  --unpair \
#  --save_dir <save directory>

# #### paired repaint
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path>t \
#  --repaint \
#  --save_dir <save directory>

# #### unpaired repaint
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path> \
#  --unpair \
#  --repaint \
#  --save_dir <save directory>

python inference.py \
  --config_path ./configs/VITONHD.yaml \
  --batch_size 4 \
  --model_load_path logs/20260730_micro_test_run/models/manual_epoch199.ckpt \
  --data_root_dir OverSampler \
  --save_dir logs/20260730_micro_test_run/inference \
  --img_H 256 --img_W 192 \
  --denoise_steps 50