# # VITONHD base
# CUDA_VISIBLE_DEVICES=3,4 python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --save_name Base_test


# # VITONHD ATVloss
# CUDA_VISIBLE_DEVICES=5,6 python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --use_atv_loss \
#  --resume_path <first stage model path> \
#  --save_name ATVloss_test

# %env CUDA_VISIBLE_DEVICES=0
# %env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#CODE FOR TRAINING
python train.py \
  --config_name VITONHD \
  --data_root_dir OverSampler \
  --save_root_dir logs \
  --save_name micro_test_run \
  --batch_size 1 \
  --accum_iter 4 \
  --img_H 512 \
  --img_W 384 \
  --max_epochs 200 \
  --valid_epoch_freq 1 \
  --logger_freq 10 \
  --precision 16 \
  --resume_path ckpts/VITONHD_PBE_pose.ckpt \
  --vae_load_path ckpts/VITONHD_VAE_finetuning.ckpt \
  --no_strict_load \
  --no_validation
