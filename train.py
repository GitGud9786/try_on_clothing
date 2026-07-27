import os
import argparse
from os.path import join as opj
import datetime
from importlib import import_module
from omegaconf import OmegaConf
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from utils import save_args

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--data_root_dir", type=str, default="./DATA/zalando-hd-resized")
    parser.add_argument("--category", type=str, default=None, choices=["upper", "lower_body", "dresses"])
    parser.add_argument("--vae_load_path", type=str, default="./ckpts/VITONHD_VAE_finetuning.ckpt")
    parser.add_argument("--batch_size", "-bs",  type=int, default=32)
    parser.add_argument("--transform_size", default=None, nargs="+", choices=["crop", "hflip", "shiftscale", "shiftscale2", "shiftscale3", "resize"])
    parser.add_argument("--transform_color", default=None, nargs="+", choices=["hsv", "bright_contrast", "colorjitter", "resize"])
    parser.add_argument("--use_atv_loss", action="store_true")
    parser.add_argument("--valid_epoch_freq", type=int, default=20)
    parser.add_argument("--save_every_n_epochs", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--save_root_dir", type=str, default="./logs")
    parser.add_argument("--save_name", type=str, default="dummy")

    parser.add_argument("--use_validation", action="store_false")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--logger_freq", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sd_unlocked", action="store_true")
    parser.add_argument("--all_unlocked", action="store_true")
    parser.add_argument("--only_mid_control", action="store_true")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--pbe_train_mode", action="store_true")

    parser.add_argument("--lambda_simple", type=float, default=1.0)
    parser.add_argument("--control_scales", nargs="+", type=float, default=None)
    parser.add_argument("--u_cond_percent", type=float, default=None)
    parser.add_argument("--imageclip_trainable", action="store_false")
    parser.add_argument("--no_strict_load", action="store_true")    
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verify_impl", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--semantic_cond_stage_key", type=str, default="txt")
    parser.add_argument("--main_unet_unfreeze_interval", type=int, default=0)
    parser.add_argument("--main_unet_unfreeze_epochs", type=int, default=1)
    parser.add_argument("--main_unet_unfreeze_lr", type=float, default=1e-6)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--no_validation", action="store_true")
    
    args = parser.parse_args()

    if args.verify_impl:
        args.sd_unlocked = True
        args.all_unlocked = True
        if args.u_cond_percent is None:
            args.u_cond_percent = 0.0
        if args.seed is None:
            args.seed = 123
        args.transform_size = None
        args.transform_color = None

    if args.no_aug:
        args.transform_size = None
        args.transform_color = None

    if args.no_validation:
        args.use_validation = False
    
    args.config_path = opj("./configs", f"{args.config_name}.yaml")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices.strip():
        args.n_gpus = len([device for device in cuda_visible_devices.split(",") if device.strip()])
    elif torch.cuda.is_available():
        args.n_gpus = torch.cuda.device_count()
    else:
        args.n_gpus = 1
    args.n_gpus = max(args.n_gpus, 1)
    args.devices = [i for i in range(args.n_gpus)]
    args.strategy = "ddp" if args.n_gpus > 1 else None
    args.sd_locked = not args.sd_unlocked
    args.no_validation = not args.use_validation
    
    args.valid_real_dir = opj(args.data_root_dir, "test", "image")
    args.save_dir = opj(args.save_root_dir, f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name)
    args.img_save_dir = opj(args.save_dir, "images")
    args.model_save_dir = opj(args.save_dir, "models")
    args.tb_save_dir = opj(args.save_dir, "tb")
    args.valid_img_save_dir = opj(args.save_dir, "validation_sampled_images")
    args.args_save_path = opj(args.save_dir, "args.json")
    args.config_save_path = opj(args.save_dir, "config.yaml")
    os.makedirs(args.img_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.tb_save_dir, exist_ok=True)
    os.makedirs(args.valid_img_save_dir, exist_ok=True)
    
    return args
def build_config(args, config_path=None):
    if config_path is None: 
        config_path = args.config_path
    config = OmegaConf.load(config_path)
    config.model.params.setdefault("use_VAEdownsample", False)
    config.model.params.setdefault("use_imageCLIP", False)
    config.model.params.setdefault("use_lastzc", False)
    config.model.params.setdefault("use_pbe_weight", False)
    config.model.params.setdefault("semantic_cond_stage_config", None)
    config.model.params.setdefault("semantic_cond_stage_key", "txt")
    config.model.params.setdefault("main_unet_unfreeze_interval", 0)
    config.model.params.setdefault("main_unet_unfreeze_epochs", 1)
    config.model.params.setdefault("main_unet_unfreeze_lr", 1e-6)
    if args is not None:
        override_keys = {"u_cond_percent"}
        for k, v in vars(args).items():
            if v is None:
                continue
            if k in override_keys:
                config.model.params[k] = v
            else:
                config.model.params.setdefault(k, v)
    if not config.model.params.get("validation_config", None):
        config.model.params.validation_config = OmegaConf.create()
    config.model.params.validation_config.ddim_steps = config.model.params.validation_config.get("ddim_steps", 50)
    config.model.params.validation_config.eta = config.model.params.validation_config.get("eta", 0.0)
    config.model.params.validation_config.scale = config.model.params.validation_config.get("scale", 1.0)
    if args is not None:
        config.model.params.unet_config.params.use_atv_loss = args.use_atv_loss
        config.model.params.validation_config.img_save_dir = args.valid_img_save_dir
        config.model.params.validation_config.real_dir = args.valid_real_dir
        
        if args.use_atv_loss:
            config.model.params.use_attn_mask = True
    return config


def filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered_state_dict = {}
    removed_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            removed_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            removed_keys.append(key)
            continue
        filtered_state_dict[key] = value
    if removed_keys:
        print("Skipping incompatible checkpoint keys:")
        for key in removed_keys:
            print(f"  - {key}")
    return filtered_state_dict
    
def main_worker(args):
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    config = build_config(args)
    OmegaConf.save(config, args.config_save_path)
    model = create_model(args.config_path, config=config).cpu()
    if args.resume_path is not None:
        resume_state_dict = load_state_dict(args.resume_path, location="cpu")
        resume_state_dict = filter_state_dict_for_model(model, resume_state_dict)
        if not args.no_strict_load:
            model.load_state_dict(resume_state_dict)
        else:
            model.load_state_dict(resume_state_dict, strict=False)
    elif config.resume_path is not None:
        resume_state_dict = load_state_dict(config.resume_path, location="cpu")
        resume_state_dict = filter_state_dict_for_model(model, resume_state_dict)
        if not args.no_strict_load:
            model.load_state_dict(resume_state_dict)
        else:
            model.load_state_dict(resume_state_dict, strict=False)
        
    # finetuned vae load
    if args.vae_load_path is not None:
        state_dict = load_state_dict(args.vae_load_path, location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if "loss." not in k:
                new_state_dict[k] = v.clone()
        model.first_stage_model.load_state_dict(new_state_dict)

    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    train_dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, 
        img_H=args.img_H, 
        img_W=args.img_W, 
        transform_size=args.transform_size, 
        transform_color=args.transform_color, 
    )
    valid_paired_dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, 
        img_H=args.img_H, 
        img_W=args.img_W, 
        is_test=True, 
        is_paired=True, 
        is_sorted=True, 
    )
    valid_unpaired_dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, 
        img_H=args.img_H, 
        img_W=args.img_W, 
        is_test=True, 
        is_paired=False, 
        is_sorted=True, 
    )
      
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=0, #CHANGED HERE from 4 to 0
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=True, 
        pin_memory=False
    )
    valid_paired_dataloader = DataLoader(
        valid_paired_dataset, 
        num_workers=0, #CHANGED HERE from 4 to 0
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=False, 
        pin_memory=False
    )
    valid_unpaired_dataloader = DataLoader(
        valid_unpaired_dataset, 
        num_workers=0, #CHANGED HERE from 4 to 0
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=False, 
        pin_memory=False
    )
    
    #### trainer >>>>
    img_logger = ImageLogger(
        batch_frequency=args.logger_freq,
        save_dir=args.img_save_dir,
        log_images_kwargs=config.get("log_images_kwargs", None)
    )
    tb_logger = TensorBoardLogger(args.tb_save_dir)
    # cp_callback = ModelCheckpoint(
    #     dirpath=args.model_save_dir, 
    #     filename="[Train]_[{epoch}]_[{train_loss_epoch:.04f}]", 
    #     save_top_k=1, 
    #     every_n_epochs=args.save_every_n_epochs, 
    #     save_last=False, #CHANGED HERE from False to True
    #     save_on_train_epoch_end=True,
    #     save_weights_only=True #ADDED THIS TO SAVE ONLY MODEL WEIGHTS, NOT THE ENTIRE CHECKPOINT (WHICH CAN BE LARGE DUE TO OPTIMIZER STATE, ETC.
    # )

    trainer_kwargs = dict(
        precision=args.precision,
        callbacks=[img_logger],
        logger=tb_logger,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accum_iter,
        check_val_every_n_epoch=args.valid_epoch_freq,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_checkpointing=False,
    )
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    #### trainer <<<<
    
    if not args.no_validation:
        trainer.fit(model, train_dataloader, [valid_paired_dataloader, valid_unpaired_dataloader])
    else:
        trainer.fit(model, train_dataloader)

    print("\n" + "="*70)
    print("TRAINING COMPLETED - Saving checkpoint manually...")
    print("="*70)
    
    import gc
    import time
    import psutil
    
    # Show current memory
    mem = psutil.virtual_memory()
    print(f"\nRAM before cleanup:")
    print(f"  Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")
    print(f"  Available: {mem.available / (1024**3):.2f} GB")
    
    # Step 1: Move model to CPU (frees GPU memory)
    print("\n[1/5] Moving model from GPU to CPU...")
    model = model.cpu()
    print("  ✅ Model moved to CPU")
    
    # Step 2: Clear CUDA cache
    print("[2/5] Clearing CUDA cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  ✅ CUDA cache cleared")
    
    # Step 3: Aggressive garbage collection
    print("[3/5] Running garbage collection (takes ~6 seconds)...")
    for i in range(3):
        collected = gc.collect()
        print(f"  GC pass {i+1}: {collected} objects collected")
        time.sleep(2)
    
    # Show memory after cleanup
    mem = psutil.virtual_memory()
    print(f"\n[4/5] RAM after cleanup:")
    print(f"  Available: {mem.available / (1024**3):.2f} GB")
    
    # Step 4: Create checkpoint
    print(f"\n[5/5] Creating and saving checkpoint...")
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': trainer.current_epoch,
        'global_step': trainer.global_step,
    }
    
    checkpoint_path = os.path.join(
        args.model_save_dir,
        f"manual_epoch{trainer.current_epoch}.ckpt"
    )
    
    print(f"  Saving to: {checkpoint_path}")
    
    try:
        torch.save(checkpoint, checkpoint_path)
        size_gb = os.path.getsize(checkpoint_path) / (1024**3)
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! Checkpoint saved successfully!")
        print(f"✅ Size: {size_gb:.2f} GB")
        print(f"✅ Path: {checkpoint_path}")
        print(f"{'='*70}")
        
    except MemoryError as e:
        print(f"\n❌ MemoryError: {e}")
        print("\nAttempting emergency save")
        
        try:
            emergency_path = f"D:/emergency_epoch{trainer.current_epoch}.ckpt"
            os.makedirs("D:/", exist_ok=True)
            torch.save(checkpoint, emergency_path)
            size_gb = os.path.getsize(emergency_path) / (1024**3)
            print(f"✅ Emergency save successful!")
            print(f"  Size: {size_gb:.2f} GB")
            
        except Exception as e2:
            print(f"❌ Emergency save also failed: {e2}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    args = build_args()
    print(args)
    save_args(args, args.args_save_path)
    main_worker(args)
    print("Done")