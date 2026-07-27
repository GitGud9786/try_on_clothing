import os
import json
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
            transform_size=None, 
            transform_color=None,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.lmm_captions = self._load_lmm_captions()
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_size_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
            transform_crop_person_lst,
            additional_targets={"agn":"image", 
                        "agn_mask":"image", 
                        "cloth_mask_warped":"image", 
                        "cloth_warped":"image", 
                        "image_densepose":"image", 
                        "image_parse":"image", 
                        "gt_cloth_warped_inner_mask":"image",
                        "gt_cloth_warped_outer_mask":"image",
                        }
        )
        self.transform_crop_cloth = A.Compose(
            transform_crop_cloth_lst,
            additional_targets={
                "cloth_inner":"image",
                "cloth_outer":"image",
                "cloth_inner_mask":"image",
                "cloth_outer_mask":"image",
            }
        )

        self.transform_size = A.Compose(
            transform_size_lst,
            additional_targets={"agn":"image", 
                        "agn_mask":"image", 
                        "cloth_inner":"image",
                        "cloth_outer":"image",
                        "cloth_inner_mask":"image",
                        "cloth_outer_mask":"image",
                        "cloth_mask_warped":"image", 
                        "cloth_warped":"image", 
                        "image_densepose":"image", 
                        "image_parse":"image", 
                        "gt_cloth_warped_inner_mask":"image",
                        "gt_cloth_warped_outer_mask":"image",
                        }
            )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image", 
                                    "cloth_inner":"image",
                                    "cloth_outer":"image",
                                    "cloth_warped":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def _load_lmm_captions(self):
        candidate_paths = [
            opj(self.drd, self.data_type, "lmm_captions.json"),
            opj(self.drd, "train", "lmm_captions.json"),
            opj(self.drd, "test", "lmm_captions.json"),
        ]
        for caption_path in candidate_paths:
            if os.path.isfile(caption_path):
                with open(caption_path, "r") as f:
                    return json.load(f)
        return {}

    def _resolve_first_existing_path(self, *candidate_paths):
        for path in candidate_paths:
            if path is not None and os.path.isfile(path):
                return path
        return None

    def _load_cloth_pair(self, cloth_name):
        cloth_inner_path = self._resolve_first_existing_path(
            opj(self.drd, self.data_type, "cloth_inner", cloth_name),
            opj(self.drd, self.data_type, "cloth", cloth_name),
        )
        cloth_outer_path = self._resolve_first_existing_path(
            opj(self.drd, self.data_type, "cloth_outer", cloth_name),
            opj(self.drd, self.data_type, "cloth", cloth_name),
        )
        cloth_inner_mask_path = self._resolve_first_existing_path(
            opj(self.drd, self.data_type, "cloth-inner-mask", cloth_name),
            opj(self.drd, self.data_type, "cloth_mask", cloth_name),
            opj(self.drd, self.data_type, "cloth-mask", cloth_name),
        )
        cloth_outer_mask_path = self._resolve_first_existing_path(
            opj(self.drd, self.data_type, "cloth-outer-mask", cloth_name),
            opj(self.drd, self.data_type, "cloth_mask", cloth_name),
            opj(self.drd, self.data_type, "cloth-mask", cloth_name),
        )
        gt_cloth_warped_inner_mask_path = self._resolve_first_existing_path(
            opj(self.drd, self.data_type, "gt_cloth_warped_inner_mask", cloth_name),
            opj(self.drd, self.data_type, "gt_cloth_warped_mask", cloth_name),
        )
        gt_cloth_warped_outer_mask_path = self._resolve_first_existing_path(
            opj(self.drd, self.data_type, "gt_cloth_warped_outer_mask", cloth_name),
            opj(self.drd, self.data_type, "gt_cloth_warped_mask", cloth_name),
        )

        cloth_inner = cloth_outer = cloth_inner_mask = cloth_outer_mask = None
        gt_cloth_warped_inner_mask = gt_cloth_warped_outer_mask = None

        if cloth_inner_path is not None:
            cloth_inner = cloth_inner_path
        if cloth_outer_path is not None:
            cloth_outer = cloth_outer_path
        if cloth_inner_mask_path is not None:
            cloth_inner_mask = cloth_inner_mask_path
        if cloth_outer_mask_path is not None:
            cloth_outer_mask = cloth_outer_mask_path
        if gt_cloth_warped_inner_mask_path is not None:
            gt_cloth_warped_inner_mask = gt_cloth_warped_inner_mask_path
        if gt_cloth_warped_outer_mask_path is not None:
            gt_cloth_warped_outer_mask = gt_cloth_warped_outer_mask_path

        return (
            cloth_inner,
            cloth_outer,
            cloth_inner_mask,
            cloth_outer_mask,
            gt_cloth_warped_inner_mask,
            gt_cloth_warped_outer_mask,
        )

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        if self.transform_size is None and self.transform_color is None:
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                in_inverse_mask=True
            )
            # cloth = imread(
            #     opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
            #     self.img_H, 
            #     self.img_W
            # )
            (
                cloth_inner_path,
                cloth_outer_path,
                cloth_inner_mask_path,
                cloth_outer_mask_path,
                gt_cloth_warped_inner_mask_path,
                gt_cloth_warped_outer_mask_path,
            ) = self._load_cloth_pair(self.c_names[self.pair_key][idx])

            if cloth_inner_path is None or cloth_outer_path is None:
                raise FileNotFoundError(f"Could not find cloth image for {cloth_fn}")
            if cloth_inner_mask_path is None or cloth_outer_mask_path is None:
                raise FileNotFoundError(f"Could not find cloth mask for {cloth_fn}")

            cloth_inner = imread(cloth_inner_path, self.img_H, self.img_W)
            cloth_outer = imread(cloth_outer_path, self.img_H, self.img_W)
            cloth_inner_mask = imread(
                cloth_inner_mask_path,
                self.img_H,
                self.img_W,
                is_mask=True,
                cloth_mask_check=True,
            )
            cloth_outer_mask = imread(
                cloth_outer_mask_path,
                self.img_H,
                self.img_W,
                is_mask=True,
                cloth_mask_check=True,
            )

            gt_cloth_warped_inner_mask = imread(
                gt_cloth_warped_inner_mask_path,
                self.img_H,
                self.img_W,
                is_mask=True,
            ) if (not self.is_test and gt_cloth_warped_inner_mask_path is not None) else np.zeros_like(agn_mask)

            gt_cloth_warped_outer_mask = imread(
                gt_cloth_warped_outer_mask_path,
                self.img_H,
                self.img_W,
                is_mask=True,
            ) if (not self.is_test and gt_cloth_warped_outer_mask_path is not None) else np.zeros_like(agn_mask)

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)

        else:
            agn = imread_for_albu(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]),
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            )
            agn_mask = imread_for_albu(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")),
                is_mask=True,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            )
            # cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            (
                cloth_inner_path,
                cloth_outer_path,
                cloth_inner_mask_path,
                cloth_outer_mask_path,
                gt_cloth_warped_inner_mask_path,
                gt_cloth_warped_outer_mask_path,
            ) = self._load_cloth_pair(self.c_names[self.pair_key][idx])

            if cloth_inner_path is None or cloth_outer_path is None:
                raise FileNotFoundError(f"Could not find cloth image for {cloth_fn}")
            if cloth_inner_mask_path is None or cloth_outer_mask_path is None:
                raise FileNotFoundError(f"Could not find cloth mask for {cloth_fn}")

            cloth_inner = imread_for_albu(
                cloth_inner_path,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            )
            cloth_outer = imread_for_albu(
                cloth_outer_path,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            )
            cloth_inner_mask = imread_for_albu(
                cloth_inner_mask_path,
                is_mask=True,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
                cloth_mask_check=True
            )
            cloth_outer_mask = imread_for_albu(
                cloth_outer_mask_path,
                is_mask=True,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
                cloth_mask_check=True
            )

            gt_cloth_warped_inner_mask = imread_for_albu(
                gt_cloth_warped_inner_mask_path,
                is_mask=True,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            ) if (not self.is_test and gt_cloth_warped_inner_mask_path is not None) else np.zeros_like(agn_mask)

            gt_cloth_warped_outer_mask = imread_for_albu(
                gt_cloth_warped_outer_mask_path,
                is_mask=True,
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            ) if (not self.is_test and gt_cloth_warped_outer_mask_path is not None) else np.zeros_like(agn_mask)
                
            image = imread_for_albu(
                opj(self.drd, self.data_type, "image", self.im_names[idx]),
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            )
            image_densepose = imread_for_albu(
                opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]),
                use_resize=True,
                height=self.img_H,
                width=self.img_W,
            )

            if self.transform_size is not None:
                transformed = self.transform_size(
                    image=image, 
                    agn=agn, 
                    agn_mask=agn_mask, 
                    cloth_inner=cloth_inner,
                    cloth_outer=cloth_outer,
                    cloth_inner_mask=cloth_inner_mask,
                    cloth_outer_mask=cloth_outer_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_inner_mask=gt_cloth_warped_inner_mask,
                    gt_cloth_warped_outer_mask=gt_cloth_warped_outer_mask,
                )
                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_inner_mask=transformed["gt_cloth_warped_inner_mask"]
                gt_cloth_warped_outer_mask=transformed["gt_cloth_warped_outer_mask"]

                cloth_inner=transformed["cloth_inner"]
                cloth_outer=transformed["cloth_outer"]
                cloth_inner_mask=transformed["cloth_inner_mask"]
                cloth_outer_mask=transformed["cloth_outer_mask"]
                
            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_inner_mask=gt_cloth_warped_inner_mask,
                    gt_cloth_warped_outer_mask=gt_cloth_warped_outer_mask,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                image_densepose=transformed_image["image_densepose"]
                gt_cloth_warped_inner_mask=transformed["gt_cloth_warped_inner_mask"]
                gt_cloth_warped_outer_mask=transformed["gt_cloth_warped_outer_mask"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    cloth_inner=cloth_inner,
                    cloth_outer=cloth_outer,
                    cloth_inner_mask=cloth_inner_mask,
                    cloth_outer_mask=cloth_outer_mask
                )

                cloth_inner=transformed_cloth["cloth_inner"]
                cloth_outer=transformed_cloth["cloth_outer"]
                cloth_inner_mask=transformed_cloth["cloth_inner_mask"]
                cloth_outer_mask=transformed_cloth["cloth_outer_mask"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn, 
                    cloth_inner=cloth_inner,
                    cloth_outer=cloth_outer,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                cloth_inner=transformed["cloth_inner"]
                cloth_outer=transformed["cloth_outer"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            agn = norm_for_albu(agn)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            cloth_inner = norm_for_albu(cloth_inner)
            cloth_outer = norm_for_albu(cloth_outer)
            cloth_inner_mask = norm_for_albu(cloth_inner_mask, is_mask=True)
            cloth_outer_mask = norm_for_albu(cloth_outer_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_inner_mask = norm_for_albu(gt_cloth_warped_inner_mask, is_mask=True)
            gt_cloth_warped_outer_mask = norm_for_albu(gt_cloth_warped_outer_mask, is_mask=True)
            
        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth_inner=cloth_inner,
            cloth_outer=cloth_outer,
            cloth_inner_mask=cloth_inner_mask,
            cloth_outer_mask=cloth_outer_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_inner_mask=gt_cloth_warped_inner_mask,
            gt_cloth_warped_outer_mask=gt_cloth_warped_outer_mask,
            txt=self.lmm_captions.get(cloth_fn, ""),
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )