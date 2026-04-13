import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from image_datasets.realesrgan import RealESRGAN_degradation
from torchvision.transforms import ToPILImage

tensor_transforms = transforms.Compose([transforms.ToTensor(),])
ram_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])


def generate_lr(img_dir='/root/autodl-tmp/data/train/', img_size=512):
    
    deg_types =['super-resolution']
    distortion = {}
    data_len = 0
    for deg_type in deg_types:
        images_gt = [os.path.join('/root/autodl-tmp/data/train/Face/generated_data_recon/lq_train', i) \
            for i in os.listdir('/root/autodl-tmp/data/train/Face/generated_data_recon/lq_train') if '.jpg' in i or '.png' in i]
        images_gt.sort()
        # print(images_gt)
        data_len = data_len + len(images_gt)
        distortion[deg_type] = images_gt
    data_lens = [len(distortion[deg_type]) for deg_type in deg_types]

    img_size = img_size
    degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')
    
    img_preproc = transforms.Compose([       
        transforms.ToTensor(),
    ])

    for i in range(0, len(images_gt)):


        index = len(images_gt) - 1 - i
        img_gt = Image.open(images_gt[i])
        GT_image_t, LR_image_t = degradation.degrade_process(np.asarray(img_gt)/255., \
                                resize_bak=True)
    
        img_gt = GT_image_t.squeeze(0)
        img_lq = LR_image_t.squeeze(0)

        img_gt = ToPILImage()(img_gt)
        img_lq = ToPILImage()(img_lq)

        lq_path = images_gt[i].replace('lq_train', 'low')
        print(lq_path)
        img_lq.save(lq_path)

generate_lr()
 