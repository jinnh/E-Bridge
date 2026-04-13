import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

def random_crop(image1, image2, crop_size):

    image_width, image_height = image1.size
    crop_height, crop_width = crop_size
    assert image_width >= crop_width and image_height >= crop_height, "裁剪尺寸不能大于图像尺寸。"
    x = random.randint(0, image_width - crop_width)
    y = random.randint(0, image_height - crop_height)
    crop1 = image1.crop((x, y, x + crop_width, y + crop_height))
    crop2 = image2.crop((x, y, x + crop_width, y + crop_height))
    return crop1, crop2
        

class MultiTaskBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, steps_per_epoch, p_new_task=0.5):
        """
        自定义批次采样器，用于多任务学习中的持续学习场景。
        
        Args:
            dataset (Dataset): 已经重构的 CustomImageDataset 对象。
            batch_size (int): 每个批次的大小。
            steps_per_epoch (int): 每个 epoch 包含多少个批次。
            p_new_task (float): 每个批次来自新任务的概率。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.p_new_task = p_new_task

        self.old_tasks_with_data = [t for t in self.dataset.old_tasks if t in self.dataset.task_data]
        self.new_tasks_with_data = [t for t in self.dataset.new_tasks if t in self.dataset.task_data]

        if not self.old_tasks_with_data and not self.new_tasks_with_data:
            raise ValueError("No tasks with data found in the dataset.")
            
        print(f"Sampler initialized. Old tasks: {self.old_tasks_with_data}, New tasks: {self.new_tasks_with_data}")
        print(f"Sampling probability for new tasks: {self.p_new_task}")

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            for _ in range(self.batch_size):
                # 决定是从新任务还是旧任务中采样
                if random.random() < self.p_new_task and self.new_tasks_with_data:
                    # 从新任务组中选择
                    task_name = random.choice(self.new_tasks_with_data)
                elif self.old_tasks_with_data:
                    # 从旧任务组中选择
                    task_name = random.choice(self.old_tasks_with_data)
                else:
                    # 如果没有旧任务，就只能从新任务中选
                    task_name = random.choice(self.new_tasks_with_data)
                
                # 从选定的任务中随机选择一张图片
                num_images_in_task = len(self.dataset.task_data[task_name])
                img_idx = random.randint(0, num_images_in_task - 1)
                
                batch.append((task_name, img_idx))
            
            yield batch

    def __len__(self):
        return self.steps_per_epoch
    

class ImageProcessor:
    """
    一个辅助类，用于处理特定任务的图像加载和增强。
    这有助于清理 __getitem__ 中的 if/else 逻辑。
    """
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _process_demoire_uhdm(self, lq_pil, gt_pil):
        """专门处理 demoire_UHDM 的多尺度裁剪逻辑"""
        p1 = np.random.randint(10)
        
        # 70% 的概率：直接在原图上裁剪
        if p1 > 3:
            # 直接从原图裁剪出目标尺寸
            lq_pil, gt_pil = random_crop(lq_pil, gt_pil, (self.img_size, self.img_size))
        # 30% 的概率：先缩放再裁剪，模拟多尺度训练
        else:
            p2 = np.random.randint(9)
            if p2 < 3:
                # 缩放到 2.5K，然后裁剪
                lq_pil_resized = lq_pil.resize((2560, 1440), Image.BICUBIC)
                gt_pil_resized = gt_pil.resize((2560, 1440), Image.BICUBIC)
                lq_pil, gt_pil = random_crop(lq_pil_resized, gt_pil_resized, (self.img_size, self.img_size))
            elif 3 <= p2 < 6:
                # 缩放到 1080p，然后裁剪
                lq_pil_resized = lq_pil.resize((1920, 1080), Image.BICUBIC)
                gt_pil_resized = gt_pil.resize((1920, 1080), Image.BICUBIC)
                lq_pil, gt_pil = random_crop(lq_pil_resized, gt_pil_resized, (self.img_size, self.img_size))
            else:
                # 直接缩放到目标尺寸
                lq_pil = lq_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
                gt_pil = gt_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
                
        return lq_pil, gt_pil

    def process(self, lq_pil, gt_pil, task_name):
        """
        根据任务名称应用不同的数据增强策略。
        """
        # 统一的随机裁剪和缩放逻辑
        if task_name == 'demoire_UHDM':
            lq_pil, gt_pil = self._process_demoire_uhdm(lq_pil, gt_pil)
        if task_name in ['noisy', 'super-resolution', ]:
            # 对于大图，优先随机裁剪
            if gt_pil.size[0] > self.img_size and gt_pil.size[1] > self.img_size:
                lq_pil, gt_pil = random_crop(lq_pil, gt_pil, (self.img_size, self.img_size))
            else: # 小图则直接缩放
                lq_pil = lq_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
                gt_pil = gt_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        else: # 其他任务默认缩放
            lq_pil = lq_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
            gt_pil = gt_pil.resize((self.img_size, self.img_size), Image.BICUBIC)

        # 统一的随机翻转
        if random.random() > 0.5:
            lq_pil = TF.hflip(lq_pil)
            gt_pil = TF.hflip(gt_pil)
        
        # 统一转换为 Tensor 并归一化到 [-1, 1]
        lq_tensor = TF.to_tensor(lq_pil) * 2.0 - 1.0
        gt_tensor = TF.to_tensor(gt_pil) * 2.0 - 1.0
        
        # 为 RAM 准备的低分辨率图像
        ram_image = self.ram_transforms(lq_pil)
        
        return lq_tensor, gt_tensor, ram_image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=1024):
        self.img_dir = img_dir
        self.img_size = img_size

        # 旧任务和新任务
        self.old_tasks = ['super-resolution',]
        self.new_tasks = ['super-resolution',]

        self.all_tasks = self.old_tasks + self.new_tasks

        self.task_data = {}
        self.task_prompts = {
            'super-resolution': 'high-resolution, ultra-sharp, detailed',
            'noisy': 'noise-free, clean, smooth',
            'raindrop': 'remove raindrops, clean',
            'low-light': 'bright, clear, vivid',
            'demoire_UHDM': 'remove moiré artifacts, clear patterns',
        }
        self.image_processor = ImageProcessor(img_size)

        print("Loading dataset paths...")
        for task_name in self.all_tasks:
            high_path = os.path.join(img_dir, task_name, 'high')
            if not os.path.exists(high_path):
                print(f"Warning: Path not found for task '{task_name}': {high_path}")
                continue
            
            image_paths = [os.path.join(high_path, f) for f in os.listdir(high_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_paths:
                self.task_data[task_name] = sorted(image_paths)
        print("Dataset paths loaded.")

    def __len__(self):
        return sum(len(paths) for paths in self.task_data.values())

    def get_lq_path(self, gt_path, task_name):
        """
        将路径查找逻辑集中到这里。
        这是本次修正的核心。
        """
        gt_dir, gt_filename = os.path.split(gt_path)
        lq_dir = gt_dir.replace('high', 'low')
        
        if task_name == 'raindrop':
            return gt_path.replace('high', 'low').replace('clean', 'rain')
        
        else:
            # high -> low
            return os.path.join(lq_dir, gt_filename)

    def __getitem__(self, index_tuple):
        task_name, img_idx = index_tuple
        
        try:
            gt_path = self.task_data[task_name][img_idx]
            lq_path = self.get_lq_path(gt_path, task_name)

            print(lq_path)
            with Image.open(gt_path).convert('RGB') as gt_pil, \
                 Image.open(lq_path).convert('RGB') as lq_pil:
                
                raw_size = gt_pil.size
                
                lq_tensor, gt_tensor, ram_image = self.image_processor.process(lq_pil, gt_pil, task_name)
                
                prompt = self.task_prompts.get(task_name, "general image restoration") 

                return gt_tensor, lq_tensor, prompt, ram_image, raw_size, task_name

        except Exception as e:
            print(f"Error loading data for task '{task_name}' at index {img_idx}: {e}")
            print(f"GT Path: {gt_path}, LQ Path: {lq_path}")
            return None



def safe_collate(batch):
    """
    一个自定义的 collate_fn 可以过滤掉在 __getitem__ 中返回 None 的坏样本。
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([]), [], [], [], []
    return torch.utils.data.dataloader.default_collate(batch)


def create_continual_learning_loader(img_dir, img_size, batch_size, num_workers, steps_per_epoch, p_new_task=0.5):
    """
    创建用于持续学习的 DataLoader。
    """
    # 1. 实例化重构后的 Dataset
    dataset = CustomImageDataset(img_dir=img_dir, img_size=img_size)

    # 2. 实例化自定义的批次采样器
    batch_sampler = MultiTaskBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        p_new_task=p_new_task
    )

    # 3. 创建 DataLoader
    # 注意：当使用 batch_sampler 时，batch_size, shuffle, sampler, drop_last 都必须为 None 或默认值
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=safe_collate  # 使用安全的数据整理函数
    )
    
    return loader

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    batch_sampler = MultiTaskBatchSampler(
        dataset=dataset,
        batch_size=train_batch_size,
        steps_per_epoch=1000,
        p_new_task=0.6
    )
    
    # 注意：当使用 batch_sampler 时, batch_size, shuffle, sampler, drop_last 都必须为 None 或默认值
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=safe_collate  # 使用安全的数据整理函数
    )
    return loader


if __name__ == '__main__':
    IMG_DIR = '/root/autodl-tmp/data/train' # 替换为你的数据集根目录
    IMG_SIZE = 1024
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    STEPS_PER_EPOCH = 500  # 假设每个 epoch 训练 500 个批次
    P_NEW_TASK = 0.6       # 60% 的批次将来自新任务，40% 来自旧任务以防止遗忘

    # 创建 DataLoader
    train_loader = create_continual_learning_loader(
        img_dir=IMG_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        steps_per_epoch=STEPS_PER_EPOCH,
        p_new_task=P_NEW_TASK
    )

    # 迭代 DataLoader 进行测试
    print("\n--- Testing DataLoader ---")
    for i, batch_data in enumerate(train_loader):
        if i >= 500: 
            break
        
        gt_tensors, lq_tensors, prompts, ram_images, raw_sizes, task_names = batch_data
        
        if gt_tensors.nelement() == 0:
            print(f"Batch {i}: Skipped due to loading errors.")
            continue

        print(f"Batch {i+1}/{STEPS_PER_EPOCH}:")
        print(f"  - Task Names: {task_names}")
        print(f"  - GT Tensor Shape: {gt_tensors.shape}")
        print(f"  - LQ Tensor Shape: {lq_tensors.shape}")
        print(f"  - Prompts: {prompts[0]}...") 
        print("-" * 20)