<h2 align="center">[ICLR 2026] 	Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models </h2>



## Quick Start

### Dependencies and Installation

- Python 3.11
- Pytorch >= 2.4
- CUDA >= 12
- HuggingFace CLI

1. Create Conda Environment

```
conda create --name E-Bridge python=3.11
conda activate E-Bridge 
```

2. Clone Repo

```
git clone https://github.com/jinnh/E-Bridge.git
```

3. Install Dependencies

```
cd E-Bridge
pip install -r requirements.txt
```

### Testing

You can refer to the following links to download the [pretrained model](https://drive.google.com/drive/folders/1-EL4_E5u8y4G3UjfNusq-4h45PCee0Gh?usp=sharing) and put it in the following folder:

```
├── checkpoints
    ├── E-Bridge-SR.bin
    ├── E-Bridge-Denoising.bin
```

```
# Super-resolution
CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'high-resolution, ultra-sharp, detailed' \
 --images_path input/super-resolution \
 --local_path checkpoints/E-Bridge-SR.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024 \
 --num_steps 10 --T_0 0.8 --guidance 4 \
 --control_weight 1 \
 --save_path results/super-resolution
```

```
# Image denoising
CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'noise-free, clean, smooth' \
 --images_path input/noisy \
 --local_path checkpoints/E-Bridge-Denoising.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024 \
 --num_steps 10 --T_0 0.7 --guidance 4 \
 --control_weight 1 \
 --save_path results/denoising
```

## Citation

If you find our work useful for your research, please cite our paper

```
@inproceedings{hou26energy,
    title={Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models},
    author={Jinhui Hou, Zhiyu Zhu, and Junhui Hou},
    booktitle={International Conference on Learning Representations},
    year={2026}
    }
```

## Acknowledgement

Our code is built upon [X-FLUX](https://github.com/XLabs-AI/x-flux). Thanks to the contributors for their great work.
