CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'high-resolution, ultra-sharp, detailed' \
 --neg_prompt '' \
 --images_path /root/autodl-tmp/data/test/super-resolution/low \
 --local_path checkpoints/sr.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  \
 --num_steps 10 --T_0 0.9 --guidance 4 \
 --control_weight 1 \
 --save_path results/super-resolution-s10-C1-To0p9


