@echo off

python -m torch.distributed.launch --nproc_per_node=1 ^
train_caption.py ^
--config configs/caption_coco.yaml ^
--output_dir output/blip
