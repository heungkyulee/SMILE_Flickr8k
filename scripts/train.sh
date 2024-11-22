#!/bin/bash

torchrun \
--nproc_per_node=1 \
--master_port 30000 \
python C:/Users/Lee/SMILE/BLIP/train_caption.py \
--config C:/Users/Lee/SMILE/BLIP/configs/caption_coco.yaml \
--output_dir C:/Users/Lee/SMILE/BLIP/output/blip