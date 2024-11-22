'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li

 * Modified by Zihao Yue
'''

import argparse
import os
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model import caption_model
import utils
from utils import warmup_lr_schedule, step_lr_schedule, cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result
# coco_caption_eval 함수는 더 이상 사용되지 않으므로 임포트에서 제거합니다.

# NLTK 라이브러리 임포트
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')
nltk.download('omw-1.4')  # WordNet 관련 추가 데이터


# NLTK 데이터 다운로드 (필요한 경우 주석을 해제하고 한 번만 실행)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)       
        
        loss = model(image, caption)      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        # 데이터로더의 반환 값 처리
        if len(samples) == 2:
            image, image_id = samples
        elif len(samples) == 3:
            image, _, image_id = samples  # caption은 사용하지 않음
        else:
            raise ValueError("Unexpected number of values returned by the data loader")

        image = image.to(device)       
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result

def evaluate_metrics(dataset, results):
    # 이미지 ID를 키로 하고, 참조 캡션 리스트를 값으로 하는 딕셔너리 생성
    refs = {}
    for ann in dataset.ann['annotations']:
        img_id = ann['image_id']
        caption = ann['caption']
        if img_id not in refs:
            refs[img_id] = []
        refs[img_id].append(caption)
    
    # 결과에서 예측 캡션 추출
    hyps = {}
    for res in results:
        img_id = res['image_id']
        caption = res['caption']
        hyps[img_id] = [caption]
    
    # BLEU, METEOR 계산
    list_of_references = []
    hypotheses = []
    for img_id in hyps.keys():
        if img_id in refs:
            # 참조 캡션을 토큰화
            tokenized_refs = [ref.split() for ref in refs[img_id]]
            # 예측 캡션을 토큰화
            tokenized_hyp = hyps[img_id][0].split()
            list_of_references.append(tokenized_refs)
            hypotheses.append(tokenized_hyp)
    
    # BLEU 점수 계산
    bleu_score = corpus_bleu(list_of_references, hypotheses)
    
    # METEOR 점수 계산
    meteor_scores = [
        meteor_score(
            [ref.split() for ref in refs[img_id]],  # 참조 캡션의 리스트 (토큰화된 리스트)
            hyps[img_id][0].split()                 # 예측 캡션 (토큰화된 리스트)
        )
        for img_id in hyps.keys() if img_id in refs
    ]
    meteor_score_avg = sum(meteor_scores) / len(meteor_scores)
    
    eval_result = {
        'BLEU': bleu_score,
        'METEOR': meteor_score_avg,
    }
    
    return eval_result




def main(args, config):
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_flickr8k', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[config['batch_size']]*3,
        num_workers=[8,8,8],
        is_trains=[True, False, False],
        collate_fns=[None,None,None]
    )         

    #### Model #### 
    print("Creating model")
    model = caption_model(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        
        if args.eval_split == 'val' or not args.evaluate:
            val_result = evaluate(model_without_ddp, val_loader, device, config)  
            val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')
        else:
            test_result = evaluate(model_without_ddp, test_loader, device, config)  
            test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')

        if utils.is_main_process():
            # Flickr8k 데이터셋에 맞게 평가 메트릭 계산
            if args.eval_split == 'val' or not args.evaluate:
                eval_result = evaluate_metrics(val_dataset, val_result)
            else:
                eval_result = evaluate_metrics(test_dataset, test_result)
            
            if args.evaluate:            
                log_stats = {
                    **{f'{args.eval_split}_{k}': v for k, v in eval_result.items()},                     
                }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                # 성능 기준에 따라 모델 저장 (여기서는 BLEU 점수를 기준으로 예시)
                if eval_result['BLEU'] > best:
                    best = eval_result['BLEU']
                    best_epoch = epoch                
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                # 에폭별로 모델 저장
                torch.save(save_obj, os.path.join(args.output_dir, 'epoch%d.pth'%epoch))
                    
                log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in eval_result.items()},                     
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break
        if args.distributed:
            dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_flickr8k.yaml')
    parser.add_argument('--output_dir', default='output/caption_flickr8k')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_split', default='val', type=str)
    args = parser.parse_args()

    yaml = YAML()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
