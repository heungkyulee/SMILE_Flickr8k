import re
import json
import os

import torch
import torch.distributed as dist
from torchvision import transforms

import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename, utils.get_rank()))
    with open(result_file, 'w') as f:
        json.dump(result, f)

    # 분산 학습 여부 확인
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        result_files = [os.path.join(result_dir, '%s_rank%d.json'%(filename, rank)) for rank in range(utils.get_world_size())]
    else:
        result_files = [result_file]  # 단일 프로세스 모드에서는 현재 결과 파일만 사용

    if utils.is_main_process():
        # 모든 결과 파일을 모아서 하나의 결과 파일로 만듦
        all_results = []
        for f in result_files:
            with open(f, 'r') as infile:
                all_results.extend(json.load(infile))

        if remove_duplicate:
            unique_results = []
            ids = set()
            for res in all_results:
                if res[remove_duplicate] not in ids:
                    ids.add(res[remove_duplicate])
                    unique_results.append(res)
            all_results = unique_results

        with open(os.path.join(result_dir, '%s.json' % filename), 'w') as outfile:
            json.dump(all_results, outfile)
        print('Result file saved to %s' % os.path.join(result_dir, '%s.json' % filename))

    return os.path.join(result_dir, '%s.json' % filename)

def get_transform(image_size, is_train, min_scale=0.5):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transform

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval