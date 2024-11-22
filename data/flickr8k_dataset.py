import os
from PIL import Image
from torch.utils.data import Dataset
import json

class Flickr8kDataset(Dataset):
    def __init__(self, ann_file, transform, image_root, is_train=True):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.is_train = is_train
        self.cache = {}

    def __len__(self):
        return len(self.ann['images'])

    def __getitem__(self, index):
        if self.is_train:
            ann = self.ann['annotations'][index]
            img_id = ann['image_id']
            caption = ann['caption']
            img_info = self.ann['images'][img_id - 1]
            img_name = img_info['file_name']

            if img_name in self.cache:
                image = self.cache[img_name]
            else:
                img_path = os.path.join(self.image_root, img_name)
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.cache[img_name] = image

            return image, caption, img_id
        else:
            # 평가 모드에서는 캡션을 반환하지 않음
            img_info = self.ann['images'][index]
            img_id = img_info['id']
            img_name = img_info['file_name']

            if img_name in self.cache:
                image = self.cache[img_name]
            else:
                img_path = os.path.join(self.image_root, img_name)
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.cache[img_name] = image

            return image, img_id

