import json
import os

# 경로 설정
captions_file = 'data/Flickr8k/captions.txt'
output_dir = 'data/Flickr8k/annotations'
os.makedirs(output_dir, exist_ok=True)

# 분할 파일 경로
train_file = 'data/Flickr8k/Flickr_8k.trainImages.txt'
val_file = 'data/Flickr8k/Flickr_8k.devImages.txt'
test_file = 'data/Flickr8k/Flickr_8k.testImages.txt'

# 분할 파일 읽기
def load_image_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

train_list = load_image_list(train_file)
val_list = load_image_list(val_file)
test_list = load_image_list(test_file)

# 이미지 이름을 키로 하는 딕셔너리 생성
split_dict = {}
for img_name in train_list:
    split_dict[img_name] = 'train'
for img_name in val_list:
    split_dict[img_name] = 'val'
for img_name in test_list:
    split_dict[img_name] = 'test'

# 이미지 및 캡션 정보를 저장할 딕셔너리 초기화
datasets = {
    'train': {'images': [], 'annotations': []},
    'val': {'images': [], 'annotations': []},
    'test': {'images': [], 'annotations': []}
}

# 이미지 ID와 어노테이션 ID를 위한 카운터 딕셔너리
image_id_counters = {'train': 0, 'val': 0, 'test': 0}
annotation_id_counters = {'train': 0, 'val': 0, 'test': 0}

# captions.txt 파일 읽기
with open(captions_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 이미지 이름과 ID 매핑을 위한 딕셔너리
image_name_to_id = {'train': {}, 'val': {}, 'test': {}}

# 첫 번째 줄 헤더 스킵
lines = lines[1:]

for line in lines:
    line = line.strip()
    if not line:
        continue
    # 쉼표로 분리
    if ',' in line:
        image_info, caption = line.split(',', 1)
    else:
        continue  # 쉼표가 없으면 스킵

    image_name = image_info.strip()

    # 해당 이미지가 어떤 분할에 속하는지 확인
    split = split_dict.get(image_name)
    if split is None:
        continue  # 분할에 포함되지 않은 이미지

    # 이미지 ID 할당
    if image_name not in image_name_to_id[split]:
        image_id_counters[split] += 1
        image_id = image_id_counters[split]
        image_name_to_id[split][image_name] = image_id

        # 이미지 정보 추가
        datasets[split]['images'].append({
            'file_name': image_name,
            'id': image_id
        })
    else:
        image_id = image_name_to_id[split][image_name]

    # 어노테이션 ID 할당
    annotation_id_counters[split] += 1
    annotation_id = annotation_id_counters[split]

    # 어노테이션 추가
    datasets[split]['annotations'].append({
        'image_id': image_id,
        'id': annotation_id,
        'caption': caption.strip()
    })

# COCO 형식의 JSON 파일 저장
for split in ['train', 'val', 'test']:
    dataset = {
        'images': datasets[split]['images'],
        'annotations': datasets[split]['annotations'],
        'type': 'captions',
        'licenses': [],
        'info': {}
    }
    output_path = os.path.join(output_dir, f'flickr8k_{split}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
