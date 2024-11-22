# SMILE 기반 이미지 캡셔닝 프로젝트

이 프로젝트는 [BLIP](https://github.com/salesforce/BLIP) 및 [SMILE](https://github.com/yuezih/SMILE) 레포지토리를 기반으로 하며, Flickr8k 데이터셋을 사용하여 이미지 캡셔닝 모델을 파인튜닝하고 Streamlit을 통해 데모를 제공합니다.

## 목차

1. [프로젝트 변경 사항](#1-프로젝트-변경-사항)
2. [Flickr8k 데이터셋 다운로드 및 파인튜닝 과정](#2-flickr8k-데이터셋-다운로드-및-파인튜닝-과정)
3. [Streamlit을 사용한 데모 실행 방법](#3-streamlit을-사용한-데모-실행-방법)

---

## 1. 프로젝트 변경 사항

원본 GitHub 레포지토리로부터 다음과 같은 변경 사항이 있습니다:

- **데이터셋 지원**: COCO 데이터셋 대신 **Flickr8k 데이터셋**을 사용하도록 변경하였습니다.
- **구성 파일 수정**: `configs/caption_flickr8k.yaml` 파일을 추가하여 Flickr8k 데이터셋에 맞게 설정을 변경하였습니다.
- **데이터 로더 수정**: `data/flickr8k_dataset.py` 파일을 추가하여 Flickr8k 데이터셋을 처리하도록 하였습니다.
- **평가 함수 수정**: Flickr8k 데이터셋에 맞는 평가 방식을 적용하였습니다.
- **모델 파인튜닝 스크립트 수정**: `train_caption.py`를 수정하여 Flickr8k 데이터셋으로 파인튜닝할 수 있도록 변경하였습니다.
- **Streamlit 데모 추가**: `app.py` 파일을 추가하여 학습된 모델을 사용한 이미지 캡셔닝 데모를 제공합니다.

---

## 2. Flickr8k 데이터셋 다운로드 및 파인튜닝 과정

### 2.1. Flickr8k 데이터셋 다운로드

1. **Flickr8k 데이터셋 페이지로 이동**: [Flickr8k 데이터셋](https://forms.illinois.edu/sec/1713398)
2. **데이터셋 다운로드 요청**:
   - 페이지에서 요구하는 정보를 입력하여 데이터셋 다운로드 링크를 요청합니다.
   - 제공된 링크를 통해 다음 파일들을 다운로드합니다:
     - `Flickr8k_Dataset.zip` (이미지 파일)
     - `Flickr8k_text.zip` (캡션 및 데이터 분할 정보)
3. **데이터셋 압축 해제**:

   - 다운로드한 ZIP 파일을 `data/Flickr8k/` 디렉토리에 압축 해제합니다.
   - 디렉토리 구조는 다음과 같아야 합니다:

     ```
     data/
     └── Flickr8k/
         ├── images/  # 이미지 파일 (.jpg)
         ├── captions.txt
         ├── Flickr_8k.trainImages.txt
         ├── Flickr_8k.devImages.txt
         └── Flickr_8k.testImages.txt
     ```

### 2.2. COCO 형식의 주석 파일 생성

1. **주석 파일 생성 스크립트 실행**:

   ```bash
   python create_coco_annotations.py
   ```

   - 이 스크립트는 Flickr8k 데이터셋의 캡션과 이미지 정보를 COCO 데이터셋 형식의 JSON 파일로 변환합니다.

2. **생성된 파일 확인**:

   ```
   data/
   └── Flickr8k/
       ├── annotations/
       │   ├── flickr8k_train.json
       │   ├── flickr8k_val.json
       │   └── flickr8k_test.json
       └── ...
   ```

### 2.3. 파인튜닝 환경 설정

1. **필요한 패키지 설치**:

   ```bash
   pip install -r requirements.txt
   ```

   주요 패키지:

   - `torch`
   - `torchvision`
   - `ruamel.yaml`
   - `nltk`
   - `streamlit`
   - `fpdf`

2. **NLTK 데이터 다운로드**:

   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

### 2.4. 파인튜닝 실행

1. **체크포인트 디렉토리 설정**:

   - 사전 학습된 모델 체크포인트를 `checkpoints/` 디렉토리에 저장합니다.
   - 예시로 `blip_smile_base.pth` 파일을 해당 디렉토리에 위치시킵니다.

2. **파인튜닝 명령 실행**:

   ```bash
   python train_caption.py --config configs/caption_flickr8k.yaml --output_dir output/flickr8k
   ```

   - `--config`: 파인튜닝에 사용할 설정 파일 경로
   - `--output_dir`: 출력(모델 및 결과) 디렉토리 경로

3. **학습 진행 상황 확인**:
   - 학습 로그와 모델 체크포인트는 `output/flickr8k/` 디렉토리에 저장됩니다.
   - 학습 완료 후 `checkpoint_best.pth` 파일이 생성됩니다.

---

## 3. Streamlit을 사용한 데모 실행 방법

### 3.1. 환경 설정

1. **필요한 패키지 설치**:

   ```bash
   pip install streamlit fpdf
   ```

2. **모델 체크포인트 준비**:
   - 앞서 파인튜닝된 모델의 체크포인트 파일 `checkpoint_best.pth`를 `output/flickr8k/` 디렉토리에 위치시킵니다.

### 3.2. Streamlit 앱 실행

1. **`app.py` 파일 수정**:

   `app.py`에서 모델 경로를 파인튜닝된 모델로 설정합니다.

   ```python
   def load_model():
       image_size = 384
       model_path = 'output/flickr8k/checkpoint_best.pth'  # 학습된 모델의 경로
       model = caption_model(pretrained=model_path, image_size=image_size, vit='base')
       model.eval()
       return model
   ```

2. **앱 실행**:

   ```bash
   streamlit run app.py
   ```

3. **데모 사용**:
   - 웹 브라우저에서 `http://localhost:8501`로 접속합니다.
   - 이미지 업로드를 통해 캡션 생성 기능을 사용합니다.
   - 생성된 캡션과 함께 이미지를 포토북에 추가하고 PDF로 다운로드할 수 있습니다.

---

## 문의 사항

- 사용 중 문제가 발생하거나 개선 사항이 있으면 이슈를 등록해 주세요.
- 팀원들과의 협업을 위해 Pull Request를 통해 코드 변경 사항을 공유해 주세요.

---

## 참고 자료

- [BLIP 레포지토리](https://github.com/salesforce/BLIP)
- [SMILE 레포지토리](https://github.com/yuezih/SMILE)
- [Flickr8k 데이터셋 정보](https://forms.illinois.edu/sec/1713398)
- [Streamlit 공식 문서](https://streamlit.io/)
