# app.py

import streamlit as st
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from fpdf import FPDF  # PDF 생성용 라이브러리
import os

# 모델 로딩 함수 (학습된 모델을 사용한다고 가정)
from models.model import caption_model  # SMILE 모델을 import 합니다

# SMILE 모델 초기화
@st.cache_resource
# def load_model():
#     image_size = 384
#     model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
#     model = caption_model( image_size=image_size, vit='base')
#     # model = caption_model(pretrained=model_url, image_size=image_size, vit='base')
#     model.eval()
#     return model

def load_model():
    image_size = 384
    model_path = 'output/flickr8k/checkpoint_best.pth'  # 학습된 모델의 경로
    model = caption_model(pretrained=model_path, image_size=image_size, vit='base')
    model.eval()
    return model


model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 로드 및 전처리 함수
def load_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# Streamlit 애플리케이션 구성
st.title("SMILE 기반 이미지 캡셔닝")
st.write("이미지를 업로드하고, SMILE 모델이 생성한 설명력 높은 캡션을 확인하세요.")

# 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_column_width=True)
    
    # 캡션 생성 버튼
    if st.button("캡션 생성"):
        with st.spinner("캡션을 생성하는 중..."):
            input_image = load_image(image, device)
            model.to(device)
            with torch.no_grad():
               caption = model.generate(input_image, sample=False, top_p=0.9, max_length=50, min_length=20, repetition_penalty=1.9) 
            st.write("### 생성된 캡션")
            st.write(caption[0])

# 포토북 PDF 생성
captions = []
if uploaded_file and st.button("포토북 PDF에 추가"):
    captions.append((image, caption[0]))
    st.write("이미지와 캡션이 포토북에 추가되었습니다.")

if captions and st.button("PDF 다운로드"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # PDF에 이미지와 캡션 추가
    for img, cap in captions:
        pdf.add_page()
        pdf.image(img, x=10, y=10, w=180)
        pdf.ln(85)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, cap)
    
    pdf_file_path = "photobook.pdf"
    pdf.output(pdf_file_path)

    with open(pdf_file_path, "rb") as pdf_file:
        st.download_button("포토북 PDF 다운로드", pdf_file, "photobook.pdf", "application/pdf")
    
    # 파일 삭제
    os.remove(pdf_file_path)
