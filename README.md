# SMILE-based Image Captioning Project

This project is based on the [BLIP](https://github.com/salesforce/BLIP) and [SMILE](https://github.com/yuezih/SMILE) repositories. It fine-tunes an image captioning model using the Flickr8k dataset and provides a demo through Streamlit.

## Table of Contents

1. [Project Modifications](#1-project-modifications)
2. [Flickr8k Dataset Download and Fine-tuning Process](#2-flickr8k-dataset-download-and-fine-tuning-process)
3. [Running the Demo with Streamlit](#3-running-the-demo-with-streamlit)

---

## 1. Project Modifications

The following changes were made from the [original GitHub repositories](https://github.com/yuezih/SMILE):

- **Dataset Support**: Replaced the COCO dataset with the **Flickr8k dataset**.
- **Configuration File Updates**: Added the `configs/caption_flickr8k.yaml` file for configurations tailored to the Flickr8k dataset.
- **Custom Data Loader**: Added the `data/flickr8k_dataset.py` file to handle the Flickr8k dataset.
- **Evaluation Function Modifications**: Adjusted evaluation methods for the Flickr8k dataset.
- **Fine-tuning Script Modifications**: Updated `train_caption.py` to support fine-tuning with the Flickr8k dataset.
- **Streamlit Demo**: Added the `app.py` file to provide an image captioning demo using the trained model.

---

## 2. Flickr8k Dataset Download and Fine-tuning Process

### 2.1. Downloading Checkpoint Model and Flickr8k Dataset

1. **Download the Checkpoint Model**: [blip_smile_base.pth](https://huggingface.co/spaces/yuezih/BLIP-SMILE/tree/main/model)
2. **Place the Checkpoint Model**:
   - Save the downloaded `blip_smile_base.pth` file at the path `checkpoints/blip_base.pth`.
3. **Visit the Flickr8k Dataset Page**: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
4. **Download the Dataset**:

   - Save the following items after clicking the download button:
     - `images` (Image folder containing 8,091 files)
     - `captions.txt` (Caption information)

5. **Extract the Dataset**:

   - Extract the ZIP file into the `data/Flickr8k/` directory.
   - Ensure the directory structure is as follows:

     ```
     data/
     └── Flickr8k/
         ├── images/  # Image files (.jpg)
         ├── captions.txt
         ├── Flickr_8k.trainImages.txt
         ├── Flickr_8k.devImages.txt
         └── Flickr_8k.testImages.txt
     ```

### 2.2. Generating COCO-style Annotation Files

1. **Run the Annotation Generation Script**:

   ```bash
   python create_coco_annotations.py
   ```

   - This script converts the captions and image information of the Flickr8k dataset into COCO-style JSON files.

2. **Verify Generated Files**:

   ```
   data/
   └── Flickr8k/
       ├── annotations/
       │   ├── flickr8k_train.json
       │   ├── flickr8k_val.json
       │   └── flickr8k_test.json
       └── ...
   ```

### 2.3. Setting Up the Fine-tuning Environment

1. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

   Key packages include:

   - `torch`
   - `torchvision`
   - `ruamel.yaml`
   - `nltk`

### 2.4. Running the Fine-tuning

1. **Set Up the Checkpoint Directory**:

   - Save the pre-trained model checkpoint in the `checkpoints/` directory.
   - For example, place the `blip_smile_base.pth` file in this directory.

2. **Run the Fine-tuning Command**:

   ```bash
   python train_caption.py --config configs/caption_flickr8k.yaml --output_dir output/flickr8k
   ```

   - `--config`: Path to the configuration file for fine-tuning
   - `--output_dir`: Path to the output directory (for models and results)

3. **Monitor Training Progress**:
   - Training logs and model checkpoints will be saved in the `output/flickr8k/` directory.
   - The `checkpoint_best.pth` file will be created upon training completion.

---

## 3. Running the Demo with Streamlit

### 3.1. Setting Up the Environment

1. **Prepare the Model Checkpoint**:
   - Place the fine-tuned model checkpoint file `checkpoint_best.pth` in the `output/flickr8k/` directory.

### 3.2. Running the Streamlit App

1. **Modify the `app.py` File**:

   Set the model path in `app.py` to the fine-tuned model.

   ```python
   def load_model():
       image_size = 384
       model_path = 'output/flickr8k/checkpoint_best.pth'  # Path to the fine-tuned model
       model = caption_model(pretrained=model_path, image_size=image_size, vit='base')
       model.eval()
       return model
   ```

2. **Run the App**:

   ```bash
   streamlit run app.py
   ```

3. **Using the Demo**:
   - Access the app at `http://localhost:8501` in your web browser.
   - Upload images to generate captions.
   - Add generated captions and images to a photo book and download it as a PDF.

---

## References

- [BLIP Repository](https://github.com/salesforce/BLIP)
- [SMILE Repository](https://github.com/yuezih/SMILE)
- [Flickr8k Dataset Information](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Streamlit Documentation](https://streamlit.io/)
