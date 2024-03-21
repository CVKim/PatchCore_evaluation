
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torchvision import models, transforms

from torchvision.utils import save_image

from typing import List

from PIL import Image
import numpy as np

import os
import random
from sklearn.model_selection import train_test_split
from feature_embedder import FeatureEmbedder, preprocess_image, preprocess_mask, load_memory_bank, label_feature_patches, save_memory_bank, extract_features_from_test_image, predict_test_image_class, extract_features_from_image, create_feature_embedder

class Config:
    def __init__(self, mode='train', memory_bank_path='memory_bank.pt', resize=(256, 256), crop_size=(224, 224), data_dir='F:/Dataset/mvtec_anomaly_detection/hazelnut', split_ratio=0.8, save_dir='path/to/save/results'):
        self.mode = mode
        self.memory_bank_path = memory_bank_path
        self.resize = resize
        self.crop_size = crop_size
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.save_dir = save_dir

def save_prediction_images(image_paths, predictions, save_dir):
    """
    이미지 경로와 예측된 클래스를 기반으로 예측 결과 이미지를 저장합니다.

    Args:
    - image_paths (list of str): 테스트 이미지 경로 목록
    - predictions (list of int): 각 이미지에 대한 예측된 클래스 목록
    - save_dir (str): 결과 이미지를 저장할 디렉토리 경로
    """
    # 결과 저장 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for img_path, pred_class in zip(image_paths, predictions):
        # 이미지 로드
        image = Image.open(img_path)
        
        # 파일명 구성: 원본 파일명에 예측 클래스 정보 추가
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        pred_file_name = f"{name}_pred_{pred_class}{ext}"
        
        # 예측 결과 이미지 저장 경로
        save_path = os.path.join(save_dir, pred_file_name)
        
        # 이미지 저장
        image.save(save_path)

    print(f"All prediction images have been saved to {save_dir}.")

def load_images_and_masks(data_dir, split_ratio):
    """
    이미지와 마스크 파일 경로를 로드하고, train/test로 분할합니다.

    Args:
    - data_dir (str): 데이터셋의 루트 디렉토리 경로
    - split_ratio (float): 트레이닝 데이터에 사용할 비율

    Returns:
    - train_image_paths (list): 트레이닝 이미지 경로 목록
    - test_image_paths (list): 테스트 이미지 경로 목록
    - train_mask_paths (list): 트레이닝 마스크 경로 목록
    - test_mask_paths (list): 테스트 마스크 경로 목록
    """
    image_dir = os.path.join(data_dir, 'test')
    mask_dir = os.path.join(data_dir, 'ground_truth')
    defect_classes = ['crack', 'cut', 'hole', 'print']

    image_paths = []
    mask_paths = []
    labels = []

    # 각 결함 클래스별로 이미지와 마스크 파일 경로 수집
    for defect_class in defect_classes:
        class_image_dir = os.path.join(image_dir, defect_class)
        class_mask_dir = os.path.join(mask_dir, defect_class)
        for image_name in os.listdir(class_image_dir):
            if image_name.endswith('.png'):
                image_path = os.path.join(class_image_dir, image_name)
                mask_path = os.path.join(class_mask_dir, image_name)

                image_paths.append(image_path)
                mask_paths.append(mask_path)
                labels.append(defect_classes.index(defect_class))

    # 데이터셋 분할
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels = train_test_split(image_paths, mask_paths, labels, test_size=1 - split_ratio, stratify=labels)

    return train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels


def process_train_images(image_paths, mask_paths, labels, config, device):
    memory_bank = {'features': [], 'labels': []}
    feature_embedder = create_feature_embedder(device, config.input_shape)

    for img_path, mask_path, label in zip(image_paths, mask_paths, labels):
        image_tensor = preprocess_image(img_path, config.resize, config.crop_size).to(device)
        mask_tensor = preprocess_mask(mask_path, config.resize, config.crop_size).to(device)
        
        # Extract and process features
        features = extract_features_from_image(image_tensor, feature_embedder)
        # Apply mask and label to the features
        memory_bank = label_feature_patches(features, mask_tensor, label, memory_bank)
    
    # Save the constructed memory bank
    save_memory_bank(memory_bank, config.memory_bank_path)
    print("Memory bank constructed and saved.")

def process_test_images(image_paths, mask_paths, config, device):
    # Load memory bank
    memory_bank = load_memory_bank(config.memory_bank_path)
    feature_embedder = create_feature_embedder(device, config.input_shape)
    predictions = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        image_tensor = preprocess_image(img_path, config.resize, config.crop_size).to(device)
        mask_tensor = preprocess_mask(mask_path, config.resize, config.crop_size).to(device)
        
        # Extract features
        features = extract_features_from_image(image_tensor, feature_embedder)
        # Predict class using memory bank
        predicted_class = predict_test_image_class(features, mask_tensor, memory_bank)
        predictions.append(predicted_class)
    
    # Here, you can compare the predictions with the ground truth and calculate accuracy.
    # Also, you can save the predictions to a file for further analysis.
    return predictions


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths, mask_paths, labels = load_images_and_masks(config.data_dir, config.split_ratio)
    
    if config.mode == 'train':
        train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, _ = image_paths
        process_train_images(train_image_paths, train_mask_paths, train_labels, config, device)
    elif config.mode == 'test':
        # Test mode does not use labels
        _, test_image_paths, _, test_mask_paths = image_paths
        predictions = process_test_images(test_image_paths, test_mask_paths, config, device)
    
    # if config.mode == 'train':
    #     # Train 모드일 때의 처리
    #     # 예시로, 한 개의 이미지와 마스크로 메모리 뱅크를 생성하고 저장하는 과정
    #     image_path = "path/to/your/train_image.png"
    #     mask_path = "path/to/your/train_mask.png"
    #     image_tensor = preprocess_image(image_path, config.resize, config.crop_size).to(device)
    #     mask_tensor = preprocess_mask(mask_path, config.resize, config.crop_size).to(device)
    #     memory_bank = label_feature_patches(image_tensor, mask_tensor, device, config.crop_size)
    #     save_memory_bank(memory_bank, config.memory_bank_path)
    #     print("Train mode completed: Memory bank constructed and saved.")
        
    # elif config.mode == 'test':
    #         # Test 모드일 때의 처리
    #         test_image_path = "000.png"
    #         test_mask_path = "000_mask.png"  # 테스트 모드용 마스크 이미지 경로 추가
    #         test_image_tensor = preprocess_image(test_image_path, config.resize, config.crop_size).to(device)
    #         test_mask_tensor = preprocess_mask(test_mask_path, config.resize, config.crop_size).to(device)  # 마스크 이미지 전처리
            
    #         test_features = extract_features_from_test_image(test_image_path, device, config.resize, config.crop_size)
            
    #         # 메모리 뱅크 로드
    #         loaded_memory_bank = load_memory_bank(config.memory_bank_path)
            
    #         # 테스트 이미지에 대한 클래스 예측
    #         # 마스크 텐서를 추가적으로 전달
    #         predicted_class = predict_test_image_class(test_features, test_mask_tensor, memory_bank=loaded_memory_bank)
    #         print(f"Test mode completed: Predicted class for the test image is: {predicted_class}")

if __name__ == "__main__":
    config = Config(mode='train')
    main(config=config)
