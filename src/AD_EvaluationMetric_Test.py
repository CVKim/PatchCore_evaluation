import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torchvision import models, transforms

from torchvision.utils import save_image

from typing import List

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_embedder import FeatureEmbedder, preprocess_image, preprocess_mask, load_memory_bank, label_feature_patches, save_memory_bank, predict_test_image_class, extract_features_from_image, create_feature_embedder, apply_mask_to_features, predict_labels_from_features


class Config:
    def __init__(self, mode='train', memory_bank_path='memory_bank.pt', resize=(256, 256), crop_size=(224, 224),  class_names=None,
                 data_dir='F:/Dataset/mvtec_anomaly_detection/bottle', split_ratio=0.8, save_dir='F:/Dataset/mvtec_anomaly_detection/bottle',
                 input_shape = [3,244,244]):
        # self.class_names = class_names or {'0': 'crack', '1': 'cut', '2': 'hole', '3': 'print'}
        self.class_names = class_names or {'0': 'broken_large', '1': 'broken_small', '2': 'contamination'}
        self.mode = mode
        self.memory_bank_path = memory_bank_path
        self.resize = resize
        self.crop_size = crop_size
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.save_dir = save_dir
        self.input_shape = input_shape
        
        last_folder_name = os.path.basename(os.path.normpath(data_dir))
        self.memory_bank_path = os.path.join(save_dir, f"{last_folder_name}_memory_bank.pt")
        self.detail_csv_name = f"{last_folder_name}_prediction_details.csv"
        self.prediction_txt_name = f"{last_folder_name}_predictions_and_labels.txt"
        

def calculate_accuracy(predictions, labels):
    """
    예측값과 실제 레이블을 기반으로 정확도를 계산합니다.

    Args:
    - predictions (list): 예측된 클래스 ID 목록
    - labels (list): 실제 클래스 ID 목록

    Returns:
    - accuracy (float): 정확도
    """
    
    # 라이브러리 사용 안 할 거면 아래 처럼 사용 가능
    # correct = sum(p == t for p, t in zip(predictions, labels))
    # return correct / len(predictions) if predictions else 0
    
    
    accuracy = accuracy_score(labels, predictions)
    return accuracy

def save_predictions_and_labels(predictions, labels, image_paths, save_path):
    """
    예측값, 실제 레이블, 이미지 이름을 파일로 저장합니다.

    Args:
    - predictions (list): 예측된 클래스 ID 목록
    - labels (list): 실제 클래스 ID 목록
    - image_paths (list of str): 이미지 경로 목록
    - save_path (str): 저장할 파일 경로
    """
    with open(save_path, 'w') as file:
        for prediction, label, image_path in zip(predictions, labels, image_paths):
            img_name = os.path.basename(image_path)
            file.write(f"Image: {img_name}, Predicted: {prediction}, Actual: {label}\n")


import time
def save_prediction_images_with_labels(image_paths, predictions, labels, save_dir, config):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    font_path = "arial.ttf"
    font_size = 30
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Font file not found. Using default font.")
        font = ImageFont.load_default()

    for img_path, prediction, label in zip(image_paths, predictions, labels):
        img_name = os.path.basename(img_path)
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        
        # 클래스 이름 사용
        prediction_name = config.class_names.get(str(prediction), "Unknown")
        label_name = config.class_names.get(str(label), "Unknown")

        text = f"Pred: {prediction_name} ({prediction}), Truth: {label_name} ({label}), Image: {img_name}"
        # pred / label 결과가 동일한 경우 초록색, 다른 경우 빨간색으로 표시
        fill_color = "green" if prediction == label else "red"
        draw.text((10, 10), text, fill=fill_color, font=font)

        pred_file_name = f"{img_name[:-4]}_pred_{prediction_name}_true_{label_name}.png"
        image.save(os.path.join(save_dir, pred_file_name))
    
    print(f"Saved prediction images with labels in {save_dir}")

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
    
    # 결함 클래스 동적 생성: 'good' 폴더 제외하고, 'test' 폴더 내의 모든 서브디렉토리를 결함 클래스로 사용
    defect_classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)) and d != 'good']
    
    # defect_classes = ['broken_large', 'broken_small', 'contamination']

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
                
                mask_name = image_name.replace('.png', '_mask.png')
                mask_path = os.path.join(class_mask_dir, mask_name)

                image_paths.append(image_path)
                mask_paths.append(mask_path)
                labels.append(defect_classes.index(defect_class))

    # 데이터셋 분할
    # train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels = train_test_split(image_paths, mask_paths, labels, test_size=1 - split_ratio, stratify=labels)
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels = train_test_split(
    image_paths, mask_paths, labels, test_size=1 - split_ratio, stratify=labels, random_state=None)

    # 결과 출력
    print("Train Image Paths:", train_image_paths)
    print("Test Image Paths:", test_image_paths)
    print("Train Mask Paths:", train_mask_paths)
    print("Test Mask Paths:", test_mask_paths)
    print("Train Labels:", train_labels)
    print("Test Labels:", test_labels)
    
    return train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels

def save_details_to_csv(predictions_details, test_image_paths, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # CSV 파일 헤더 작성
        writer.writerow(["Image Name", "Test Vector", "Nearest Vector", "Nearest Distance", "Nearest Label"])
        
        # 각 테스트 이미지와 상세 정보를 CSV 파일에 기록
        for detail, img_path in zip(predictions_details, test_image_paths):
            img_name = os.path.basename(img_path)
            for d in detail:  # 여기서 detail은 하나의 이미지에 대한 여러 상세 정보를 담은 리스트입니다.
                writer.writerow([
                    img_name,
                    str(d['test_vector']),
                    str(d['nearest_vector']),
                    d['nearest_distance'],
                    d['nearest_label']
                ])
    print(f"Prediction details saved to {save_path}")

def process_train_images(image_paths, mask_paths, labels, config, device):
    memory_bank = {'features': [], 'labels': []}
    feature_embedder = create_feature_embedder(device, config.input_shape)
    
    for img_path, mask_path, label in zip(image_paths, mask_paths, labels):
        image_tensor = preprocess_image(img_path, config.resize, config.crop_size).to(device)
        mask_tensor = preprocess_mask(mask_path, config.resize, config.crop_size).to(device)
        
        # Extract and process features
        features = extract_features_from_image(image_tensor, device, feature_embedder, config.input_shape)
        # Apply mask and label to the features
        memory_bank = label_feature_patches(features, mask_tensor, label, memory_bank)
    
    results_dir = os.path.dirname(config.memory_bank_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save the constructed memory bank
    save_memory_bank(memory_bank, config.memory_bank_path)
    print("Memory bank constructed and saved.")

def process_test_images(image_paths, mask_paths, config, device):
    # Load memory bank
    memory_bank = load_memory_bank(config.memory_bank_path)
    feature_embedder = create_feature_embedder(device, config.input_shape)
    
    predictions = []
    prediction_details = []  # 각 예측에 대한 추가 정보를 저장합니다.

    for img_path, mask_path in zip(image_paths, mask_paths):
        image_tensor = preprocess_image(img_path, config.resize, config.crop_size).to(device)
        mask_tensor = preprocess_mask(mask_path, config.resize, config.crop_size).to(device)
        
        # Extract features
        features = extract_features_from_image(image_tensor, device, feature_embedder, config.input_shape)
        
        # Apply mask and get masked features
        masked_features = apply_mask_to_features(features, mask_tensor)
        
        # Predict class and get details using memory bank
        predicted_label, details = predict_labels_from_features(masked_features, memory_bank)
        
        predictions.append(predicted_label)
        prediction_details.append(details)  # 예측 상세 정보를 추가합니다.
    
    return predictions, prediction_details

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels = load_images_and_masks(config.data_dir, config.split_ratio)
    
    # if config.mode == 'train':
    process_train_images(train_image_paths, train_mask_paths, train_labels, config, device)
    ##  train mode end
    
    # elif config.mode == 'test':
        # Test mode does not use labels
    predictions, prediction_details = process_test_images(test_image_paths, test_mask_paths, config, device)
    
    # result_save_path = os.path.join(config.save_dir, 'prediction_details.csv')
    result_save_path = os.path.join(config.save_dir, config.detail_csv_name)
    save_details_to_csv(prediction_details, test_image_paths, result_save_path)
    
    # for img_path, details in zip(test_image_paths, predictions_details):
        
    #save_details_to_csv(predictions_details, test_image_paths, "prediction_details.csv")
    # Claulate Accuracy
    accuracy = calculate_accuracy(predictions, test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Result Data Save
    # result_save_path = os.path.join(config.save_dir, 'predictions_and_labels.txt')
    result_save_path = os.path.join(config.save_dir, config.prediction_txt_name)
    
    save_predictions_and_labels(predictions, test_labels, test_image_paths, result_save_path)
    save_prediction_images_with_labels(test_image_paths, predictions, test_labels, config.save_dir, config)
    
    print(f"Predictions and labels have been saved to {result_save_path}")

def get_class_names(test_dir):
    # "good" 폴더를 제외한 모든 서브디렉토리를 클래스 이름으로 사용
    class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)) and d != "good"]
    return {str(idx): class_name for idx, class_name in enumerate(class_dirs)}

from pathlib import Path
def process_directory(data_dir, mode='test', split_ratio=0.8):
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        test_dir = os.path.join(subdir_path, "test")
        print(f"Processing {subdir_path}...")

        save_dir = os.path.join(subdir_path, "results")
        memory_bank_path = os.path.join(save_dir, f"{subdir}_memory_bank.pt")
        
        # 클래스 이름을 test 디렉토리를 기반으로 자동으로 생성
        class_names = get_class_names(test_dir)
        
        config = Config(
            mode=mode,
            memory_bank_path=memory_bank_path,
            resize=(256, 256),
            crop_size=(224, 224),
            class_names=class_names,
            data_dir=subdir_path,
            split_ratio=split_ratio,
            save_dir=save_dir,
            input_shape=[3, 244, 244]
        )

        start_time = time.time()
        main(config=config)
        end_time = time.time()
        print(f"Processed {subdir} in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # class_names = {'0': 'crack', '1': 'cut', '2': 'hole', '3': 'print'}
    # class_names = {'0': 'broken_large', '1': 'broken_small', '2': 'contamination'}
    start_time = time.time()  # 시작 시간 기록
    # config = Config(mode='test')
    # main(config=config)
    data_dir = 'F:/Dataset/mvtec_anomaly_detection'
    process_directory(data_dir)
    end_time = time.time()  # 종료 시간 기록
    print(f"Processed time : {end_time - start_time:.2f} seconds.")