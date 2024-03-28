

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
        

def calculate_top_k_accuracy(prediction_details, test_labels, k=5):
    correct = 0
    for details, true_label in zip(prediction_details, test_labels):
        found = False  # 정답 레이블을 찾았는지 여부
        for detail in details:  # 여기서 details는 각 이미지에 대한 예측 상세 정보의 리스트
            top_k_labels = detail['top_k_labels']
            if true_label in top_k_labels:
                correct += 1
                found = True
                break  # 현재 이미지에 대한 정답을 찾았으므로 더 이상 확인하지 않음
        if found:
            continue  # 다음 이미지로 넘어감

    accuracy = correct / len(test_labels) if test_labels else 0
    return accuracy





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

def save_predictions_and_labels(prediction_details, labels, image_paths, save_path, class_names):
    with open(save_path, 'w') as file:
        for details, label, image_path in zip(prediction_details, labels, image_paths):
            # img_name = os.path.basename(image_path)
            
            path_parts = image_path.replace("\\", "/").split("/")
            # 마지막 파일명과 그 앞 디렉토리 이름 가져오기
            if len(path_parts) > 1:
                modified_img_name = f"annotated_{path_parts[-2]}_{path_parts[-1]}"
            else:
                modified_img_name = f"annotated_{path_parts[-1]}"
            
            all_top_k_labels = []
            
            # detail에서 모든 예측된 top k 레이블을 수집합니다.
            for detail in details:
                if 'top_k_labels' in detail:
                    top_k_labels = detail['top_k_labels']
                    all_top_k_labels.extend(top_k_labels)
            
            # 중복을 제거하고 레이블 이름을 조회합니다.
            unique_top_k_labels = set(all_top_k_labels)
            predicted_labels_names = [class_names.get(str(l), "Unknown") for l in unique_top_k_labels]
            predicted_text = '/'.join(predicted_labels_names)
            
            actual_text = class_names.get(str(label), "Unknown")
            file.write(f"Image: {modified_img_name}, Predicted: {predicted_text}, Actual: {actual_text}\n")


import time
def save_prediction_images_with_labels(image_paths, prediction_details, labels, save_dir, class_names):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    font_path = "arial.ttf"
    font_size = 20  # 글자 크기를 조금 줄여 여러 라벨을 나열하기 용이하게 합니다.
    initial_text_height = 10  # 텍스트 시작 높이
    text_height_step = 25  # 다음 라벨을 그릴 때 높이 간격

    for img_path, details, true_label in zip(image_paths, prediction_details, labels):
        image = Image.open(img_path).convert("RGB")  # 그레이스케일 이미지도 컬러로 변환
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        # 모든 top k labels을 처리합니다.
        all_top_k_labels = []
        for detail in details:  # detail에는 여러 feature vector 정보가 포함됩니다.
            if 'top_k_labels' in detail:
                top_k_labels = detail['top_k_labels']
                all_top_k_labels.extend(top_k_labels)

        # 중복을 제거하고 모든 top k labels의 이름을 가져옵니다.
        unique_top_k_labels = set(all_top_k_labels)
        predicted_labels_names = [class_names.get(str(l), "Unknown") for l in unique_top_k_labels]
        
        true_label_name = class_names.get(str(true_label), "Unknown")

        # 실제 레이블이 예측된 top k labels 중 하나인지 확인
        match = true_label in unique_top_k_labels
        text_color = "green" if match else "red"

        # 실제 레이블 그리기
        draw.text((10, initial_text_height), f"Truth: {true_label_name}", fill=text_color, font=font)
        
        # 예측 레이블 그리기
        for i, label_name in enumerate(predicted_labels_names, start=1):
            draw.text((10, initial_text_height + text_height_step * i), f"Pred {i}: {label_name}", fill=text_color, font=font)

        # modified_img_name = f"annotated_{os.path.basename(img_path)}"
        path_parts = img_path.replace("\\", "/").split("/")
        # 마지막 파일명과 그 앞 디렉토리 이름 가져오기
        if len(path_parts) > 1:
            modified_img_name = f"annotated_{path_parts[-2]}_{path_parts[-1]}"
        else:
            modified_img_name = f"annotated_{path_parts[-1]}"
        
        save_path = os.path.join(save_dir, modified_img_name)
        image.save(save_path)

    print(f"All prediction images with labels have been saved to {save_dir}.")


def load_images_and_masks_class_wise(data_dir, split_ratio):
    image_dir = os.path.join(data_dir, 'test')
    mask_dir = os.path.join(data_dir, 'ground_truth')
    
    defect_classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)) and d != 'good']
    
    train_image_paths = []
    test_image_paths = []
    train_mask_paths = []
    test_mask_paths = []
    train_labels = []
    test_labels = []
    
    for defect_class in defect_classes:
        class_image_dir = os.path.join(image_dir, defect_class)
        class_mask_dir = os.path.join(mask_dir, defect_class)
        
        class_image_paths = []
        class_mask_paths = []
        class_labels = []
        
        for image_name in os.listdir(class_image_dir):
            if image_name.endswith('.png'):
                image_path = os.path.join(class_image_dir, image_name)
                mask_name = image_name.replace('.png', '_mask.png')
                mask_path = os.path.join(class_mask_dir, mask_name)
                
                class_image_paths.append(image_path)
                class_mask_paths.append(mask_path)
                class_labels.append(defect_classes.index(defect_class))
                
        # 각 클래스 내에서 데이터 분할
        class_train_image_paths, class_test_image_paths, class_train_mask_paths, class_test_mask_paths, class_train_labels, class_test_labels = train_test_split(
            class_image_paths, class_mask_paths, class_labels, test_size=1 - split_ratio, stratify=class_labels, random_state=None)
        
        # 분할된 데이터를 전체 목록에 추가
        train_image_paths.extend(class_train_image_paths)
        test_image_paths.extend(class_test_image_paths)
        train_mask_paths.extend(class_train_mask_paths)
        test_mask_paths.extend(class_test_mask_paths)
        train_labels.extend(class_train_labels)
        test_labels.extend(class_test_labels)
    
    return train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels


# def load_images_and_masks(data_dir, split_ratio):
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
    # print("Train Image Paths:", train_image_paths)
    # print("Test Image Paths:", test_image_paths)
    # print("Train Mask Paths:", train_mask_paths)
    # print("Test Mask Paths:", test_mask_paths)
    # print("Train Labels:", train_labels)
    # print("Test Labels:", test_labels)
    
    return train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels

import csv
def save_details_to_csv(prediction_details, test_image_paths, save_path):
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Image Name", "Top K Labels"])

        for detail, img_path in zip(prediction_details, test_image_paths):
            img_name = os.path.basename(img_path)
            top_k_labels_str = '/'.join(map(str, detail['top_k_labels']))
            writer.writerow([img_name, top_k_labels_str])
    
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
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths, train_labels, test_labels = load_images_and_masks_class_wise(config.data_dir, config.split_ratio)
    
    # if config.mode == 'train':
    process_train_images(train_image_paths, train_mask_paths, train_labels, config, device)
    ##  train mode end
    
    # elif config.mode == 'test':
        # Test mode does not use labels
    predictions, prediction_details = process_test_images(test_image_paths, test_mask_paths, config, device)
    
    # result_save_path = os.path.join(config.save_dir, 'prediction_details.csv')
    result_save_path = os.path.join(config.save_dir, config.detail_csv_name)
    # save_details_to_csv(prediction_details, test_image_paths, result_save_path)
    
    # for img_path, details in zip(test_image_paths, predictions_details):
        
    #save_details_to_csv(predictions_details, test_image_paths, "prediction_details.csv")
    # Claulate Accuracy
    accuracy = calculate_top_k_accuracy(prediction_details, test_labels, k=5)
    # accuracy = calculate_accuracy(predictions, test_labels)
    
    print("-----------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    accuracy_file_path = os.path.join(config.save_dir, 'accuracy.txt')

    with open(accuracy_file_path, 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("-----------")
    print('Save Accuracy text file!!')
    print("-----------")
    
    # Result Data Save
    # result_save_path = os.path.join(config.save_dir, 'predictions_and_labels.txt')
    result_save_path = os.path.join(config.save_dir, config.prediction_txt_name)
    
    # predictions 대신 prediction_details를 전달합니다.
    save_predictions_and_labels(prediction_details, test_labels, test_image_paths, result_save_path, config.class_names)
    save_prediction_images_with_labels(test_image_paths, prediction_details, test_labels, config.save_dir, config.class_names)
    
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

        save_dir = os.path.join(subdir_path, f"{subdir}_results")
        # save_dir = os.path.join(subdir_path, "results")
        memory_bank_path = os.path.join(save_dir, f"{subdir}_memory_bank.pt")
        
        # 클래스 이름을 test 디렉토리를 기반으로 자동으로 생성
        class_names = get_class_names(test_dir)
        class_names[-1] = "Unknown"
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