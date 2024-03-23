import torch
import torch.nn.functional as F
from typing import List
import csv
import scipy
import torch.nn as nn
from torchvision import models, transforms

from patch_maker import PatchMaker
from feature_aggregator import NetworkFeatureAggregator
from common import Preprocessing, Aggregator

from PIL import Image
from torch.nn.functional import cosine_similarity

import numpy as np
from scipy.spatial import distance
from collections import Counter

def calculate_centroids(features):
    """각 불량 타입별 특징 벡터의 평균(centroid)을 계산합니다."""
    centroids = {}
    for defect_type, feature_vectors in features.items():
        centroids[defect_type] = torch.mean(torch.stack(feature_vectors), dim=0)
    return centroids

def predict_defect_type(new_feature, centroids):
    """새로운 이미지의 특징 벡터와 각 centroid 간의 코사인 유사도를 기반으로 불량 타입을 예측합니다."""
    similarities = {defect_type: cosine_similarity(new_feature, centroid, dim=0) for defect_type, centroid in centroids.items()}
    predicted_defect_type = max(similarities, key=similarities.get)
    return predicted_defect_type

class FeatureEmbedder(torch.nn.Module):
    def __init__(self, 
                 device,
                 input_shape,
                 backbone,
                 layers_to_extract_from,
                 pretrain_embed_dimension=1024,
                 target_embed_dimension=1024,
                 patchsize=3,
                 patchstride=1,
                 class_embedding_layer_to_extract_from: List[str] = None):
        super(FeatureEmbedder, self).__init__()

        self.device = device
        self.backbone = backbone

        self.input_shape = input_shape
        self.layers_to_extract_from = layers_to_extract_from

        self.patch_maker = PatchMaker(patchsize, patchstride)

        layers_to_extract_from = self.layers_to_extract_from
        
        if class_embedding_layer_to_extract_from:
            layers_to_extract_from += class_embedding_layer_to_extract_from

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)[:len(layers_to_extract_from)]

        preprocessing = Preprocessing( # layer에서 나온 놈
            feature_dimensions, pretrain_embed_dimension
        )
        
        preadapt_aggregator = Aggregator( # coreset subsampling 통해 나온 놈
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules = torch.nn.ModuleDict({})
        self.forward_modules["feature_aggregator"] = feature_aggregator
        self.forward_modules["preprocessing"] = preprocessing
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.feature_map_shape = self._compute_feature_map_shape()
        self.target_embed_dimension = target_embed_dimension
    
    @torch.no_grad()
    def forward(self, images, detach=True, provide_patch_shapes=False) -> torch.Tensor:
        """Returns feature embeddings for images."""

        images = images.to(torch.float).to(self.device)

        def _detach(features):
            if detach:
                return features.detach().cpu()
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]   # [(b, c, #p, #p), ..]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )   # (batch, #patch, #patch, channel, kernel, kernel))
            _features = _features.permute(0, -3, -2, -1, 1, 2)   # (batch, channel, kernel, kernel, #patch, #patch)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)   # (batch, #patch, #patch, channel, kernel, kernel)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])  # (batch, #patch*#patch, channel, kernel, kernel)
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]   # list of (#total, channel, kernel, kernel)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)   # (#total, #layers, designated_channel -> pretrain_embed_dimension)
        features = self.forward_modules["preadapt_aggregator"](features)   # (#total, designated_channel -> target_embed_dimension) (layers are averaged)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
    
    def _compute_feature_map_shape(self):
        _input = torch.ones([1] + list(self.input_shape)).to(self.device)
        dummy_feas, feature_map_shapes = self(_input, provide_patch_shapes=True)
        return feature_map_shapes[0] + [dummy_feas.shape[-1]]
    
    def get_feature_map_shape(self):
        return self.feature_map_shape

def preprocess_image(image_path, resize, crop_size):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # (1, C, H, W) 형태로 변환
    return image_tensor

# 마스크 전처리 함수: 리사이즈 후 센터 크롭 적용
def preprocess_mask(mask_path, resize, crop_size):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.# `extract_features_from_image` is a function that takes an image tensor, device,
        # feature embedder, and input shape as input parameters. It processes the input
        # image tensor to extract features using the provided feature embedder model. The
        # extracted features are then returned for further processing or analysis.
        # `extract_features_from_image` is a function that takes an image tensor, device,
        # feature embedder, and input shape as input parameters. It processes the input
        # image tensor to extract features using the provided feature embedder model. The
        # function returns the extracted features as a tensor.
        CenterCrop(crop_size),
        transforms.ToTensor()
    ])
    mask = Image.open(mask_path).convert("L")  # Grayscale로 변환
    mask_tensor = transform(mask).unsqueeze(0)  # (1, C, H, W) 형태로 변환
    return mask_tensor

# 새로운 이미지에 대한 특징 벡터를 추출하고, 불량 타입을 예측하는 함수
def analyze_defect(image_path, device, input_shape, centroids):
    new_features = extract_features_from_image(image_path, device, input_shape)
    predicted_defect_type = predict_defect_type(new_features, centroids)
    is_defective = "Defective" if predicted_defect_type else "Normal"  # 정상/불량 판정 여부
    
    return {
        "defect_type": predicted_defect_type,
        "features": new_features.cpu().numpy(),
        "is_defective": is_defective
    }

def create_feature_embedder(device, input_shape):
    
    backbone = models.wide_resnet50_2(pretrained=True)
    layers_to_extract_from = ['layer2', 'layer3'] # ?? lyaer를 몇으로 사용??
    feature_embedder = FeatureEmbedder(
        device=device,
        input_shape=input_shape,
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        pretrain_embed_dimension=1024, # resnet 50?? resnet 50 마지막 차원은 2048
        target_embed_dimension=1024
    ).to(device)
    return feature_embedder

def extract_features_from_image(input_tensor, device, feature_embedder, input_shape):
    
    # image_tensor = preprocess_image(image_path, input_shape)
    
    # feature_embedder = create_feature_embedder(device, input_shape)
    feature_embedder.eval() # evaluation mode
    
    with torch.no_grad():
        features = feature_embedder(input_tensor.to(device))
    return features


def get_labels_for_masked_features(feature_vectors):
    # Placeholder function for labeling feature vectors.
    # Replace this with your actual method for labeling.
    labels = torch.full((feature_vectors.size(0),), fill_value=0)  # Dummy example label
    return labels

def label_feature_patches(features, mask_tensor, label, memory_bank):
    """
    Apply mask to features, label them, and update the memory bank.
    
    Args:
    - features (Tensor): The feature tensor of shape [784, 1024].
    - mask_tensor (Tensor): The mask tensor of shape [1, 1, H, W].
    - label (int): The label for masked features.
    - memory_bank (dict): The memory bank to update.
    
    Returns:
    - memory_bank (dict): Updated memory bank.
    """
    # Resize the mask to match the shape of features
    # Assuming original mask is [1, 1, H, W] and we want to resize it to [28, 28]
    mask_resized = torch.nn.functional.interpolate(mask_tensor, size=(28, 28), mode='nearest').squeeze()
    # Flatten the mask to match the first dimension of features
    mask_flat = mask_resized.view(-1)

    # Select features based on the mask
    masked_features = features[mask_flat > 0, :]  # Select rows where mask is positive

    # Convert to numpy arrays for compatibility with memory bank
    masked_features_np = masked_features.cpu().numpy()
    labels_np = np.full(len(masked_features_np), fill_value=label, dtype=np.int64)  # Create labels array

    # Initialize or update memory bank
    if 'features' not in memory_bank or not len(memory_bank['features']):
        memory_bank['features'] = masked_features_np
        memory_bank['labels'] = labels_np
    else:
        # Concatenate new features and labels with existing ones in memory bank
        memory_bank['features'] = np.concatenate((memory_bank['features'], masked_features_np), axis=0)
        memory_bank['labels'] = np.concatenate((memory_bank['labels'], labels_np), axis=0)

    return memory_bank


def apply_mask_to_features(features, mask):
    # 마스크의 차원을 특성 벡터의 개수에 맞춰 조정합니다.
    mask_resized = F.interpolate(mask, size=(28, 28), mode='nearest').squeeze()
    mask_flat = mask_resized.view(-1).bool()

    # 마스크에 따라 특성 벡터를 선택합니다.
    masked_features = features[mask_flat]

    return masked_features



# def label_feature_patches(features, mask_tensor, label, memory_bank):
#     # Create and prepare the feature embedder
#     # feature_embedder = create_feature_embedder(device, input_shape)
#     # feature_embedder.eval()  # evaluation mode
    
#     # Reshape to (1, 28, 28, 1024)
#     reshaped_features = features.view(1, 28, 28, -1)

#     # Resize mask to match feature dimensions (1, 28, 28)
#     resized_mask = F.interpolate(mask_tensor, size=(28, 28), mode='nearest')

#     # Make sure that mask is broadcastable to the features size
#     mask_broadcastable = resized_mask.unsqueeze(-1)
    
#     # Apply the mask to the features
#     masked_features = reshaped_features * mask_broadcastable
    
#     # Find indices where the mask is positive
#     positive_mask_indices = mask_broadcastable.squeeze().nonzero(as_tuple=False)
    
#     # Gather the feature vectors corresponding to the mask
#     masked_feature_vectors = reshaped_features[0, positive_mask_indices[:, 0], positive_mask_indices[:, 1], :]
    
#     # Here you would label the masked_feature_vectors based on your labeling logic
#     # For simplicity, we're just using the provided label for all vectors
#     labels = torch.full((masked_feature_vectors.size(0),), fill_value=label, dtype=torch.long)
    
#     # Append to the memory bank
#     memory_bank['features'].append(masked_feature_vectors.cpu().numpy())
#     memory_bank['labels'].append(labels.cpu().numpy())

#     return memory_bank
    
    # with torch.no_grad():
    #     # Extract features from the image
    #     features = feature_embedder(image_tensor.to(device))
        
    #     # Reshape to (1, 28, 28, 1024)
    #     reshaped_features = features.view(1, 28, 28, -1)

    #     # Resize mask to match feature dimensions (1, 28, 28)
    #     resized_mask = F.interpolate(mask_tensor, size=(28, 28), mode='nearest').to(device)

    #     # Make sure that mask is broadcastable to the features size
    #     # We need the mask to be of size (1, 28, 28, 1) to broadcast along the channel dimension
    #     mask_broadcastable = resized_mask.unsqueeze(-1)
        
    #     # Apply the mask to the features
    #     # We use broadcasting here: the mask will automatically be expanded to match the features' dimensions
    #     masked_features = reshaped_features * mask_broadcastable
        
    #     # Find indices where the mask is positive
    #     positive_mask_indices = mask_broadcastable.squeeze().nonzero(as_tuple=False)
        
    #     # Gather the feature vectors corresponding to the mask
    #     # We only take feature vectors where the mask is non-zero
    #     masked_feature_vectors = reshaped_features[0, positive_mask_indices[:, 0], positive_mask_indices[:, 1], :]
        
    #     # Obtain labels for the masked feature vectors
    #     labels = get_labels_for_masked_features(masked_feature_vectors)
        
    #     # Construct the memory bank with feature vectors and labels
    #     memory_bank = {'features': masked_feature_vectors.cpu().numpy(), 'labels': labels.cpu().numpy()}
        
    #     return memory_bank

# 테스트 이미지의 특성을 추출하는 함수 (이전에 정의된 가정)
def extract_features_from_test_image(image_path, device, resize, crop_size):
    image_tensor = preprocess_image(image_path, resize, crop_size)
    feature_embedder = create_feature_embedder(device, input_shape)
    feature_embedder.eval()  # evaluation mode
    
    with torch.no_grad():
        features = feature_embedder(image_tensor.to(device))
        
    # 추출된 특성을 원하는 형태로 변환 (예: 1, 28, 28, 1024)
    reshaped_features = features.view(1, 28, 28, -1)
    # reshaped_features = features.view(1, -1)  # 1D 벡터로 변환
    return reshaped_features

# 테스트 이미지에 대한 클래스를 예측하는 함수
def predict_class_for_test_image(test_feature, memory_bank):
    memory_features = np.array(memory_bank['features'])
    memory_labels = np.array(memory_bank['labels'])

    # 각 테스트 특성 벡터에 대해 메모리 뱅크의 모든 벡터와의 거리 계산
    distances = distance.cdist(test_feature.numpy(), memory_features, 'euclidean')
    
    # # 가장 가까운 이웃의 인덱스 찾기
    # nearest_indices = np.argmin(distances, axis=1)
    
    # 가장 가까운 이웃의 인덱스 찾기 (예: k=1)
    k = 1
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    
    # 가장 가까운 이웃의 레이블 가져오기
    nearest_labels = memory_labels[nearest_indices.flatten()]

    # 가장 빈번한 클래스 (레이블) 결정
    most_common_label, _ = Counter(nearest_labels).most_common(1)[0]
    
    return most_common_label

def resize_mask(mask, target_size):
    # mask는 1x1xHxW 형태로 가정합니다.
    # target_size는 (target_height, target_width) 형태의 튜플입니다.
    # resized_mask = torch.nn.functional.interpolate(mask, size=target_size, mode='nearest')
    # return resized_mask
    return F.interpolate(mask, size=target_size, mode='nearest')

def apply_mask(features, mask):
    # 마스크가 boolean 타입이고, features와 마스크의 모양이 일치하도록 합니다
    mask = mask.squeeze()  # features와 모양을 맞추기 위해 차원을 줄입니다
    selected_features = features[mask, :]  # boolean 인덱싱을 사용합니다
    return selected_features

def predict_test_image_class(test_features, mask_tensor, memory_bank):
    # 테스트 특성에 마스크를 리사이즈하고 적용합니다
    resized_mask = F.interpolate(mask_tensor, size=(28, 28), mode='nearest').squeeze()
    masked_features = apply_mask(test_features.squeeze(), resized_mask > 0.5)

    if masked_features.dim() == 1:
        # masked_features가 2차원이 되도록 보장합니다 (N, 1024)
        masked_features = masked_features.unsqueeze(0)

    # scipy 함수와 호환되도록 numpy로 변환합니다
    masked_features_np = masked_features.numpy()
    memory_features_np = np.array(memory_bank['features'])
    memory_labels_np = np.array(memory_bank['labels'])

    # 각 마스크된 특성과 메모리 은행 특성 간의 거리를 계산합니다
    distances = scipy.spatial.distance.cdist(masked_features_np, memory_features_np, 'euclidean')
    
    # 각 마스크된 특성에 대해 가장 가까운 메모리 은행 특성을 찾습니다
    nearest_indices = np.argmin(distances, axis=1)
    nearest_labels = memory_labels_np[nearest_indices]

    # 가장 흔한 라벨을 결정합니다
    predicted_label = Counter(nearest_labels).most_common(1)[0][0]
    return predicted_label



def predict_labels_from_features(masked_features, memory_bank):
    memory_features = np.array(memory_bank['features'])
    memory_labels = np.array(memory_bank['labels'])

    distances = scipy.spatial.distance.cdist(masked_features.numpy(), memory_features, 'euclidean')
    nearest_indices = np.argmin(distances, axis=1)
    nearest_labels = memory_labels[nearest_indices]

    most_common_label, _ = Counter(nearest_labels).most_common(1)[0]

    # 예측 상세 정보를 생성합니다.
    details = []
    for i, feature in enumerate(masked_features):
        nearest_distance = distances[i, nearest_indices[i]]
        nearest_vector = memory_features[nearest_indices[i]]
        test_vector = feature.numpy()

        detail = {
            'test_vector': test_vector,
            'nearest_vector': nearest_vector,
            'nearest_distance': nearest_distance,
            'nearest_label': nearest_labels[i]
        }
        details.append(detail)

    return most_common_label, details

def append_to_csv(img_name, prediction, label, distance, nearest_vector, test_vector, csv_path):
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_name, prediction, label, distance, nearest_vector, test_vector])

def predict_class_from_labels(labels):
    most_common_label, _ = Counter(labels).most_common(1)[0]
    return most_common_label

def save_memory_bank(memory_bank, file_path):
    torch.save(memory_bank, file_path)
    print(f"Memory bank saved to {file_path}")
    
def load_memory_bank(file_path):
    memory_bank = torch.load(file_path)
    print(f"Memory bank loaded from {file_path}")
    return memory_bank



# file_path = "memory_bank.pt"
# loaded_memory_bank = load_memory_bank(file_path)

# # Sample usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_shape = [3, 224, 224]  # Replace with the actual input shape
# image_path = "000.png"
# mask_path = "000_mask.png"

# resize = (256, 256)  # 리사이징할 크기
# crop_size = (224, 224)  # 크롭할 크기

# image_tensor = preprocess_image(image_path, resize, crop_size)
# mask_tensor = preprocess_mask(mask_path, resize, crop_size)

# # 테스트 이미지로부터 특성 추출
# test_feature = extract_features_from_test_image(image_path, device, resize, crop_size)


# predicted_class = predict_test_image_class(test_feature, mask_tensor, memory_bank=loaded_memory_bank)
# # 테스트 이미지의 예측 클래스 결정
# # predicted_class = predict_class_for_test_image(test_feature, loaded_memory_bank)
# print(f"Predicted class for the test image is: {predicted_class}")

# # memory_bank = label_feature_patches(image_tensor, mask_tensor, device, input_shape)
# # save_memory_bank(memory_bank, file_path)

# print("Memory Bank Constructed")


if __name__ == "__main__":
    # 이 파일이 직접 실행될 때만 동작하는 코드
    print("Directly running current ~ .py")