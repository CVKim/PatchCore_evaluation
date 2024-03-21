import torch
import torch.nn.functional as F
from typing import List

import torch.nn as nn
from torchvision import models, transforms

from patch_maker import PatchMaker
from feature_aggregator import NetworkFeatureAggregator
from common import Preprocessing, Aggregator

from PIL import Image
from torch.nn.functional import cosine_similarity

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

def preprocess_image(image_path, input_shape):
    transform = transforms.Compose([
        transforms.Resize(input_shape[1:3]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


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

def extract_features_from_image(image_path, device, input_shape):
    
    image_tensor = preprocess_image(image_path, input_shape)
    
    feature_embedder = create_feature_embedder(device, input_shape)
    feature_embedder.eval() # evaluation mode
    
    with torch.no_grad():
        features = feature_embedder(image_tensor.to(device))
    return features

## sample
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "000.png"
input_shape = [3, 224, 224]  # 3,244,244가 맞나??
features = extract_features_from_image(image_path, device, input_shape)
print("Extracted Features Shape:", features.shape)

# 784, 1024 layer2,3 나온 걸 784을 1024로 변환 max or avg? pooling해서 resize
# 1,2 → 2, 1 