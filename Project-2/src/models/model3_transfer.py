from torchvision import models
import torch.nn as nn


def build_transfer_model(num_classes: int = 6, pretrained: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_feature_extractor(pretrained: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    layers = list(model.children())[:-1]
    return nn.Sequential(*layers)
