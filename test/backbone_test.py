import timm



_available_backbones = {
    "resnet18": {"model_name": "resnet18",
                 "features_only": True,
                 "out_indices": (1, 2, 3, 4),
                 "pretrained": True,
                 "num_classes": 0,
                 "global_pool": ""},
    "resnet50": {"model_name": "resnet50",
                 "features_only": True,
                 "out_indices": (1, 2, 3, 4),
                 "pretrained": True,
                 "num_classes": 0,
                 "global_pool": ""},
    "efficientnet_b3": {"model_name": "efficientnet_b3",
                        "features_only": True,
                        "out_indices": (1, 2, 3, 4),
                        "pretrained": True,
                        "num_classes": 0,
                        "global_pool": ""},

}

resnet50 = timm.create_model(**_available_backbones["resnet18"])
print(resnet50)
print(resnet50.feature_info.info)