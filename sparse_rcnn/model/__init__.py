from .sparseRCNN import SparseRCNN
from .dynamicSparseRCNN import DynamicSparseRCNN
from .head import DynamicHead

_available_model = {
    "SparseRCNN": SparseRCNN,
    "DynamicSparseRCNN": DynamicSparseRCNN
}


def build_model(cfg, num_classes, backbone, raw_outputs=False):
    model_name = cfg.MODEL.NAME
    if model_name not in _available_model.keys():
        raise ValueError("Model {} not supported".format(model_name))
    return _available_model[model_name](cfg, num_classes, backbone, raw_outputs)
