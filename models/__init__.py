from utils.registry import MODEL_REGISTRY


def get_model(opt):
    MODEL_REGISTRY.scan_and_register()
    return MODEL_REGISTRY.get(opt['model_class'])(**opt['params'])
