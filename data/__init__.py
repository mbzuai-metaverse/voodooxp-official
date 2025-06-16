from utils.registry import DATASET_REGISTRY


def get_dataset(opt):
    DATASET_REGISTRY.scan_and_register()
    return DATASET_REGISTRY.get(opt['data_class'])(**opt['kwargs'])
