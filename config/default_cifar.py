from config.default_mnist import get_baseline_mnist_config

def get_baseline_cifar_config():
    config = get_baseline_mnist_config()
    config.dataset = "cifar10"
    config.data_res = 32
    config.data_width = 3    
    return config
