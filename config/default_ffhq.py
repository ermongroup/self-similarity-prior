from config.default_mnist import get_baseline_mnist_config

def get_baseline_ffhq_config():
    config = get_baseline_mnist_config()
    config.dataset = "ffhq256"
    config.datadir = "/ssd003/projects/ffhq"
    config.data_res = 256
    config.data_width = 3
    config.exp_name = "baseline_ffhq256"

    return config
