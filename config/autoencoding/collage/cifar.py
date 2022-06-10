"""python scripts/train_autoencode.py --cfg config/autoencoding/collage/cifar.py
"""
from config.default_cifar import get_baseline_cifar_config

def get_config():
    config = get_baseline_cifar_config()
    config.model = "CollageAE"
    ##### Experiment handling ####
    config.seed = 123
    config.n_devices = 2
    config.wandb_dir = './checkpoints/wandb/' # '/checkpoints/winniexu/wandb'
    config.project_name = 'cifar'
    config.user = 'xwinxu' # wandb user
    config.exp_name = "collage_ae" # log name to wandb. if placeholder `config.exp_name` is used instead.
    config.run_id = ""

    config.restore_from_checkpoint = False
    config.checkpoint_dir = "checkpoints/autoencode/"
    config.rhs = [8, 8, 8]
    config.rws = [8, 8, 8]
    config.dhs = [32, 32, 32]
    config.dws = [32, 32, 32]
    config.quaddepths = [2,2,2]
    config.co = 'general' # defaults GeneralizedCollageOperator, alt: 'bias'
    config.superres_eval_factors = []

    return config