"""python scripts/train_autoencode.py --cfg config/autoencoding/vanilla/cifar.py"""

from config.default_cifar import get_baseline_cifar_config

def get_config():
    config = get_baseline_cifar_config()
    config.model = "FCAE"

    ##### Experiment handling ####
    config.seed = 123
    config.n_devices = 2
    config.wandb_dir = '/checkpoint/wandb/'
    config.project_name = 'cifar'
    config.user = 'xwinxu' # wandb user
    config.exp_name = "fc_ae" # log name to wandb. if placeholder `config.exp_name` is used instead.
    config.run_id = ""

    ####### CollageAutoencoder Parameters ########
    config.restore_from_checkpoint = True
    config.checkpoint_dir = "/checkpoint/test_fcae"
    config.rhs = [8, 8, 8]
    config.rws = [8, 8, 8]
    config.dhs = [32, 32, 32]
    config.dws = [32, 32, 32]
    config.quaddepths = [2, 2, 2]
    config.co = 'general' # defaults GeneralizedCollageOperator, alt: 'bias'
    config.superres_eval_factors = []

    ####### Autoencoder Parameters ########
    config.hidden_sizes = (3, 3, 3)
    config.activation = 'relu'
    config.kernel_init = [0.5, 0.5, 0.5]
    config.bias_init = 'zeros'

    return config
