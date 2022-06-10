from config.generation.generative_cifar import get_generative_cifar_config

def get_config():
    config = get_generative_cifar_config()

    ##### Experiment handling ####
    config.seed = 123
    config.n_devices = 2
    config.wandb_dir = './checkpoints/wandb/'
    config.project_name = 'cifar'
    config.user = 'user' # wandb user
    config.exp_name = "vdvae_cifar" # log name to wandb. if placeholder `config.exp_name` is used instead.
    config.run_id = ""

    ##### Checkpointing #####
    config.restore_from_checkpoint = True
    config.checkpoint_dir = f'./checkpoints/'
    config.load_step = 0
    config.ckpt_steps = 10000    

    ##### Model #####
    config.model = "VDVAE"
    config.loss_fn =  'logistic' 
    config.dec_num_mixtures = 10
    config.n_mix_params = 3
    config.no_bias_above = 64
    config.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
    config.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
    config.enc_widths = None
    config.dec_widths = None
    config.enc_default_width = 384
    config.dec_default_width = 384
    config.n_fourier_enc = ()
    config.use_fourier_enc = False
    config.n_fourier_dec = ()
    config.use_fourier_dec = False 
    config.enc_bottleneck_factor = 0.25
    config.dec_bottleneck_factor = 0.25
    config.dec_latent_dim = 16
    config.dec_num_mixtures = 10
    config.no_bias_above = 64
    config.normalize = True
    config.norm_factor = 1
    config.residual_posteriors = False
    
    return config

