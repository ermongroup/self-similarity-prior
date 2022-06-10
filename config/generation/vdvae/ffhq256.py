from config.generation.generative_ffhq import get_generative_ffhq_config

def get_config():
    config = get_generative_ffhq_config()

    ##### Experiment handling ####
    config.seed = 123
    config.n_devices = 2
    config.wandb_dir = '/checkpoint/wandb/'
    config.project_name = 'ffhq256'
    config.user = 'xwinxu' # wandb user
    config.exp_name = "vdvae_ffhq256" # log name to wandb. if placeholder `config.exp_name` is used instead.
    config.run_id = ""

    ##### Checkpointing #####
    config.restore_from_checkpoint = True
    config.checkpoint_dir = f'/checkpoint/'
    config.load_step = 0
    config.ckpt_steps = 1000

    ##### Model #####
    config.model = "VDVAE"
    config.loss_fn =  'logistic' 
    config.dec_num_mixtures = 10
    config.n_mix_params = 3
    config.no_bias_above = 256 # TODO: this should be 64 in accordance w/ vdvae
    config.enc_blocks = "256x1,256d4,64x1,64d2,32x1,32d8,4x1,4d4,1x1" # less shallow
    config.dec_blocks = "1x1,4m1,4x1,32m4,32x1,64m32,64x1,256m64"
    config.enc_widths = None
    config.dec_widths = None
    config.enc_default_width = 128 # 512
    config.dec_default_width = 128 # 512
    config.n_fourier_enc = ()
    config.use_fourier_enc = False
    config.n_fourier_dec = ()
    config.use_fourier_dec = False 
    config.enc_bottleneck_factor = 0.25
    config.dec_bottleneck_factor = 0.25
    config.dec_latent_dim = 16
    config.dec_num_mixtures = 10
    config.normalize = True
    config.norm_factor = 1
    config.residual_posteriors = False
    
    return config

