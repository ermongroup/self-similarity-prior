from config.default_cifar import get_baseline_cifar_config

def get_config():
    config = get_baseline_cifar_config()
    config.model = "VDCVAE"
    config.exp_name = f"{config.model}_{config.dataset}"
    config.bs = config.n_devices * 4
    config.restore_from_checkpoint = False

    config.enc_blocks = "32x1,32d2,16x1,16d2,8x6,8d2,4x3,4d4,1x3"
    config.dec_blocks = "1x20"

    config.enc_widths = None
    config.dec_widths = None
    config.enc_default_width = 200 
    config.dec_default_width = 200 

    config.n_fourier_enc = ()
    config.use_fourier_enc = False
    config.n_fourier_dec = ()
    config.use_fourier_dec = False 

    config.enc_bottleneck_factor = 0.25
    config.dec_bottleneck_factor = 1
    config.dec_latent_dim = 128
    config.residual_posteriors = False
    config.no_bias_above = 64
    config.beta = 1.0

    ##### Collage Operator #####
    config.collage_width = config.dec_num_mixtures * ((config.data_width * 3) + 1)
    config.decode_steps = 10
    config.n_aux_sources = 20
    config.use_rotation_aug = False
    config.separate_maps_per_channel = True
    config.depthwise_patch_to_theta = False  
    config.range_dims = (16, 16)

    config.superres_eval_factors = [3]
    return config