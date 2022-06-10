from config.default_mnist import get_baseline_mnist_config

def get_config():
    config = get_baseline_mnist_config()
    config.model = "VDCVAE"
    config.exp_name = f"{config.model}_{config.dataset}"

    config.enc_blocks = "28x1,28d2,14x1,14d2,7x1,7d7,1x1" 
    config.dec_blocks = "1x2,7m1,7x4,14m7,14x1"

    config.enc_widths = None
    config.dec_widths = None
    config.enc_default_width = 64 # 200
    config.dec_default_width = 64 # 200

    config.n_fourier_enc = ()
    config.use_fourier_enc = False
    config.n_fourier_dec = ()
    config.use_fourier_dec = False 

    config.enc_bottleneck_factor = 0.25
    config.dec_bottleneck_factor = 0.25
    config.dec_latent_dim = 1
    config.residual_posteriors = False
    config.no_bias_above = 128

    ##### Collage Operator #####
    config.collage_width = config.dec_num_mixtures * ((config.data_width * 3) + 1)
    config.decode_steps = 5
    config.n_aux_sources = 0
    config.use_rotation_aug = False
    config.separate_maps_per_channel = True
    config.depthwise_patch_to_theta = False  
    config.range_dims = (7, 7)

    config.superres_eval_factors = [2,5,8]
    return config
