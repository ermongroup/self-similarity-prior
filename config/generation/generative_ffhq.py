"""python scripts/train_generative.py --cfg config/generation/vdvae/ffhq256.py"""

from config.default_ffhq import get_baseline_ffhq_config

def get_generative_ffhq_config():
    config = get_baseline_ffhq_config()

    # according to vdvae: https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/hps.py#L54

    ##### Training and optimization #####
    config.lr = 0.00015
    config.bs = 2 * config.n_devices # 4 works, but ooms
    config.epochs = 2000
    config.max_steps = 100000000 # arbitrary high num.
    config.max_grad_norm = 130.
    config.skip_threshold = 180
    config.wd = 0.01
    config.warmup_iters = 100 # TODO: actually implement warmup!
    config.ema_rate = 0.9999

    ##### Logging #####
    config.log_steps = 100
    config.plot_steps = 1000
    config.base_temperature = 1.0

    ##### Eval #####
    config.test_bs = 1 * config.n_devices
    config.normalize = True
    config.norm_factor = 1
    config.use_ema = True # whether to use exponential moving average of params for evaluation    
    config.n_samples = 36 
    config.eval_epochs = 1
    config.n_eval_batches = -1 # -1 for entire val set TODO: just 1 for testing now

    ##### Loss fn. #####
    config.beta = 1.0 # weighting coeff. between rate and distortion in ELBO.
    config.likelihood_func = "dmol"
    config.loss_fn =  "logistic"   

    ##### Collage Operator aux. flags #####
    # These should be removed eventually, they are needed for compatibility since `train_generative` uses
    # them in several places
    config.augment_aux_sources = False
    config.style_source_img = None
    config.collage_width = config.data_width # for compatibility with Collage VAE, all models need a collage_width default
    config.superres_eval_factors = []
    config.n_aux_latents = -1

    return config
