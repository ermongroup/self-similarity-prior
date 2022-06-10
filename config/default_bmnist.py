import ml_collections

def get_baseline_bmnist_config():
    config = ml_collections.ConfigDict()
    config.seed = 123

    ##### Wandb ####
    config.project_name = 'baseline_bmnist'
    config.exp_name = 'placeholder'
    config.user = 'user' # wandb user
    config.exp_name = "baseline_bmnist" # log name to wandb. if placeholder `config.exp_name` is used instead.
    
    ##### Data #####
    config.dataset = "binarized_mnist"
    config.datadir = f"~/data/{config.dataset}"
    config.data_res = 28
    config.data_width = 1

    ##### Checkpointing #####
    config.restore_from_checkpoint = True
    config.checkpoint_dir = f'./checkpoints/{config.dataset}'
    config.load_step = 0
    config.ckpt_steps = 10000 

    ##### Aux. #####  
    config.subset_size = None # for debugging purposes; should be multiple of n_devices
    config.n_devices = 2 

    ##### Fixed model hyperparameters #####
    config.dec_num_mixtures = 10
    config.no_bias_above = 64

    ##### Collage Operator aux. flags #####
    config.augment_aux_sources = False
    config.style_source_img = None
    config.collage_width = config.data_width # for compatibility with Collage VAE, all models need a collage_width default
    config.superres_eval_factors = []
    config.n_aux_latents = -1

    ##### Training and optimization #####
    config.lr = 0.0001
    config.bs = 16 * config.n_devices
    config.epochs = 2000
    config.max_steps = 100000000 # arbitrary high num.
    config.max_grad_norm = 600.
    config.skip_threshold = 600 
    config.wd = 0.000
    config.warmup_iters = 100
    config.ema_rate = 0.

    ##### Logging #####
    config.log_steps = 1000
    config.plot_steps = 1000
    config.base_temperature = 1.0

    ##### Eval #####
    config.test_bs = 32 * config.n_devices
    config.normalize = True
    config.norm_factor = 1
    config.use_ema = True # whether to use exponential moving average of params for evaluation    
    config.n_samples = 9 
    config.eval_epochs = 10
    config.n_eval_batches = -1 # -1 for entire val set

    ##### Loss fn. #####
    config.beta = 1.0 # weighting coeff. between rate and distortion in ELBO.
    config.likelihood_func = "bernoulli"
    config.loss_fn =  'logistic'   


    return config
