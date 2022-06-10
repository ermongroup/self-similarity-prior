import ml_collections

def get_baseline_mnist_config():
    config = ml_collections.ConfigDict()
    config.seed = 123
    
    ##### Data #####
    config.dataset = "mnist"
    config.datadir = f"~/data/{config.dataset}"
    config.data_res = 28
    config.data_width = 1
    config.n_channels = 1 # this is not used

    ##### Checkpointing #####
    config.restore_from_checkpoint = True
    config.checkpoint_dir = f'./checkpoints/{config.dataset}'
    config.load_step = 0
    config.ckpt_steps = 10000 

    ##### Aux. #####  
    config.subset_size = None # for debugging purposes; should be multiple of n_devices
    config.n_devices = 2 
    config.style_source_img = None
    config.block_dims = None
    config.reweight_factor_a = 1.0
    config.reweight_factor_d = 1.0

    ##### Fixed model hyperparameters #####
    config.dec_num_mixtures = 10
    config.n_mix_params = 3
    config.no_bias_above = 64
    
    ##### Training and optimization #####
    config.lr = 0.001
    config.bs = 16 * config.n_devices
    config.epochs = 1000
    config.max_steps = 1000000
    config.max_grad_norm = 600.
    config.skip_threshold = 300 
    config.wd = 0.001
    config.warmup_iters = 100
    config.ema_rate = 0.99

    ##### Logging #####
    config.log_steps = 10
    config.plot_steps = 100
    config.base_temperature = 1.0
    

    ##### Eval #####
    config.test_bs = 32 * config.n_devices
    config.normalize = True
    config.norm_factor = 1
    config.use_ema = True # whether to use exponential moving average of params for evaluation    
    config.n_samples = 64 
    config.eval_epochs = 1
    config.n_eval_batches = -1 # -1 for entire val set

    ##### Loss fn. #####
    config.beta = 1.
    config.likelihood_func = "dmol"
    config.loss_fn =  'logistic'   
    
    config.collage_width = config.data_width # for compatibility with FractalVAE, all models need a collage_width default
    config.superres_eval_factors = []
    
    config.style_source_img = False
    config.exp_name = "baseline_mnist"

    return config
