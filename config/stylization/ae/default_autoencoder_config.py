"""Default configs for fractal autoencoder."""
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    ########### Distributed / checkpointing / restore ##########
    config.n_devices = 2 #4
    config.datadir = f"../../data" #f"../" 
    config.checkpoint_dir = "" # This is overwritten by workdir in `train.py` and `autoencode.py`
    config.restore_from_checkpoint = False
    config.load_step = 0

    # ############ Dota Dataset ############
    # config.dataset = 'dota'
    # config.data_res = 40
    config.dataset = "cifar10"
    config.datadir = f"~/data/{config.dataset}" # replace with your path to dataset
    config.data_res = 32
    config.data_width = 3
    config.n_channels = 3

    config.model = "FractalAE"

    ######## PatchEmbedder configs ##############
    config.enc_blocks = "20h20wx4" # szs of range patches and depth of encoder
    config.range_szs = (20, 20) # should match with the above e.g (20, 20) and 20h20wx3

    config.enc_patch_emb_dim = 10
    config.domain_szs = (40, 40) # szs of domain patches

    ######### CollageDecoder configs ##################
    config.decode_steps = 10
    config.separate_maps_per_channel = True
    config.n_aux_sources = 10
    config.use_rotation_aug = False
    config.depthwise_patch_to_theta = True
    config.collage_width = config.data_width
    config.significant_digits = 2
    
    ########## Train / log / eval / plot ##########
    config.seed = 123
    config.subset_size = None
    config.bs = 32 * config.n_devices
    config.lr = 1e-3
    config.max_grad_norm = 1000
    config.wd = 0.000
    config.l2_reg = True
    config.epochs = 100
    config.max_steps = 1000000
    config.log_steps = 40
    config.plot_steps = 200
    config.ckpt_steps = 2000 # 2000
    config.eval_epochs = 1
    # to eval on stitched 120x120 and 1200x1200 images made up of smaller patches. 
    # Should be available under `datadir` as processed by `data_utils`. Only compatible with `dota`
    config.eval_on_stitch = True 
    config.superres_on_eval = 10 # produce and plot upscaled validation reconstructions at the end of each validation loop. Leave to `1` for standard resolution.
    config.n_eval_batches = -1 # -1 for entire val set
    return config
