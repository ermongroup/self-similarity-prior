"""The template config file for launching Fractal Autoencoder runs with a stylizing image."""
import ml_collections

def get_style_config():
    config = ml_collections.ConfigDict()

    ########### Distributed / checkpointing / restore ##########
    config.n_devices = 1 
    config.checkpoint_dir = "" # This is overwritten by workdir in `train.py` and `autoencode.py`
    config.restore_from_checkpoint = True
    config.load_step = 0

    ###### Turn on for single source dataset ########
    config.dataset = "single_source" 
    config.datadir = "data/kheart_1000.png"
    ###### Turn on for entire dataset training
    # config.dataset = "cifar10"
    # config.datadir = f"~/data/{config.dataset}" # replace with your path to dataset
    
    # data res needs to divide the encoder block counts xhxw and also the block_size
    config.data_res = 40
    config.original_data_res = 1000
    config.data_width = 3
    config.n_channels = 3

    config.model = "FractalAE"
    config.style_source_img = None # set this to an image and this will disable auxilliary source training
    config.block_dims = (40, 40)
    config.reweight_factor_d = 1.0 # set both to 1. to have no reweighting and train normally (i.e. if style_source_img is None)
    config.reweight_factor_a = 1.0
    config.augment_aux_sources = False # if True, will do 4 rotations, flip, no flip, and flip x 4 rotations = 8 extra aux_dims
    config.sub_batch_eval = True # if augmentations is on and resource limited, iteratively reconstruct image
    config.sub_bs = 100 # this is empirically derived, ensure at least original_data_res // sub_bs > 0
    config.debug = False # prevents the image from being unpartitioned before plotting for quicker running

    ######## PatchEmbedder configs ##############
    config.enc_blocks = "20h20wx1" # szs of range patches and depth of encoder, should divide data_res
    config.range_szs = (20, 20) # should match with the above, cannot exceed domain
    # compute all theta for one range patch using the corresponding embedding for computing the range patch.
    config.reduce_type = 'pool' # 'crop' for small images

    config.enc_patch_emb_dim = 3 # changes model size drastically
    config.domain_szs = (40, 40) # szs of domain patches (should be compat w/ original data res)

    ######### CollageDecoder configs ##################
    config.decode_steps = 10
    config.separate_maps_per_channel = True
    config.n_aux_sources = 0 # only need 1 if `style_source_img` is not None, set to > 1 if not using style_source_img
    config.use_rotation_aug = False
    config.depthwise_patch_to_theta = True
    config.collage_width = config.data_width
    config.significant_digits = 2
    
    ########## Train / log / eval / plot ##########
    config.seed = 123
    config.subset_size = None
    config.bs = 1 * config.n_devices
    config.lr = 1e-4
    config.max_grad_norm = 1000
    config.wd = 0.001
    config.l2_reg = True
    config.epochs = 3000
    config.max_steps = 1000000
    config.log_steps = 1
    config.plot_steps = 5
    config.ckpt_steps = 50
    config.eval_epochs = 5
    # to eval on stitched 120x120 and 1200x1200 images made up of smaller patches. 
    # Should be available under `datadir` as processed by `data_utils`. Only compatible with `dota`
    config.eval_on_stitch = False # True if dota
    config.superres_on_eval = 1 # produce and plot upscaled validation reconstructions at the end of each validation loop. Leave to `1` for standard resolution.
    config.n_eval_batches = -1 # -1 for entire val set

    config.exp_name = f"enc{config.enc_blocks}-blk{config.block_dims[0]}-factorda{config.reweight_factor_d}_{config.reweight_factor_a}" # this is for logging only
    return config
