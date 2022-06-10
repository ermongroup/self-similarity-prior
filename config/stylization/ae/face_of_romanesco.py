"""Reconstruct face using auxilliary image as romanesco."""
from config.ae.default_style_config import get_style_config

def get_config():
    config = get_style_config()
    
    config.model = "StyleFractalAE"
    config.dataset = "data/khear_1000.png"
    config.original_data_res = 1000
    config.style_source_img = "data/romanesco.jpg"
    config.reweight_factor_a = 0.5
    config.n_aux_sources = 1

    return config
