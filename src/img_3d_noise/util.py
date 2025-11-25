import numpy as np


def img_3d_noise_as_sigma_dict(sig_vector):
    component_names = ['sig_t', 'sig_v', 'sig_h', 'sig_tv', 'sig_th', 'sig_vh', 'sig_tvh']
    return {k: float(np.sqrt(v)) for k, v in zip(component_names, sig_vector)}