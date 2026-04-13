from .noise3d_calc import noise3d_calc
from .util import img_3d_noise_as_sigma_dict
import pandas as pd


def unbiased_sigmas(Cube, Defective=None):
    """
    The first output of noise3d_calc is the variance VarDH   : Unbiased 7x1 vector of variances estimates.
    
    The helper function img_3d_noise_as_sigma_dict is used to convert the output of noise3d_calc to 
    a dictionary of standard deviations.
    """
    return img_3d_noise_as_sigma_dict(noise3d_calc(Cube, Defective)[0])


def unbiased_sigmas_series(Cube, Defective=None):
    """
    The first output of noise3d_calc is the variance VarDH   : Unbiased 7x1 vector of variances estimates.
    
    The helper function img_3d_noise_as_sigma_dict is used to convert the output of noise3d_calc to 
    a pd.Series of standard deviations.
    """
    return pd.Series(img_3d_noise_as_sigma_dict(noise3d_calc(Cube, Defective)[0]))