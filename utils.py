import numpy as np


def noise3d_to_dict(VarDH, VarC, VarRaw, MDH, MC):
    """
    Convert output of the main noise3d_calc to a dictionary of standard deviations.
    """
    result = {}
    for name, var in zip(['SigDH', 'SigC', 'SigRaw'], [VarDH, VarC, VarRaw]):
        result[name] = {}
        for k, v in zip(['s_t', 's_v', 's_h', 's_tv', 's_th', 's_vh', 's_tvh'], var.flatten()):
            result[name][k] = float(np.sqrt(v) * 1000)

