import numpy as np
from noise3d_calc import noise3d_calc
import pandas as pd


def test_synthetic_img():

    s_v = 10
    s_h = 30
    s_t = 50

    results = []

    for ntrial in range(500):
        cube = np.zeros((100, 100, 100))


        for n in range(100):
            cube[n, :, :] += np.random.randn(1) * s_v

        for n in range(100):
            cube[:, n, :] += np.random.randn(1) * s_h

        for n in range(100):
            cube[:, :, n] += np.random.randn(1) * s_t

        vardh, _, _, _, _ = noise3d_calc(cube)
        result = {}
        for k, v in zip(['s_t', 's_v', 's_h', 's_tv', 's_th', 's_vh', 's_tvh'], vardh.flatten()):
            result[k] = float(np.sqrt(v))

        result['std'] = cube[:, :, 0].std()

        results.append(result)

    df = pd.DataFrame(results)
    
    print(df)
    print(df.mean(axis=0))