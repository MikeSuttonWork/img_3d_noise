import numpy as np
import struct
from noise3d_calc import noise3d_calc


def test_method1():

    cube = np.reshape(struct.unpack('d' * 24*32*30, open('tests/cube.raw' ,'rb').read()), (24,32,30), order='F')

    # Also VarDH1 in Noise3d_Examples.m
    VarDH_matlab = np.vstack(struct.unpack('d' * 7, open('tests/VarDH.raw', 'rb').read()))
    VarC_matlab = np.vstack(struct.unpack('d' * 7, open('tests/VarC.raw', 'rb').read()))
    VarRaw_matlab = np.vstack(struct.unpack('d' * 7, open('tests/VarRaw.raw', 'rb').read()))

    VarDH, VarC, VarRaw, MDH, MC = noise3d_calc(cube)

    np.testing.assert_allclose(VarDH, VarDH_matlab)
    np.testing.assert_allclose(VarC, VarC_matlab)
    np.testing.assert_allclose(VarRaw, VarRaw_matlab)


def test_with_defective_pix():
    cube = np.reshape(struct.unpack('d' * 24*32*30, open('tests/cube.raw' ,'rb').read()), (24,32,30), order='F')
    defective = np.reshape(struct.unpack('B' * 24 * 32, open('tests/defective.raw', 'rb').read()), (24, 32), order='F')

    # Also VarDH1 in Noise3d_Examples.m
    VarDH_matlab = np.vstack(struct.unpack('d' * 7, open('tests/VarDH_defective.raw', 'rb').read()))
    VarC_matlab = np.vstack(struct.unpack('d' * 7, open('tests/VarC_defective.raw', 'rb').read()))
    VarRaw_matlab = np.vstack(struct.unpack('d' * 7, open('tests/VarRaw_defective.raw', 'rb').read()))

    VarDH, VarC, VarRaw, MDH, MC = noise3d_calc(cube, defective)

    np.testing.assert_allclose(VarDH, VarDH_matlab)
    np.testing.assert_allclose(VarC, VarC_matlab)
    np.testing.assert_allclose(VarRaw, VarRaw_matlab)

