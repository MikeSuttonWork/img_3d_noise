import numpy as np

#-------------------------------------------------------------------------------
def noise3d_sim(t, v, h, tv, th, vh, tvh, T, V, H):
    """
    Noise3d_Sim will simulate the 3D noise cube defined by Parameters

       Inputs:
               Parameters=[t;v;h;tv;th;vh;tvh;T;V;H];

       The first 7 elements are the standard deviation of the Gaussian
       distributed independent random processes

       t   % 1D random process Temporal
       v   % 1D random process Vertical
       h   % 1D random process Horizontal
       tv  % 2D random process Temporal, Vertical
       th  % 2D random process Temporal, Horizontal
       vh  % 2D random process Vertical, Horizontal
       tvh % 3D random process Temporal, Vertical, Horizontal

       The final elements Parameters(8:10) are the number of sampling in each
       dimension
       T % Number of frames
       V % Number of vertical pixels
       H % Number of horizonal pixels

       Outputs:
               Cube : 3D noise cube as defined by Parameters
               VarV : vector of variances
               Nv   : Degrees of freedom

       See also Noise3d_Example, Noise3d_Calc, Noise3d_CI

     D. Haefner, NVESD January 2015

     Additional details of method and rationale found in:
     Sampling Corrected 3D Noise with Confidence Intervals
     https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-15-4907
    """
    # Initialize Cube with 3D tvh term
    Cube = np.random.randn(V,H,T) * tvh;

    # Add the replitcated 1D random processes
    Cube += np.tile(np.random.randn(V,1,T) * tv, (1,H,1))
    Cube += np.tile(np.random.randn(1,H,T) * th, (V,1,1))
    Cube += np.tile(np.random.randn(V,H,1) * vh, (1,1,T))

    # Add the replicated 1D random processes
    Cube += np.tile(np.random.randn(1,1,T) * t, (V,H,1))
    Cube += np.tile(np.random.randn(V,1,1) * v, (1,H,T))
    Cube += np.tile(np.random.randn(1,H,1) * h, (V,1,T))

    VarV = np.array([t,v,h, tv, th, vh,  tvh])**2
    Nv   = np.array([T,V,H,T*V,T*H,V*H,T*V*H])

    return [Cube, VarV, Nv]

#-------------------------------------------------------------------------------
# function [VarDH,VarC,VarRaw,MDH,MC] = Noise3d_Calc(Cube, Defective)
def noise3d_calc(Cube, Defective=None):
    """
    Noise3d_Calc will calculate the variance for the 7 independent random
    processes of the standard 3D model

       Inputs:
               Cube        : either 2D or 3D noise cube
               Defective   : binary array detailing map of defective pixels
                             value of 1 = defective, 0 = acceptable
       Outputs:
               VarDH   : Unbiased 7x1 vector of variances estimates
               VarC    : Classic 7x1 vector of variances estimates
               VarRaw  : Raw variances after directional averaging
               MDH     : Unbiased measurement mixing matrix
               MC      : Classic measurement mixing matrix

    Outputs variance vectors are in order : t;v;h;tv;th;vh;tvh

    See also Noise3d_Example, Noise3d_CI, Noise3d_Sim

    D. Haefner, NVESD January 2015

    Additional details of method and rationale found in:
    Sampling Corrected 3D Noise with Confidence Intervals
    https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-15-4907
    """

    if Defective is None:
        Defective = (0 * Cube[:, :, 0]).astype(int)

    # Populate sizes
    V, H, T = Cube.shape

    # Verify global mean (ignoring defective) is removed...if running, this
    # should have already been done
    Omega = Cube.mean(axis=2)
    Omega = Omega[Defective == 0].mean()
    Cube = Cube - Omega

    # Set Defective pixels to zero to be ignored during summations
    Cube[Defective == 1, :] = 0

    Dv = Cube.sum(axis=0)
    Dh = Cube.sum(axis=1)
    Dt = Cube.sum(axis=2)
    E1tvh =        Cube.sum(axis=2)
    E2tvh = (Cube ** 2).sum(axis=2)

    # average discounting those pixels which were marked as defective
    Vsum = (Defective == 0).sum(axis=0)
    Hsum = np.vstack((Defective == 0).sum(axis=1))
    Tsum = (T * (Defective==0).astype(float))

    #    Normalize
    Dv = Dv / np.tile(Vsum, (T, 1)).T
    Dh = Dh / np.tile(np.vstack(Hsum), (1, T))
    Dt = Dt / Tsum

    # Adjust for any dead columns
    Dv[Vsum == 0, :] = 0

    # Adjust for any dead rows
    Dh[Hsum.flatten() == 0, :] = 0

    # Calculate additional directional sums
    DvDh = (Dh * np.tile(Hsum, [1, T])).sum(axis=0) / Hsum.sum()
    DtDv = Dv.sum(axis=1) / T
    DtDh = Dh.sum(axis=1) / T

    # Adjust arrays to remove where dead pixels have
    DtDv = DtDv[Vsum != 0]
    DtDh = DtDh[Hsum.flatten() != 0]

    Dv = Dv[Vsum != 0, :]
    Dh = Dh[Hsum.flatten() !=0, :]
    Dt = Dt[Defective == 0]

    # Calculate bias corrected variance
    var_t  = np.var(DvDh.flatten(), ddof=1)
    var_h  = np.var(DtDv.flatten(), ddof=1)
    var_v  = np.var(DtDh.flatten(), ddof=1)
    var_th = np.var(  Dv.flatten(), ddof=1)
    var_tv = np.var(  Dh.flatten(), ddof=1)
    var_vh = np.var(  Dt.flatten(), ddof=1)

    # Calculate the total variance
    TT = T * (Defective.flatten() == 0).sum()

    var_tvh = (1 / (TT - 1) * (E2tvh.flatten() - TT * (E1tvh.flatten() / TT).sum()**2).sum())

    # Create the variance vector
    VarVector = np.vstack([var_t, var_v, var_h, var_tv, var_th, var_vh, var_tvh])

    # Adjust size of V, H, and their multiplication to discount defective pixels
    Vs = V - (Vsum==0).sum()
    Hs = H - (Hsum==0).sum()
    VHs = V * H - Defective.sum()

    # The mixing matrix derived in paper
    MDH = np.array([
        [1, 0, 0, 1/Vs, 1/Hs,    0, 1/( VHs)],
        [0, 1, 0, 1/ T,    0, 1/Hs, 1/(T*Hs)],
        [0, 0, 1,    0, 1/ T, 1/Vs, 1/(T*Vs)],

        [(T*Vs)/(T*Vs-1)-Vs/(T*Vs-1),
         (T*Vs)/(T*Vs-1)- T/(T*Vs-1),
         0,
         1,
         (Vs-T*Vs)/(Hs-T*VHs),
         (T -T*Vs)/(Hs-T*VHs),
         1/Hs],

        [(T*Hs)/(T*Hs-1)-Hs/(T*Hs-1),
         0,
         (T*Hs)/(T*Hs-1)-T/(T*Hs-1),
         (Hs-T*Hs)/(Vs-T*VHs),
         1,
         (T-T*Hs)/(Vs-T*VHs),
         1/Vs],

        [0,
         (VHs)/(VHs-1)-Hs/(VHs-1),
         (VHs)/(VHs-1)-Vs/(VHs-1),
         (Hs-VHs)/(T-T*VHs),
         (Vs-VHs)/(T-T*VHs),
         1,
         1/T],

        [(T*VHs)/(T*VHs-1)- (VHs)/(T*VHs-1),
         (T*VHs)/(T*VHs-1)-(T*Hs)/(T*VHs-1),
         (T*VHs)/(T*VHs-1)-(T*Vs)/(T*VHs-1),
         (T*VHs)/(T*VHs-1)-    Hs/(T*VHs-1),
         (T*VHs)/(T*VHs-1)-    Vs/(T*VHs-1),
         (T*VHs)/(T*VHs-1)-     T/(T*VHs-1), 1]]
    )

    # The classic mixing matrix (MDH under limit of infinite sampling)
    MC = np.array([[1,     0,     0,     0,     0,     0,     0],
                   [0,     1,     0,     0,     0,     0,     0],
                   [0,     0,     1,     0,     0,     0,     0],
                   [1,     1,     0,     1,     0,     0,     0],
                   [1,     0,     1,     0,     1,     0,     0],
                   [0,     1,     1,     0,     0,     1,     0],
                   [1,     1,     1,     1,     1,     1,     1]], dtype=float)

    VarDH = (np.linalg.inv(MDH) @ VarVector)
    VarC  = (np.linalg.inv(MC)  @ VarVector)
    VarRaw = VarVector

    return [VarDH.flatten(), VarC.flatten(), VarRaw.flatten(), MDH, MC]

#-------------------------------------------------------------------------------
# Unit test when called stand-alone
if __name__ == '__main__':

    # standard deviations in each dimension
    t   = 1
    v   = 2
    h   = 3
    tv  = 4
    th  = 5
    vh  = 6
    tvh = 7

    # size of test cube
    T = 100
    V = 500
    H = 500

    # Generate the test cube with desired standard deviations
    Cube, VarV, Nv = noise3d_sim(t, v, h, tv, th, vh, tvh, T, V, H)

    # Compute variances
    VarDH, VarC, VarRaw, MDH, MC = noise3d_calc(Cube)

    # Check that we're within some percent of true standard deviations
    ideal_stds = np.array([t, v, h, tv, th, vh, tvh])
    estim_stds = np.sqrt(VarDH)
    print(f'Ideal     standard deviations = {ideal_stds}')
    print(f'Estimated standard deviations = {estim_stds}')
    maxPctErr = 5
    if (np.any(np.abs(ideal_stds - estim_stds) * 100 / ideal_stds > maxPctErr)):
        print('ERROR: failed unit test')
    else:
        print('PASSED unit test(s)')
