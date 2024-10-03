import numpy as np


# function [VarDH,VarC,VarPSD,VarRaw,MDH,MC,Run]=Noise3d_Calc(Cube,Defective,Run)
def noise3d_calc(Cube, Defective=None):
    """
%Noise3d_Calc will calculate the variance for the 7 independent random 
% processes of the standard 3D model
%
%   Inputs:
%           Cube        : either 2D or 3D noise cube
%           Defective   : binary array detailing map of defective pixels
%                         value of 1 = defective, 0 = acceptable
%           Run         : either scalar for initialization or structure as
%                         created by previous run for building a running
%                         compuation of 3D noise results (when memory is
%                         insufficient for entire cube)
%
%
%   Outputs:
%           VarDH   : Unbiased 7x1 vector of variances estimates
%           VarC    : Classic 7x1 vector of variances estimates
%           VarPSD  : PSD 7x1 vector of variances estimates (not calculated
%                     in running moment geometry and ignores Defective)
%           VarRaw  : Raw variances after directional averaging
%           MDH     : Unbiased measurement mixing matrix
%           MC      : Classic measurement mixing matrix
%           Run     : Structure of running moments
%    
%   Outputs variance vectors are in order : t;v;h;tv;th;vh;tvh
% 
%   See also Noise3d_Example, Noise3d_CI, Noise3d_Sim 
%
% D. Haefner, NVESD January 2015
% 
% Additional details of method and rationale found in:
% Sampling Corrected 3D Noise with Confidence Intervals
% https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-15-4907
"""

    if Defective is None:
        Defective = (0 * Cube[:, :, 0]).astype(int)

    # Populate sizes
    V = Cube.shape[0]
    H = Cube.shape[1]

    # Verify global mean (ignoring defective) is removed...if running, this
    # should have already been done
    Omega = Cube.mean(axis=2)
    Omega = Omega[Defective == 0].mean()
    Cube = Cube - Omega

    # Set Defective pixels to zero to be ignored during summations
    Cube[Defective == 1, :] = 0

    T = Cube.shape[2]
    Dv = Cube.sum(axis=0)
    Dh = Cube.sum(axis=1)
    Dt = Cube.sum(axis=2)
    E1tvh = Cube.sum(axis=2)
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
    var_t = np.var(DvDh.flatten(), ddof=1)
    var_h = np.var(DtDv.flatten(), ddof=1)
    var_v = np.var(DtDh.flatten(), ddof=1)
    var_th = np.var(Dv.flatten(), ddof=1)
    var_tv = np.var(Dh.flatten(), ddof=1)
    var_vh = np.var(Dt.flatten(), ddof=1)

    # Calculate the total variance
    TT = T * (Defective.flatten() == 0).sum()

    var_tvh = (1 / (TT - 1) * (E2tvh.flatten() - TT * (E1tvh.flatten() / TT).sum()**2).sum())

    # Create the variance vector
    VarVector = np.vstack([var_t, var_v, var_h, var_tv, var_th, var_vh, var_tvh])

    # Adjust size of V, H, and their multiplication to discount defective
    # pixel effects

    Vs = V - (Vsum==0).sum()
    Hs = H - (Hsum==0).sum()
    VHs = V * H - Defective.sum()

    # The mixing matrix derived in paper
    MDH = np.array([[1,0,0,1/Vs,1/Hs,0,1/(VHs)],
        [0,1,0,1/T,0,1/Hs,1/(T*Hs)],
        [0,0,1,0,1/T,1/Vs,1/(T*Vs)],
        [(T*Vs)/(T*Vs-1)-Vs/(T*Vs-1),(T*Vs)/(T*Vs-1)-T/(T*Vs-1),0,1,(Vs-T*Vs)/(Hs-T*VHs),(T-T*Vs)/(Hs-T*VHs),1/Hs],
        [(T*Hs)/(T*Hs-1)-Hs/(T*Hs-1),0,(T*Hs)/(T*Hs-1)-T/(T*Hs-1),(Hs-T*Hs)/(Vs-T*VHs),1,(T-T*Hs)/(Vs-T*VHs),1/Vs],
        [0,(VHs)/(VHs-1)-Hs/(VHs-1),(VHs)/(VHs-1)-Vs/(VHs-1),(Hs-VHs)/(T-T*VHs),(Vs-VHs)/(T-T*VHs),1,1/T],
        [(T*VHs)/(T*VHs-1)-(VHs)/(T*VHs-1),(T*VHs)/(T*VHs-1)-(T*Hs)/(T*VHs-1),(T*VHs)/(T*VHs-1)-(T*Vs)/(T*VHs-1),(T*VHs)/(T*VHs-1)-Hs/(T*VHs-1),(T*VHs)/(T*VHs-1)-Vs/(T*VHs-1),(T*VHs)/(T*VHs-1)-T/(T*VHs-1),1]]
    )

    # The classic mixing matrix (MDH under limit of infinite sampling)
    MC = np.array([[1,     0,     0,     0,     0,     0,     0],
                   [0,     1,     0,     0,     0,     0,     0],
                   [0,     0,     1,     0,     0,     0,     0],
                   [1,     1,     0,     1,     0,     0,     0],
                   [1,     0,     1,     0,     1,     0,     0],
                   [0,     1,     1,     0,     0,     1,     0],
                   [1,     1,     1,     1,     1,     1,     1]], dtype=float)

    VarDH = ((np.linalg.inv(MDH) @ VarVector))
    VarC = ((np.linalg.inv(MC) @ VarVector))
    VarRaw = VarVector

    return [VarDH, VarC, VarRaw, MDH, MC]