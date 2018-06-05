import numpy as np

# Martin
#ims = np.load("../miccai_data_noskull/train_ims_aligned_noskull.npy")
# Pola
ims = np.load("../MICCAI_data/noskull/train_ims_aligned_noskull.npy")
I1 = ims[0,:,:,:]
I2 = ims[5,:,:,:]
f = 5
I1_low = I1[0::f,0::f,0::f]
I2_low = I2[0::f,0::f,0::f]
# Martin
np.save("I1_low.npy", I1_low)
np.save("I5_low.npy", I2_low)
# Pola
np.save("../MICCAI_data/lowres/I1_low.npy", I1_low)
np.save("../MICCAI_data/lowres/I5_low.npy", I2_low)

# Martin
#lab = np.load("../miccai_data_noskull/train_labs_aligned_noskull.npy")
# Pola
lab = np.load("../MICCAI_data/noskull/train_labs_aligned_noskull.npy")
I1 = lab[0,:,:,:]
I2 = lab[5,:,:,:]
I1_low = I1[0::f,0::f,0::f]
I2_low = I2[0::f,0::f,0::f]
# Martin
# np.save("L1_low.npy", I1_low)
# np.save("L5_low.npy", I2_low)
# Pola
np.save("../MICCAI_data/lowres/L1_low.npy", I1_low)
np.save("../MICCAI_data/lowres/L5_low.npy", I2_low)
