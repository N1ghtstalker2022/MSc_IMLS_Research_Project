import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Number of pixels N_side
N_side = 64
npix = hp.nside2npix(N_side)

# Creating an array of data
data = np.arange(npix)

# Visualizing the data
hp.mollview(data, title="HEALPix visualization", unit="mK", norm="hist", cmap="jet")
plt.show()
