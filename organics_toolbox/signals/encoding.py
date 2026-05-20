import numpy as np

def joint_encode(x_theta, x_space, Wtheta_enc, Wspace_enc):
    
    y_theta = Wtheta_enc @ x_theta
    y_space = Wspace_enc @ x_space
    z = np.outer(y_theta, y_space)
    return z.reshape(-1)

