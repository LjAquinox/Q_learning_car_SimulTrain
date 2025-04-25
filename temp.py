import numpy as np

x = 1
y = 2
num_rays = 10
print(np.array([x, y]*num_rays).reshape(num_rays, 2))