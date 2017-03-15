import numpy as np

actions = 15
step = 0.005

for i in range (0,16):
    actions=i

    if actions % 2 == 0:
        step = -0.005

        position = np.ceil(actions/2)
    else:
        step = 0.005
        position = np.ceil(actions/2)


