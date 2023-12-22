
import numpy as np
import matplotlib.pyplot as plt
from models import * 
from lightly.transforms.simclr_transform import SimCLRTransform




x = np.linspace(-6, 10, 10000)

def my_tanh(x, a=1):
    numerator = np.exp(a*x) - np.exp(-a*x)
    denominator = np.exp(a*x) + np.exp(-a*x)
    return (numerator/denominator + 1)/2

y = my_tanh(x, a=5)

plt.plot(x, y)
plt.show()























