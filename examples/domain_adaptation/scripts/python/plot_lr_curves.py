import math
import matplotlib.pyplot as plt
x = []
def inv(base_lr, gamma, it, power):
    return base_lr * math.pow((1 + gamma * it), -power)
for i in range(0, 50000):
    x.append( inv(.00001, .00001, i, 2))
plt.plot(range(0, 50000), x)
plt.show()