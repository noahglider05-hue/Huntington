import matplotlib.pyplot as plt
import numpy as np

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

print(len(plt.colormaps()))

for cmap in plt.colormaps():
    print(cmap)
    plt.figure(figsize=(6, 0.5))
    plt.title(cmap, fontsize=8)
    plt.imshow(gradient, aspect='auto', cmap=cmap)
    plt.axis('off')
    plt.show()

