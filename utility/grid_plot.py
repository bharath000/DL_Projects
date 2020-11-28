import numpy as np
import matplotlib.pyplot as plt

def plot_images(images):
    
    w=10
    h=10
    fig=plt.figure(figsize=(w, h))
    columns = 8
    rows = 2
    for i in range(1, columns*rows +1):
        #img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[0][i-1])
        plt.xlabel(str(images[1][i-1]))
    plt.show()