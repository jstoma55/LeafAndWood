import matplotlib.pyplot as plt

def las_plt(data):
    fig = plt.figure(figsize=[20, 5])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(data.x, data.y, data.z, c=data.z ,s=0.1, marker='o', cmap="Spectral")
    plt.colorbar(sc)
    plt.show()