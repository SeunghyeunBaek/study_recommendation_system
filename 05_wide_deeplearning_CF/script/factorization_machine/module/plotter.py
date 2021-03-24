from matplotlib import pyplot as plt

def plot_lines(x, y_train, y_val):
    
    fig = plt.figure(figsize=(12, 7))
    plt.plot(x, y_train, marker='.', c='blue', label='train')
    plt.plot(x, y_val, marker='.', c='red', label='validation')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    return fig