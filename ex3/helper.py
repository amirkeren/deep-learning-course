import matplotlib.pyplot as plt

from keras.callbacks import Callback

def plot_image(data):
    plt.figure(figsize=(20, 2))
    for i in range(10):
        ax = plt.subplot(1, 10, i + 1)
        plt.imshow(data[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


class Evaluator(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        plot_image(self.test_data)
