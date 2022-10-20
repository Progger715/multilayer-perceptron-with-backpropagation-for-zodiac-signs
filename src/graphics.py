import matplotlib.pyplot as plt
from pathlib import Path


def draw_all_graphics():
    file_path_accuracy = Path(Path.cwd().parent, "output data", "value_accuracy.txt")
    file_path_loss = Path(Path.cwd().parent, "output data", "value_loss_function.txt")

    bar_eras = []
    bar_loss = []
    bar_accuracy = []

    for line in open(file_path_loss, "r"):
        bar_era, value_loss = line.split()
        bar_eras.append(float(bar_era))
        bar_loss.append(float(value_loss))
    for line in open(file_path_accuracy, "r"):
        _, value_accuracy = line.split()
        bar_accuracy.append(float(value_accuracy))

    print(bar_eras)
    print(bar_accuracy)
    print(bar_loss)

    plt.figure("graphic loss")
    plt.title("Loss function vs era")
    plt.xlabel("era")
    plt.ylabel("value loss")
    plt.plot(bar_eras, bar_accuracy, label='loss function')
    plt.legend()
    plt.grid(True)

    plt.figure("graphic accuracy")
    plt.title("accuracy vs era")
    plt.xlabel("era")
    plt.ylabel("accuracy")
    plt.plot(bar_eras, bar_loss, label='accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    draw_all_graphics()
