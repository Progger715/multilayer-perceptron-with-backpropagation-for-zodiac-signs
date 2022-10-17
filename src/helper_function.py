import random
import numpy as np

weights = []  # веса связей между нейронами
weights_delta = []  # веса связей между нейронами
I = []  # выходное значение нейронов до функции активации
delta = []  # считается при обратном проходе, пока хз
O = []  # выходное значение нейронов после функции активации
layer_neurons = {
    0: {'start': 0, 'end': 1},
    1: {'start': 2, 'end': 4},
    2: {'start': 5, 'end': 6},
    3: {'start': 7, 'end': 7},
}
alpha = 0.1  # скорость обучения
gamma = 0.1
size = 8  # количество нейронов


def init_I():
    for _ in range(size):
        I.append(0)
        delta.append(0)
        O.append(0)


def create_weights():
    for z in range(size):
        weights.append([])
        weights_delta.append([])
        for i in range(size):
            weights[z].append(0)
            weights_delta[z].append(0)


def init_weights():
    for z in range(size):
        for i in range(size):
            if i == z:
                continue
            random_weight = random.random() * 2 - 1
            weights[z][i] = weights[i][z] = random_weight


def print_weight():
    for i in range(size):
        print(weights[i])


def set_neuron_output_value(neuron_index, layer_index):  # подсчет I, не подается входной слой
    start = layer_neurons[layer_index - 1]['start']
    end = layer_neurons[layer_index - 1]['end']
    for i in range(start, end + 1, 1):
        I[neuron_index] += I[i] * weights[i][neuron_index]


def set_neuron_output_value_after_activation(neuron_index):
    O[neuron_index] = activation_function(I[neuron_index])


def activation_function(x):
    return 1 / (1 + np.exp(-x))


def get_derivative(x):
    buf = activation_function(x)
    return buf * (1 - buf)


def set_output_reverse_trip(neuron_index, true_value):
    delta[neuron_index] = (true_value - O[neuron_index]) * get_derivative(I[neuron_index])


def set_hidden_reverse_trip(neuron_index, layer_index):
    start = layer_neurons[layer_index + 1]['start']
    end = layer_neurons[layer_index + 1]['end']
    for i in range(start, end + 1, 1):
        delta[neuron_index] += delta[i] * weights[neuron_index][i]
    delta[neuron_index] *= get_derivative(I[neuron_index])


def change_weights():  # все границы включительно
    for a in range(len(layer_neurons) - 1, 0, -1):  # перебор по слоям
        start = layer_neurons[a]['start']
        end = layer_neurons[a]['end']
        for j in range(end, start - 1, -1):  # перебор по нейронам на слое
            start_j = layer_neurons[a - 1]['start']
            end_j = layer_neurons[a - 1]['end']
            for i in range(end_j, start_j - 1, -1):  # перебор по нейронам на предыдущем слое
                delta_w = alpha * delta[j] * O[i] + gamma * weights_delta[i][j]
                weights_delta[i][j] = delta_w
                weights[i][j] += delta_w


def working():  # одна тренировка для одной картинки
    create_weights()
    init_weights()
    # print_weight()
    init_I()
    I[1] = 1
    for i in range(layer_neurons[1]['start']):
        O[i] = I[i]

    # установим все значения нейронов до функции активации
    for layer in range(1, len(layer_neurons)):
        for i in range(layer_neurons[layer]['start'], layer_neurons[layer]['end'] + 1):
            set_neuron_output_value(i, layer)

    # установим все значения нейронов после функции активации
    for layer in range(1, len(layer_neurons)):
        for i in range(layer_neurons[layer]['start'], layer_neurons[layer]['end'] + 1):
            set_neuron_output_value_after_activation(i)

    # считаем дельту для выходного слоя
    for output_neuron in range(layer_neurons[3]["start"], layer_neurons[3]["end"] + 1):
        set_output_reverse_trip(output_neuron,
                                0.9)  # тру вэлью будет массивом, где 1 бдует верный нейрон, а остальные нули
        # добавить массив тру вальью, когда известна картинка

    # считаем дельту для скрытых слоев
    for hidden_layer in range(len(layer_neurons) - 2, 0, -1):
        for i in range(layer_neurons[hidden_layer]['end'], layer_neurons[hidden_layer]['start'] - 1, -1):
            set_hidden_reverse_trip(i, hidden_layer)

    # меняем веса
    change_weights()


if __name__ == '__main__':
    working()
