import random
import numpy as np
from pathlib import Path
import time
import reader_image

weights = []  # веса связей между нейронами
weights_delta = []  # веса связей между нейронами
I = []  # выходное значение нейронов до функции активации
delta = []  # считается при обратном проходе, пока хз
O = []  # выходное значение нейронов после функции активации
layer_neurons = {
    0: {'start': 0, 'end': 1023},
    1: {'start': 1024, 'end': 1535},
    2: {'start': 1536, 'end': 1791},
    3: {'start': 1792, 'end': 1803},
}
alpha = 0.1  # скорость обучения
gamma = 0.1
size = 1804  # 1024+512+256+12 количество нейронов


def create_empty_I_delta_O():
    for _ in range(size):
        I.append(0)
        delta.append(0)
        O.append(0)


def clean_weights_delta_and_delta():
    for i in range(size):
        delta[i] = 0
        for j in range(size):
            weights_delta[i][j] = 0


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


def print_I():
    print("I")
    z = 0
    for i in range(32):
        for j in range(32):
            print(I[z], end=' ')
            z += 1
        print()
    print()


def print_O():
    print("O")
    z = 0
    for i in range(32):
        for j in range(32):
            print(O[z], end=' ')
            z += 1
        print()
    print()


def set_neuron_output_value(neuron_index, layer_index):  # подсчет I, не подается входной слой
    start = layer_neurons[layer_index - 1]['start']
    end = layer_neurons[layer_index - 1]['end']
    for i in range(start, end + 1, 1):
        I[neuron_index] += O[i] * weights[i][neuron_index]
        set_neuron_output_value_after_activation(neuron_index)


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


def identify_image(file_name, true_index_value):  # одна тренировка для одной картинки
    path = Path(Path.cwd().parent, "pictures for learning", file_name)
    global I
    I = reader_image.read_values_image(I, path)
    # print_I()

    for i in range(layer_neurons[1]['start']):
        O[i] = I[i]
    # print_O()
    # установим все значения нейронов до функции активации
    for layer in range(1, len(layer_neurons)):
        for i in range(layer_neurons[layer]['start'], layer_neurons[layer]['end'] + 1):
            # print(f"i = {i}, layer = {layer}")
            set_neuron_output_value(i, layer)

    # установим все значения нейронов после функции активации
    # for layer in range(1, len(layer_neurons)):
    #     for i in range(layer_neurons[layer]['start'], layer_neurons[layer]['end'] + 1):
    #         set_neuron_output_value_after_activation(i)

    # считаем дельту для выходного слоя
    for output_neuron in range(layer_neurons[3]["start"], layer_neurons[3]["end"] + 1):
        index = output_neuron - 1792
        if index == true_index_value:
            set_output_reverse_trip(output_neuron, 1)
        elif index != true_index_value:
            set_output_reverse_trip(output_neuron, 0)
            # тру вэлью будет массивом, где 1 бдует верный нейрон, а остальные нули
        # добавить массив тру вальью, когда известна картинка

    # считаем дельту для скрытых слоев
    for hidden_layer in range(len(layer_neurons) - 2, 0, -1):
        for i in range(layer_neurons[hidden_layer]['end'], layer_neurons[hidden_layer]['start'] - 1, -1):
            set_hidden_reverse_trip(i, hidden_layer)

    # меняем веса
    change_weights()


def train():
    count_eras = 5
    zodiac_signs = ["Aries", "Taurus", "Gemini",
                    "Cancer", "Leo", "Virgo",
                    "Libra", "Scorpio", "Sagittarius",
                    "Capricorn", "Aquarius", "Pisces"]
    create_weights()
    init_weights()
    create_empty_I_delta_O()
    # true_sign_zodiac = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for number_era in range(count_eras):  # era
        print("era = ", number_era)
        for number_var in range(1, 17):
            for number_sign in range(12):
                # true_sign_zodiac[number_sign] = 1
                file_name = f"{zodiac_signs[number_sign]}{number_var}.png"
                print(file_name)
                identify_image(file_name, number_sign)
                print(zodiac_signs[choose_name_image()], "\n")
                clean_weights_delta_and_delta()
                # true_sign_zodiac[number_sign] = 0


def choose_name_image():
    max = -1
    index_true_value = 0
    for i in range(0, 12, 1):
        print(f"{i} = {O[i + 1792]}")
        if O[i + 1792] > max:
            max = O[i + 1792]
            index_true_value = i
    return index_true_value


if __name__ == '__main__':
    start_time = time.time()
    train()
    # end_time = time.time()
    print(f"{time.time() - start_time} seconds")
    # create_empty_I_delta_O()
    # path = Path(Path.cwd().parent, "pictures for learning", "Aquarius1.png")
    # reader_image.read_values_image(I, path)
    # print_I()
