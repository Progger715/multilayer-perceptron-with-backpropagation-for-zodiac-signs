import os
import random
from pathlib import Path
from PIL import Image
from alive_progress import alive_bar

zodiac_signs = ["Aries", "Taurus", "Gemini",
                "Cancer", "Leo", "Virgo",
                "Libra", "Scorpio", "Sagittarius",
                "Capricorn", "Aquarius", "Pisces"]
image = None
weights = []
w_active = 0.1
bias = 0


def init_weights():
    for z in range(12):
        weights.append([])
        for i in range(32):
            weights[z].append([])
            for j in range(32):
                random_weight = random.random() * 2 - 1
                weights[z][i].append(random_weight)


def start_perceptron():
    init_weights()
    train_evenly()


def print_weight(number):
    for i in range(32):
        print(weights[number][i])


def print_all_weights():
    for i in range(12):
        print("\nweight for", zodiac_signs[i])
        print_weight(i)


def open_image(name_image):
    try:
        global image
        path = Path(Path.cwd().parent, "pictures for learning", name_image)
        image = Image.open(path)
        return image
    except FileNotFoundError as ex:
        print("[FILE NOT FOUND]\t", ex)


def identify_image(file_name=image):
    if file_name:
        open_image(file_name)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()  # массив пикселей
    ignore_pixel = 280  # 255 * 3
    net = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count = 0  # количество черных пикселей, отладочная информация

    for x in range(width):
        for y in range(height):
            r = pix[x, y][0]
            g = pix[x, y][1]
            b = pix[x, y][2]
            color_pixel = (r + g + b)
            if color_pixel < ignore_pixel:
                count += 1
                s = w_active
                # print(f"({x}, {y})\tr = {r}\tg = {g}\tb = {b}")
            else:
                s = 0
            for i in range(12):
                net[i] += s * weights[i][x][y]
    # print("net = ", net)
    # print("count = ", count)
    return choose_name_image(net)


def choose_name_image(net):
    res = []
    for i in range(12):
        if net[i] >= bias:
            res.append([net[i], i])
    # print("res = ", res)
    if len(res) > 1:
        max = 0
        max_index = 0
        for i in range(len(res)):
            if res[i][0] > max:
                max = res[i][0]
                max_index = res[i][1]
        return zodiac_signs[max_index]
    else:
        if len(res) > 0:
            return zodiac_signs[res[0][1]]
        else:
            return None


def add_weight(number):  # номер знака зодиака
    pix = image.load()
    white_pixel = 255 * 3
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r = pix[x, y][0]
            g = pix[x, y][1]
            b = pix[x, y][2]
            color_pixel = (r + g + b)
            if color_pixel != white_pixel:
                weights[number][x][y] += w_active


def reduce_weight(number):
    pix = image.load()
    white_pixel = 255 * 3
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r = pix[x, y][0]
            g = pix[x, y][1]
            b = pix[x, y][2]
            color_pixel = (r + g + b)
            if color_pixel != white_pixel:
                weights[number][x][y] -= w_active


def train_random():
    def find_index():
        index_in_mass_for_cute = files[number_file].find(".") - 3
        string_for_search = files[number_file][:index_in_mass_for_cute]
        for j in range(len(zodiac_signs)):
            if zodiac_signs[j].startswith(string_for_search):
                return j

    file_path = Path(Path.cwd().parent, "False_answers_history.txt")
    file_record_false = open(file_path, 'w')
    path = Path(Path.cwd().parent, "pictures for learning")
    for _, _, files in os.walk(path):
        pass
    count_training = 10000
    with alive_bar(count_training, dual_line=True) as bar:
        for i in range(count_training):
            # print("#", i)
            bar.text = '-> The system is being trained, please wait...'
            global image
            number_file = random.randint(0, len(files) - 1)
            image = open_image(files[number_file])  # files[number_file] - имя файла
            result = identify_image()  # имя знака зодиака, которое определила программа
            # print("Image = ", files[number_file])
            # print("Result = ", result)
            if result is None:
                # print(f"[None]\n")
                continue
            if files[number_file].startswith(result):
                # print(f"[TRUE]")
                number = zodiac_signs.index(result)
                # print("number true image = ", number)
                add_weight(number)
            else:
                # print("[FALSE]")
                number_false = zodiac_signs.index(result)
                # print("number false image = ", number_false)
                index = f'# {i}\n' \
                        f'Image = {files[number_file]}\nResult = {result}\n\n '
                file_record_false.write(index)
                buf_index = find_index()
                reduce_weight(number_false)
                add_weight(buf_index)
            bar()


def train_evenly():
    file_path = Path(Path.cwd().parent, "False_answers_history.txt")
    file_record_false = open(file_path, 'w')
    count_eras = 25
    with alive_bar(count_eras, dual_line=True) as bar:
        for number_era in range(count_eras):  # era
            # print(number_era)
            bar.text = '-> The system is being trained, please wait...'
            for number_var in range(1, 21):
                for number_sign in range(12):
                    # print("#", number_era, number_sign, number_var)
                    global image
                    file_name = zodiac_signs[number_sign] + f"{number_var}" + ".png"
                    image = open_image(file_name)
                    result = identify_image()  # имя знака зодиака, которое определила программа
                    # print("Image = ", file_name)
                    # print("Result = ", result)
                    if result is None:
                        # print(f"[None]\n")
                        continue
                    if file_name.startswith(result):
                        # print(f"[TRUE]\n")
                        number = zodiac_signs.index(result)
                        add_weight(number)
                    else:
                        # print("[FALSE]\n")
                        number_false = zodiac_signs.index(result)
                        reduce_weight(number_false)
                        add_weight(number_sign)
                        record = f'# {number_era} {zodiac_signs[number_sign]} {number_var}\n' \
                                 f'Image = {file_name}\nResult = {result}\n\n '
                        file_record_false.write(record)
            bar()
    file_record_false.write("\n\n\n\nend train\n\n\n\n")
    file_record_false.close()
    print("system has finished training")


def train_with_teacher(index_sign_zodiac, name_file):
    result = identify_image(name_file)
    if result == zodiac_signs[index_sign_zodiac]:
        add_weight(index_sign_zodiac)
    else:
        add_weight(index_sign_zodiac)
        print("res = ", result)
        if result is None:
            return "Знак не был определен"
        reduce_weight(zodiac_signs.index(result))
    return result


if __name__ == '__main__':
    init_weights()
    print("weight before training:")
    # print_all_weights()
    train_evenly()
    print("weight after training:")
    # print_all_weights()

