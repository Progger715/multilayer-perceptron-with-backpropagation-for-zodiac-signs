import random
from pathlib import Path
from PIL import Image

image = None


def open_image(name_image):
    try:
        global image
        path = Path(Path.cwd().parent, "pictures for learning", name_image)
        image = Image.open(path)
        return image
    except FileNotFoundError as ex:
        print("[FILE NOT FOUND]\t", ex)


def read_values_image(I, file_name=image):
    open_image(file_name)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()  # массив пикселей
    ignore_pixel = 280  # 255 * 3

    count = 0  # количество черных пикселей, отладочная информация
    index_I = 0
    for y in range(height):
        for x in range(width):
            r = pix[x, y][0]
            g = pix[x, y][1]
            b = pix[x, y][2]
            color_pixel = (r + g + b)
            if color_pixel < ignore_pixel:
                I[index_I] = 1
            index_I += 1
    return I


if __name__ == '__main__':
    pass
