from tkinter import *
from PIL import Image, EpsImagePlugin
from tkinter.ttk import Combobox
from tkinter import scrolledtext
from pathlib import Path
# import perceptron
import helper_function

canvas_width = 128
canvas_height = 128
brush_size = 3
color = "black"
canvas = ''
result = ''
root = Tk()
zodiac_signs_rus = {"Aries": "Овен", "Taurus": "Телец", "Gemini": "Близнецы",
                    "Cancer": "Рак", "Leo": "Лев", "Virgo": "Дева",
                    "Libra": "Весы", "Scorpio": "Скорпион", "Sagittarius": "Стрелец",
                    "Capricorn": "Козерог", "Aquarius": "Водолей", "Pisces": "Рыбы"}


def paint(event):
    global brush_size
    global color
    x1 = event.x - brush_size
    x2 = event.x + brush_size
    y1 = event.y - brush_size
    y2 = event.y + brush_size
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)


def delete():
    canvas.delete("all")
    result.delete(1.0, END)


def save():
    buffer_picture = Path(Path.cwd().parent, "picture reads", "Image.ps")
    original_picture = Path(Path.cwd().parent, "picture reads", "tmp100.png")
    compressed_image = Path(Path.cwd().parent, "picture reads", "Image.png")
    canvas.postscript(file=buffer_picture, colormode="color")
    # !!!надо изменить путь до исполняемого файла в строке ниже
    EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.56.1\bin\gswin64c'
    img = Image.open(buffer_picture)
    img.save(original_picture, "png")
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    img.save(compressed_image, "png")


def get_result():
    save()
    image_for_identy = Path(Path.cwd().parent, "picture reads", "Image.png")
    # answer = perceptron.identify_image(image_for_identy)
    index_answer = helper_function.identify_image(image_for_identy)
    answer = f" - {zodiac_signs_rus[index_answer]}"
    result.delete(1.0, END)
    result.insert(INSERT, answer)


def select():
    new_window = Toplevel()
    new_window.geometry("310x150")
    new_window.resizable(False, False)
    Label(new_window, text="Выберите, какой знак зодиака был Вами нарисован: \n").pack()
    combo = Combobox(new_window)
    combo['values'] = (
        "Овен", "Телец", "Близнецы", "Рак", "Лев", "Дева", "Весы", "Скорпион", "Стрелец", "Козерог", "Водолей", "Рыбы",)
    combo.current(1)
    combo.pack(pady=20)

    def get_combobox():
        print(combo)
        print(type(combo))
        index = combo.current()
        selected_sign = combo.get()
        new_window.destroy()
        save()
        # training
        image_for_identy = Path(Path.cwd().parent, "picture reads", "Image.png")
        # perceptron.init_weights()

        # answer = perceptron.train_with_teacher(index, image_for_identy)
        helper_function.identify_image_for_train(image_for_identy, index)
        answer = helper_function.choose_name_image()
        answer += f" - {zodiac_signs_rus[answer]}"
        message = f"Сеть прошла тренировку.\nНа картинке: {selected_sign}\nСеть обнаружила: {answer}"
        result.delete(1.0, END)
        result.insert(INSERT, message)

    Button(new_window, text="Выбрать", command=get_combobox).pack()


def create_window():
    root.title("Perceptron")
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    width = width // 2
    height = height // 2
    width = width - 200
    height = height - 200
    root.geometry('300x350+{}+{}'.format(width, height))
    root.resizable(False, False)

    global canvas
    canvas = Canvas(width=canvas_width, height=canvas_height, bg="white")
    canvas.bind("<B1-Motion>", paint)
    canvas.pack()

    delete_btn = Button(text="Очистить", width=10, command=delete)
    delete_btn.pack(side=BOTTOM, pady=10)

    learning_btn = Button(root, text="Выбрать и обучить", width=30, command=select)
    learning_btn.pack(side=BOTTOM, pady=10)

    save_btn = Button(text="Распознать", width=10, command=get_result)
    save_btn.pack(side=BOTTOM, pady=10)

    global result
    result = scrolledtext.ScrolledText(root, width=30, height=2)
    result.pack(side=TOP, pady=20)


if __name__ == "__main__":
    create_window()
    helper_function.train()
    # perceptron.start_perceptron()
    # perceptron.print_all_weights()
    root.mainloop()
