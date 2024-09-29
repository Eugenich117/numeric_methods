from tkinter import *
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

# функция для обработки нажатия на кнопку
def ButtonClick():
    # получаем количество столбцов из компонента Entry
    num_bins = entry.get()
    try:
        # преобразуем количество столбцов в целое число
        num_bins = int(num_bins)
        messagebox.showinfo(title='Success!', message="You have successfully built the Laplacian distribution!")
    except:
        # если в Entry не задано число, выводим сообщение
        messagebox.showwarning("Warning", "Please enter a valid number of bins")
        return
    # генерируем массив случайных чисел с распределением Лапласа
    data = np.random.laplace(loc=0, scale=1, size=1000)
    # создаем гистограмму
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, bins=num_bins, density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Laplace Distribution')
    # выводим гистограмму на форму программы
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().grid(row=2, column=0, sticky='news')

# создаем главное окно программы
root = Tk()
root.title("Histogram of Laplace Distribution")

# создаем компоненты формы
frame=Frame(root,bg='gray')
frame.place(relx=0.15, rely=0.15, relwidth=0.8, relheight= 0.8)
label = Label(frame, text="Number of bins:")

root.wm_attributes('-alpha', 0.8)
root.resizable(width=False, height=False)
root.geometry('800x660')


label.grid(row=0, column=4)
entry = Entry(frame)
entry.grid(row=1, column=4)
button = Button(frame, text="Generate Histogram", command=ButtonClick)
button.grid(row=2, column=3, columnspan=4)

# запускаем цикл обработки сообщений
root.mainloop()
