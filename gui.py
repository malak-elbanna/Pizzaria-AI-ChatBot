from tkinter import *
from chat import pizzaria_response, bot_name


root = Tk()
root.title("Pizzaria ChatBot")
root.geometry('500x570+100+30')


def click():
    user_input = questionPlace.get()
    bot_answer = pizzaria_response(user_input)
    chatWindow.insert(END, "\n" + "You: " + user_input + "\n" )
    chatWindow.insert(END, "\n" + f"{bot_name}: " + bot_answer + "\n")
    questionPlace.delete(0, END)

centerFrame = Frame(root)
centerFrame.pack()

my_scrollbar = Scrollbar(centerFrame)
my_scrollbar.pack(side=RIGHT)


chatWindow = Text(centerFrame, font=('times new roman',12,'bold'), bd =1, bg= "lightblue", width= 80, height=20, yscrollcommand=my_scrollbar.set, wrap='word')
chatWindow.pack(side=LEFT)
my_scrollbar.config(command=chatWindow.yview)

questionPlace = Entry(root, font=('Arial', 12))
questionPlace.pack(fill=X, pady=30)

pizzaria_LOGO = PhotoImage(file='rsz_3rsz_1pizzaria-logo.png')
Button = Button(root, image=pizzaria_LOGO, bg="#1e9ae0", activebackground='#e1b2f0', width=11, height=4, font=('Arial', 10, 'bold'), command=click)
Button.place(x=20, y=450, height=70, width=100)

root.mainloop()