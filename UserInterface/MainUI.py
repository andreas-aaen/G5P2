import customtkinter
import tkinter
import CommandsCompile
from CommandsCompile import PromptButtonCommand

Program = customtkinter.CTk()
Frames = customtkinter.CTkFrame
Buttons = customtkinter.CTkButton
Entries = customtkinter.CTkEntry
Labels = customtkinter.CTkLabel
TB = customtkinter.CTkTextbox

Program.title("Technician Notes AI")
Program.geometry("800x800")
Program.resizable(False, False)


InputFrame = Frames(master=Program,
                                    width=400,
                                    height=600,
                                    corner_radius=20,
                                    bg_color="blue")

OutputFrame = Frames(master= Program,
                                    width=400,
                                    height=600,
                                    corner_radius=20,
                                    bg_color="red")

TitleFrame = Frames(master= Program,
                                    width=800,
                                    height=200,
                                    corner_radius=20,
                                    bg_color="yellow")

Title = Labels(master=TitleFrame,
               width=800,
               height=200,
               text="Technician AI",
               wraplength = 800,
               font=("Arial", 24))

LogsTitle = Labels(master=InputFrame,
               width=100,
               height=150,
               text="Previous entries:",
               font=("Arial", 12))

LogsTextbox = TB(master = InputFrame,
                width=350,
                height=400,)
LogsTextbox.configure(state="disabled")

PromptTextbox = TB(master = InputFrame,
                width=300,
                height=100)

PromptButton = Buttons(master=InputFrame,
                       width = 100,
                       height = 100,
                       text="Enter",
                       command = lambda: PromptButtonCommand(PromptTextbox, LogsTextbox))




# Placements
Title.place(x = 0, y = 0)

InputFrame.place(x=0, y=200)
OutputFrame.place(x = 400, y = 200)
TitleFrame.place(x=0, y=0)

PromptTextbox.place(x=0, y=500)
PromptButton.place(x=300, y=500)

LogsTitle.place(x=0, y=0)
LogsTextbox.place(x= 20, y=90)
Program.mainloop()
