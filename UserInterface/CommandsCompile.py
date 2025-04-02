def PromptButtonCommand(box1, box2):
    text = box1.get("1.0", "end-1c")
    box2.configure(state = "normal")
    box2.insert("end", text + "\n")
    box2.configure(state = "disabled")