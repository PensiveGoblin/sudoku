import tkinter as tk
import numpy as np

input_sud=np.array([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])

window = tk.Tk()



for i in range(3):
	window.columnconfigure(i, weight=1, minsize=75)
	window.rowconfigure(i, weight=1, minsize=50)

	for j in range(3):
		frame = tk.Frame(
			master=window,
			relief=tk.RAISED,
			borderwidth=1
		)
		frame.grid(row=i, column=j, padx=5, pady=5)
		label = tk.Label(master=frame, text=f"Row {i}\nColumn{j}", font=("Arial", 50))
		label.pack(padx=5, pady=5)

window.mainloop()