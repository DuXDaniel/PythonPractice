import math
import time
import sys
import socket
import subprocess
import random
import datetime
import threading
import queue
import os.path as path
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import PhotoImage
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog

def main(argv):
    windowWidget = WidgetGallery()

class WidgetGallery():
    def __init__(self, parent=None):
        self.mainWindow = tk.Tk()
        self.stepHistoryTable = ttk.Treeview(
            self.mainWindow,
            columns=("Step","Timepoint","Filename","Status"),
            show="headings"
        )
        self.stepHistoryTable.heading("Step",text="Step")
        self.stepHistoryTable.column("Step",minwidth=0)
        self.stepHistoryTable.heading("Timepoint",text="Timepoint")
        self.stepHistoryTable.column("Timepoint",minwidth=0)
        self.stepHistoryTable.heading("Filename",text="Filename")
        self.stepHistoryTable.column("Filename",minwidth=0)
        self.stepHistoryTable.heading("Status",text="Status")
        self.stepHistoryTable.column("Status",minwidth=0)

        self.stepHistoryTable.grid(column=0, row=0, columnspan=2, padx=5, pady=5, sticky="e")

        self.addbutton = ttk.Button(
            self.mainWindow,
            text="Add",
            command=lambda: self.addRow()
        )

        self.editbutton = ttk.Button(
            self.mainWindow,
            text="Edit",
            command=lambda: self.editRow()
        )

        self.addbutton.grid(column=0, row=1, columnspan=2, padx=5, pady=5, sticky="e")
        self.editbutton.grid(column=0, row=2, columnspan=2, padx=5, pady=5, sticky="e")
        
        self.mainWindow.mainloop()
        return
    
    def addRow(self):
        self.stepHistoryTable.insert("",'end',values=('a','b','c','d'))
        return

    def editRow(self):
        childRow = self.stepHistoryTable.get_children()[-1]
        self.stepHistoryTable.item(childRow,values=("e",'e','e','e'))
        return

if __name__ == '__main__':
    main(sys.argv)
