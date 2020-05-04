#!bin/bash
"""
VIAN - Visual Movie Annotation and Analysis


copyright by
Gaudenz Halter
University of Zurich
2017

Visualization and MultimediaLab

"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import platform
import sys


import json

abspath = os.path.abspath(sys.executable)
dname = os.path.dirname(abspath)

import logging
logging.getLogger('tensorfyylow').disabled = True
import csv
# from OpenGL import GL

from functools import partial

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import glob
from random import sample

RANGE0 = 10000
RANGE1 = 20000


class MW(QMainWindow):
    def __init__(self, widget):
        super(MW, self).__init__()
        self.dataset = []
        all_files = glob.glob("data/thumbnails/*.jpg")
        all_files = all_files[RANGE0:RANGE1]
        self.files = sorted(sample(all_files, 800))

        for f in self.files:
            self.dataset.append(dict(path=f, name=os.path.split(f)[1], annotation=""))

        self.current_index = 0

        self.central = QWidget(self)
        self.central.setLayout(QVBoxLayout())

        self.v = QLabel(self)
        self.lineedit = QLineEdit(self)
        # self.lineedit.editingFinished.connect(self.on_change)
        self.setCentralWidget(self.central)
        self.hbox = QHBoxLayout(self)

        self.central.layout().addItem(self.hbox)
        self.centralWidget().layout().addWidget(self.v)

        self.btn_prev = QPushButton("<<", self)
        self.btn_next= QPushButton(">>", self)

        self.hbox.addWidget(self.btn_prev)
        self.hbox.addWidget(self.lineedit)
        self.hbox.addWidget(self.btn_next)

        self.btn_save = QPushButton("Save")
        self.btn_export = QPushButton("Export")
        self.centralWidget().layout().addWidget(self.btn_save)
        self.centralWidget().layout().addWidget(self.btn_export)

        self.btn_save.clicked.connect(self.on_save_json)
        self.btn_export.clicked.connect(self.on_save)

        self.btn_prev.clicked.connect(partial(self.on_change, -1))
        self.btn_next.clicked.connect(partial(self.on_change, 1))

        if os.path.isfile("saved.json"):
            self.load()

        self.on_change(incr=None)
        self.show()


    def on_change(self, incr=1):
        self.dataset[self.current_index]['annotation'] = self.lineedit.text()

        if incr is None:
            for idx, i in enumerate(self.dataset):
                if i['annotation'] != "":
                    self.current_index = idx
        else:
            self.current_index += incr


        print(self.current_index)
        if self.current_index >= 0 and self.current_index < len(self.dataset):
            px = QPixmap(self.files[self.current_index])
            self.v.setPixmap(px)
            self.lineedit.setText(self.dataset[self.current_index]['annotation'])

    def on_save_json(self):
        with open("saved.json", "w") as f:
            json.dump( dict(data = self.dataset, files = self.files), f)

    def load(self):
        with open("saved.json", "r") as f:
            d = json.load(f)
            self.files = d['files']
            self.dataset = d['data']

            self.on_change(0)

    def on_save(self):
        with open("saved.csv", "w") as f:
            writer = csv.writer(f)
            for t in self.dataset:
                writer.writerow([t['name'], t['annotation']])


app = QApplication(sys.argv)

main = MW(None)
main.show()
sys.exit(app.exec_())



