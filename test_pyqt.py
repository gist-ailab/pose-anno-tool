import os
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

from multiprocessing import Process


from test_o3d import run_test_o3d

form_class = uic.loadUiType("./assets/ui/6d_pose_annotator.ui")[0]


class WindowClass(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("GIST AILAB 6D Object Pose Annotator")

        ## load gist logo
        self.label_gistlogo = self.findChild(QLabel, "QLabel_GISTLogo")
        h = self.label_gistlogo.size().height()
        pixmap = QPixmap("./assets/imgs/gist_logo.png")
        pixmap = pixmap.scaledToWidth(h)
        pixmap = pixmap.scaledToHeight(h)
        self.label_gistlogo.setPixmap(pixmap)
        
        self.label_ailablogo = self.findChild(QLabel, "QLabel_AILABLogo")
        h = self.label_ailablogo.size().height()
        pixmap = QPixmap("./assets/imgs/ailab_logo.jpg")
        pixmap = pixmap.scaledToWidth(h)
        pixmap = pixmap.scaledToHeight(h)
        self.label_ailablogo.setPixmap(pixmap)

        ## object selecter
        self.listwidget_object = self.findChild(QListWidget, "QListWidget_Object")
        for i in range(20):
            self.listwidget_object.insertItem(i, f"object_{i}")
        self.listwidget_object.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.listwidget_object.itemClicked.connect(self.object_select)

        #WebEngineView의 시그널
        self.webEngineView_Test.loadStarted.connect(self.printLoadStart)
        self.webEngineView_Test.loadProgress.connect(self.printLoading)
        self.webEngineView_Test.loadFinished.connect(self.printLoadFinished)
        self.webEngineView_Test.urlChanged.connect(self.urlChangedFunction)

        #버튼들에 기능을 연결
        self.btn_setUrl.clicked.connect(self.urlGo)
        self.btn_back.clicked.connect(self.btnBackFunc)
        self.btn_forward.clicked.connect(self.btnForwardFunc)
        self.btn_reload.clicked.connect(self.btnRelaodFunc)
        self.btn_stop.clicked.connect(self.btnStopFunc)

    #WebEngineView의 시그널에 연결된 함수들
    def printLoadStart(self): print("Start Loading")
    def printLoading(self): print("Loading")
    def printLoadFinished(self): print("Load Finished")

    def urlChangedFunction(self):
        self.line_url.setText(self.webEngineView_Test.url().toString())
        print("Url Changed")

    #버튼을 눌렀을 때 실행될 함수들
    def urlGo(self):
        self.webEngineView_Test.load(QUrl(self.line_url.text()))

    def btnBackFunc(self):
        self.webEngineView_Test.back()

    def btnForwardFunc(self):
        self.webEngineView_Test.forward()

    def btnRelaodFunc(self):
        self.webEngineView_Test.reload()

    def btnStopFunc(self):
        self.webEngineView_Test.stop()

    def object_select(self):
        print()

if __name__ == "__main__":

    o3d_p = Process(target=run_test_o3d)
    o3d_p.start()

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()