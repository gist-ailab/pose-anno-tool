import os
import sys
import logging
# from src.annotator import Annotator
# import cv2

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

from multiprocessing import Process

from test_o3d import run_test_o3d

from easy_tcp_python2_3 import socket_utils as su

formClass = uic.loadUiType("./assets/ui/6d_pose_annotator.ui")[0]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent.findChild(QPlainTextEdit, "Logger")
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

class WindowClass(QMainWindow, formClass):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("GIST AILAB 6D Object Pose Annotator")

        logTextBox = QTextEditLogger(self)
        # You can format what is printed to text box
        logTextBox.setFormatter(
                logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s',
                datefmt = '%I:%M:%S',
                    ))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("hi debugger")


        # textBrowser_Logger

        ## load gist logo
        h = self.gistLogo.size().height()
        pixmap = QPixmap("./assets/imgs/gist_logo.png")
        pixmap = pixmap.scaledToWidth(h)
        pixmap = pixmap.scaledToHeight(h)
        self.gistLogo.setPixmap(pixmap)
        ## load ailab logo
        h = self.ailabLogo.size().height()
        pixmap = QPixmap("./assets/imgs/ailab_logo.png")
        pixmap = pixmap.scaledToWidth(h)
        pixmap = pixmap.scaledToHeight(h)
        self.ailabLogo.setPixmap(pixmap)

        ## object selecter
        for i in range(20):
            self.objectSelecter.insertItem(i, f"object_{i}")
        self.objectSelecter.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.objectSelecter.itemClicked.connect(self.select_object)

        ## [QtreeView] fileSystem
        self.fileSystem.setHeaderHidden(True)
        self.fileSystem.clicked.connect(self.select_anno_file)

        ## [Open Button] folder selecter
        self.openFolder.clicked.connect(self.select_working_dir)



        #WebEngineView의 시그널
        self.webEngineView.loadStarted.connect(self.printLoadStart)
        self.webEngineView.loadProgress.connect(self.printLoading)
        self.webEngineView.loadFinished.connect(self.printLoadFinished)

        #버튼들에 기능을 연결
        self.btn_back.clicked.connect(self.btnBackFunc)
        self.btn_forward.clicked.connect(self.btnForwardFunc)
        self.btn_reload.clicked.connect(self.btnRelaodFunc)
        self.btn_stop.clicked.connect(self.btnStopFunc)

    def keyPressEvent(self, event):
        su.sendall_pickle(self.sock, event.text())
        if event.key() == Qt.Key_Space:
            self.test_method()

    def test_method(self):
        pass

    def select_object(self):
        print()

    def select_working_dir(self):

        path_selected = QFileDialog.getExistingDirectory(self, "Open File")
        self.model = QFileSystemModel()
        self.model.setRootPath(path_selected)
        logging.info(f"set working directory to {path_selected}")
        self.index_root = self.model.index(self.model.rootPath())
        self.fileSystem.setModel(self.model)
        self.fileSystem.setRootIndex(self.index_root)
        for i in range(1, self.fileSystem.model().columnCount()):
            self.fileSystem.header().hideSection(i)



    def select_anno_file(self, index):

        indexItem = self.model.index(index.row(), 0, index.parent())
        filePath = self.model.filePath(indexItem)
        logging.debug("selected anno file: f{filePath}")
        fileName = self.model.fileName(indexItem)



    #WebEngineView의 시그널에 연결된 함수들
    def printLoadStart(self): print("Start Loading")
    def printLoading(self): print("Loading")
    def printLoadFinished(self): print("Load Finished")


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



if __name__ == "__main__":


    o3d_p = Process(target=run_test_o3d)
    o3d_p.start()
    sock, add = su.initialize_server('localhost', 5555)

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.sock = sock
    myWindow.show()
    app.exec_()