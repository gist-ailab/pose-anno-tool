import os
import sys
import logging
# from src.annotator import Annotator
# import cv2

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtTest import QTest

from multiprocessing import Process

from src.annotator import runObjectPoseAnnotator

from os.path import basename, dirname, abspath
import json
from pathlib import Path
import time

formClass = uic.loadUiType("./assets/ui/6d_pose_annotator.ui")[0]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent.findChild(QPlainTextEdit, "logger")
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

class WindowClass(QMainWindow, formClass):

    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.setWindowTitle("GIST AILAB 6D Object Pose Annotator")

        self.commLogfile = open("./comm.json", "w")

        logTextBox = QTextEditLogger(self)
        # You can format what is printed to text box
        logTextBox.setFormatter(
                logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s',
                datefmt = '%I:%M:%S',
                    ))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("start program")


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

        ## [QtreeView] dataDirSelecter
        self.dataDirSelecter.setHeaderHidden(True)
        # self.dataDirSelecter.clicked.connect(self.selectAnnoData)
        self.setDataDirRoot(abspath(os.getcwd()) + "/data/lm/test/000002")
        ## [Open Button] openDataDir 
        self.openDataDirButton.clicked.connect(self.openDataDir)
        ## [Select Button] selectAnnoDataButton
        self.selectAnnoDataButton.clicked.connect(self.selectAnnoData)

        ## [QtreeView] objectDirSelecter
        self.objectDirSelecter.setHeaderHidden(True)
        # self.objectDirSelecter.clicked.connect(self.selectAnnoObject)
        ## [Open Button] openObjectDir 
        self.openObjectDirButton.clicked.connect(self.openObjectDir)

        ## [Add Button] addAnnoObjectButton
        self.addAnnoObjectButton.clicked.connect(self.addAnnoObject)
        ## [Delete Button] addAnnoObjectButton
        self.deleteAnnoObjectButton.clicked.connect(self.deleteAnnoObject)


        self.webEngineView.load(QUrl("http://localhost:8888/"))

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
        logging.info("Key input: <" + event.text() + ">")
        data = {
            "keyInput": event.text()
        }
        self.sendData("keyPress", data)
        self.webEngineView.zoomFactor()
        # for i in range(10):
        # QTest.mouseClick(self.webEngineView, Qt.LeftButton, pos=QPoint(512, 360))
        #     time.sleep(0.01)

    def sendData(self, action, data):
        logData = {"action": action, "data": data}
        self.commLogfile.write("{}\n".format(
                json.dumps(logData), ensure_ascii=False))
        self.commLogfile.flush()
        time.sleep(0.01)

    def test_method(self):
        pass


    def openDataDir(self):

        pathSelected = QFileDialog.getExistingDirectory(self, "Open File")
        self.setDataDirRoot(pathSelected)

    def selectAnnoData(self):
        index = self.dataDirSelecter.selectedIndexes()[0]
        info = self.dataDirSelecter.model().fileInfo(index)
        filePath = info.absoluteFilePath()
        fileName = basename(filePath)

        logging.debug(f"selected anno data: {filePath}")
        if basename(dirname(filePath)) == "rgb" and fileName.split(".")[-1] == "png":
            annoRGBPath = filePath
            annoDepthPath = annoRGBPath.replace(f"rgb/{fileName}", f"depth/{fileName}")
            annoSceneCameraPath = annoRGBPath.replace(f"rgb/{fileName}", f"scene_camera.json")
            annoScenesPath = str(Path(annoRGBPath).parent.parent.parent)
            annoObjectDirPath = os.path.join(str(Path(annoRGBPath).parent.parent.parent.parent), "models")
            if not os.path.exists(annoDepthPath):
                logging.warn(f"File does not exist: {annoDepthPath}")
            if not os.path.exists(annoSceneCameraPath):
                logging.warn(f"File does not exist: {annoSceneCameraPath}")
            if not os.path.exists(annoScenesPath):
                logging.warn(f"File does not exist: {annoScenesPath}")
            if not os.path.exists(annoObjectDirPath):
                logging.warn(f"File does not exist: {annoObjectDirPath}")
            data = {
                "annoRgbPath": annoRGBPath,
                "annoDepthPath": annoDepthPath,
                "annoSceneCameraPath": annoSceneCameraPath,
                "annoScenesPath": annoScenesPath,
                "annoObjectDirPath": annoObjectDirPath
            }
            self.sendData("selectAnnoData", data)
            # auto import object models
            self.setObjectDirRoot(annoObjectDirPath)


    def setDataDirRoot(self, path):
        self.modelFS = QFileSystemModel()
        self.modelFS.setRootPath(path)
        logging.info(f"set data directory to {path}")
        self.indexRootFS = self.modelFS.index(self.modelFS.rootPath())
        self.dataDirSelecter.setModel(self.modelFS)
        self.dataDirSelecter.setRootIndex(self.indexRootFS)
        for i in range(1, self.dataDirSelecter.model().columnCount()):
            self.dataDirSelecter.header().hideSection(i)

    def openObjectDir(self):
        pathSelected = QFileDialog.getExistingDirectory(self, "Open File")
        self.setObjectDirRoot(pathSelected)
        data = {
            "annoObjectDirPath": pathSelected
        }
        self.sendData("openObjectDir", data)

    def setObjectDirRoot(self, path):
        self.modelOS = QFileSystemModel()
        self.modelOS.setRootPath(path)
        logging.info(f"set object directory to {path}")
        self.indexRootOS = self.modelOS.index(self.modelOS.rootPath())
        self.objectDirSelecter.setModel(self.modelOS)
        self.objectDirSelecter.setRootIndex(self.indexRootOS)
        for i in range(1, self.objectDirSelecter.model().columnCount()):
            self.objectDirSelecter.header().hideSection(i)


    def addAnnoObject(self, index):
        index = self.objectDirSelecter.selectedIndexes()[0]
        info = self.objectDirSelecter.model().fileInfo(index)
        filePath = info.absoluteFilePath()
        fileName = basename(filePath)
        logging.debug(f"selected anno object: {fileName}")
        if fileName.split(".")[-1] == "ply":
            data = {
                "annoObjectPath": filePath
            }
            self.sendData("addAnnoObject", data)

    def deleteAnnoObject(self, index):
        index = self.objectDirSelecter.selectedIndexes()[0]
        info = self.objectDirSelecter.model().fileInfo(index)
        filePath = info.absoluteFilePath()
        logging.warn("Not implemented yet!")
        # fileName = basename(filePath)
        # logging.debug(f"selected anno object: {fileName}")
        # if fileName.split(".")[-1] == "ply":
        #     self.sendData("annoObjectPath", filePath)

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

    # run open3d annotator as a subprocess
    # communicate via webRTC and Json
    o3d_p = Process(target=runObjectPoseAnnotator)
    o3d_p.start()

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()