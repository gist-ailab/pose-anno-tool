# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import os
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import json

def tail(stream_file):
    """ Read a file like the Unix command `tail`. Code from https://stackoverflow.com/questions/44895527/reading-infinite-stream-tail """
    stream_file.seek(0, os.SEEK_END)  # Go to the end of file

    while True:
        if stream_file.closed:
            raise StopIteration

        line = stream_file.readline()

        yield line

class poseAnnotator():

    def __init__(self):

        # self.window = gui.Application.instance.create_window(
            # "Hand Labeling", 1024, 768)

        self.getData()
        self.workingFilePath = None
        
    def getData(self):
        with open("./comm.json", "r") as log_file:
            for line in tail(log_file):
                try:
                    data = json.loads(line)
                except ValueError:
                    # Bad json format, maybe corrupted...
                    continue  # Read next line

                # Do what you want with data:
                # db.execute("INSERT INTO ...", log_data["level"], ...)
                print(data)

def run_test_o3d():

    o3d.visualization.webrtc_server.enable_webrtc()
    # o3dpc =  o3d.io.read_point_cloud("./tmp/cloud_in.ply")
    # o3d.visualization.draw([o3dpc])

    p = poseAnnotator()
    
    gui.Application.instance.initialize()
    gui.Application.instance.run()





if __name__ == "__main__":
    run_test_o3d()