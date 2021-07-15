#source:https://www.codegrepper.com/code-examples/python/merge+all+mp4+video+files+into+one+file+python
from moviepy.editor import *
import os
from natsort import natsorted

L =[]

for root, dirs, files in os.walk("C:/Users/Lakshmi Prasanna.B/website/blendervideo1"):

    #files.sort()
    files = natsorted(files)
    for file in files:
        if os.path.splitext(file)[1] == '.mp4':
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("output.mp4", fps=24, remove_temp=False)
