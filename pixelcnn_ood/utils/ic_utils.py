import os
import cv2
import numpy as np

def find_png_size(x):
    x = x.astype('int32')
    tmp = []
    for i in range (x.shape[0]):
        tmp.append(len(cv2.imencode('.png', np.squeeze(x[i,:,:,:]), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])[1])*8)
    return tmp

def find_jpeg_size(x):
    x = x.astype('int32')
    tmp = []
    for i in range (x.shape[0]):
        cv2.imwrite('./temp.png', np.squeeze(x[i,:,:,:]))
        os.system("magick ./temp.png -quality 0 ./temp.jp2 > out.log 2>&1")
        tmp.append(os.path.getsize('./temp.jp2')*8)
        os.remove('./temp.jp2')
    return tmp

def find_flif_size(x):
    x = x.astype('int32')
    tmp = []
    for i in range (x.shape[0]):
        cv2.imwrite('./temp.png', np.squeeze(x[i,:,:,:]))
        os.system("flif -e -o ./temp.png ./temp.flif > out.log 2>&1")
        tmp.append(os.path.getsize('./temp.flif')*8)
        os.remove('./temp.flif')
    return tmp