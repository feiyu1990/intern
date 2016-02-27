__author__ = 'wangyufei'

import os
import sys
root = '/Users/wangyufei/Documents/Study/intern_adobe/demo/'
import PIL
from PIL import Image

basewidth = 300

def resize_images(folder):
    files = [root + folder +'/'+ f for f in os.listdir(root + folder) if f.endswith('jpg') or f.endswith('JPG') or f.endswith('png') or f.endswith('JPG')]
    for file in files:
        img = Image.open(file)
        if img.size[0] > img.size[1]:
            h = basewidth;w = basewidth*img.size[0]/img.size[1]
        else:
            w = basewidth;h = basewidth*img.size[1]/img.size[0]
        img = img.resize((w,h), PIL.Image.ANTIALIAS)
        img.save(file)

if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    folder = args[1]
    resize_images(folder)

