from PIL import Image
import os

image_width = 640
image_height = 640


def fixed_size(filePath, savePath):
    """按照固定尺寸处理图片"""
    im = Image.open(filePath)
    out = im.resize((image_width, image_height))
    out.save(savePath)


def changeSize():
    filePath = 'inputs/Eye3/masks/0'
    destPath = 'inputs/Eye3/masks/0'
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file[-1] == 'g':
                fixed_size(os.path.join(filePath, file), os.path.join(destPath, file))
    print('Done')


if __name__ == '__main__':
    changeSize()