from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Resize
from utils.utils import split_ext, correct_size
import torch

def get_gt_sobel(gt):
    gt = np.asarray(gt).astype(np.uint8)
    x = cv2.Sobel(gt, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gt, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Image.fromarray(dst)

def get_ostu(img):
    img = np.asarray(img).astype(np.uint8)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th2)

def get_file_name(filepath):
    name = os.listdir(filepath)
    name = [os.path.join(filepath, i) for i in name if os.path.splitext(i)[1] in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.tiff', '.TIFF', '.tif', '.TIF']]
    return len(name), name

def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def color_jitter(imgs):
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    saturation = np.random.uniform(0.8, 1.2)
    hue = np.random.uniform(-0.1, 0.1)

    img_hsv = cv2.cvtColor(imgs, cv2.COLOR_RGB2HSV)
    img_hsv = img_hsv.astype(np.float32)

    img_hsv[:, :, 2] *= brightness
    img_hsv[:, :, 1] *= contrast
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    img_hsv[:, :, 0] *= saturation
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 255)
    img_hsv[:, :, 0] += hue * 255
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 255)

    imgs = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    imgs = Image.fromarray(imgs)

    return imgs


def random_rotate(imgs, color, angle):
    img = np.array(imgs).astype(np.uint8)
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), borderValue=color)
    img = Image.fromarray(img_rotation)
    return img

def ImageTransform(loadSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        ToTensor(),
    ])

def ImageTransform_fullsize():
    return Compose(
        [ToTensor()]
    )

class DocData(Dataset):
    def __init__(self, imgRoot, loadsize, training=True, only_one=False):
        super(DocData, self).__init__()
        self.number, self.ImgFiles = get_file_name(imgRoot)
        self.loadsize = loadsize
        self.training = training
        self.Imgtrans = ImageTransform(loadsize)
        self.Imgtrans_fullsize = ImageTransform_fullsize()
        self.dataOriPath = r'./datasets'
        self.only_one = only_one

    def __getitem__(self, index):
        dataRoot, (dataName, dataExt) = split_ext(self.ImgFiles[index])
        dataName = dataName.split('_')
        if not self.only_one:
            if dataName[0] != 'H':
                if dataName[0] == 'DIBCO':
                    datasetsName = '_'.join(dataName[0:2])
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[2:-2])
                else:
                    datasetsName = dataName[0]
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[1:-2])
            else:
                datasetsName = '_'.join(dataName[0:3])
                img_x = dataName[-2]
                img_y = dataName[-1]
                imgName = '_'.join(dataName[3:-2])
        else:
            datasetsName = os.path.basename(os.path.dirname(dataRoot))
            img_x = dataName[-2]
            img_y = dataName[-1]
            imgName = '_'.join(dataName[0:-2])

        imgOriRoot = os.path.join(self.dataOriPath, os.path.join(datasetsName, 'img'))
        imgName_with_coordinates = f"{imgName}_{img_x}_{img_y}"
        imgRoot = os.path.join(imgOriRoot, imgName_with_coordinates + dataExt)

        img = Image.open(self.ImgFiles[index]).convert('RGB')
        gt = Image.open(self.ImgFiles[index].replace('img', 'gt')).convert('L')
        ostu = Image.open(self.ImgFiles[index].replace('img', 'ostu')).convert('L')
        sobel = Image.open(self.ImgFiles[index].replace('img', 'sobel')).convert('L')
        gray = img.convert('L')
        gt_Sobel = get_gt_sobel(gt).convert('L')
        if self.training:
            if random.random() < 0.3:
                max_angle = 10
                angle = random.random() * 2 * max_angle - max_angle
                img = random_rotate(img, 0, angle)
                ostu = random_rotate(ostu, (255, 255, 255), angle)
                gt = random_rotate(gt, (255, 255, 255), angle)
            if random.random() < 0.5:
                img = color_jitter(np.array(img))
        inputImg = self.Imgtrans(img)
        ostu = self.Imgtrans(ostu)
        sobel = self.Imgtrans(sobel)
        gt = self.Imgtrans(gt)
        gray = self.Imgtrans(gray)
        gt_Sobel = self.Imgtrans(gt_Sobel)
        path, name = os.path.split(self.ImgFiles[index])
        return inputImg, ostu, sobel, gt, gt_Sobel, gray, img_x, img_y, name

    def __len__(self):
        return len(self.ImgFiles)


class TestData(Dataset):
    def __init__(self, imgRoot, loadsize, training=True, only_one=False):
        super(TestData, self).__init__()
        self.number, self.ImgFiles = get_file_name(imgRoot)
        self.loadsize = loadsize
        self.training = training
        self.Imgtrans = ImageTransform(loadsize)
        self.Imgtrans_fullsize = ImageTransform_fullsize()
        self.dataOriPath = r'./datasets'
        self.only_one = only_one
    def __getitem__(self, index):
        dataRoot, (dataName, dataExt) = split_ext(self.ImgFiles[index])
        dataName = dataName.split('_')
        if not self.only_one:
            if dataName[0] != 'H':
                if dataName[0] == 'DIBCO':
                    datasetsName = '_'.join(dataName[0:2])
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[2:-2])
                else:
                    datasetsName = dataName[0]
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[1:-2])
            else:
                datasetsName = '_'.join(dataName[0:3])
                img_x = dataName[-2]
                img_y = dataName[-1]
                imgName = '_'.join(dataName[3:-2])
        else:
            datasetsName = os.path.basename(os.path.dirname(dataRoot))
            img_x = dataName[-2]
            img_y = dataName[-1]
            imgName = '_'.join(dataName[0:-2])

        imgOriRoot = os.path.join(self.dataOriPath, os.path.join(datasetsName, 'img'))
        imgName_with_coordinates = f"{imgName}_{img_x}_{img_y}"
        imgRoot = os.path.join(imgOriRoot, imgName_with_coordinates + dataExt)
        img = Image.open(self.ImgFiles[index]).convert('RGB')
        ostu = Image.open(self.ImgFiles[index].replace('img', 'ostu')).convert('L')
        sobel = Image.open(self.ImgFiles[index].replace('img', 'sobel')).convert('L')
        # prewitt = Image.open(self.ImgFiles[index].replace('img', 'prewitt')).convert('L')

        if self.training and random.random() < 0.3:
            max_angle = 10
            angle = random.random() * 2 * max_angle - max_angle
            img = random_rotate(img, 0, angle)
            ostu = random_rotate(ostu, (255, 255, 255), angle)
            sobel = random_rotate(sobel, 0, angle)

        inputImg = self.Imgtrans(img)
        ostu = self.Imgtrans(ostu)
        sobel = self.Imgtrans(sobel)
        # prewitt=self.Imgtrans(prewitt)
        path, name = os.path.split(self.ImgFiles[index])
        return inputImg, ostu, sobel, img_x, img_y, name

    def __len__(self):
        return self.number