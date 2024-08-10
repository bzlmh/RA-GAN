import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.Binarizationdataloder import  TestData
from models.Generator import RA_GAN

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--batchSize', type=int, default=1, help='batch size')
parser.add_argument('--loadSize', type=int, default=256,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='datasets/BD/img', help='path for test data')
parser.add_argument('--pretrained', type=str, default='./checkSave/D2019/Net_2.pth',
                    help='pretrained models')
parser.add_argument('--savePath', type=str, default='./results', help='path for saving results')
parser.add_argument('--only_one', type=bool, default=True, help='only one')
args = parser.parse_args()
cuda = torch.cuda.is_available()
if not os.path.exists('./results'):
    os.makedirs('./results')

if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True
batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
savePath = args.savePath
mask_out_path = os.path.join(savePath, r'mask_out')
if not os.path.exists(savePath):
    os.makedirs(savePath)
    os.makedirs(mask_out_path)
Doc_data = TestData(dataRoot, loadSize, training=False, only_one=args.only_one)
Do_data = DataLoader(Doc_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False,
                      pin_memory=True)
netG =RA_GAN(5)
netG.load_state_dict(torch.load(args.pretrained))

if cuda:
    netG = netG.cuda()
for param in netG.parameters():
    param.requires_grad = False
    netG.eval()
for num, (inputImg, ostu, sobel, img_x, img_y, name) in enumerate(Do_data):  # 只加载6张图片
    if cuda:
        inputImg = inputImg.cuda()
        ostu = ostu.cuda()
        sobel = sobel.cuda()
    mid_out = netG(inputImg, ostu,sobel)
    mask_out = mid_out[0].data.cpu()

    save_image(mask_out, os.path.join(mask_out_path, name[0]), normalize=False)

