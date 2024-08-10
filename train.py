import os
import argparse
import torch
import cv2
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from data.Binarizationdataloder import DocData
from loss.loss import Loss_Doc
from models.Generator  import RA_GAN
from models.Discriminator import Discriminator

import random
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0, help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='./checkSave/D2019', help='path for saving models')
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--loadSize', type=int, default=256, help='image loading size')
parser.add_argument('--dataRoot', type=str, default=r'datasets/BD/img')
parser.add_argument('--pretrained', type=str, default=r'', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=60, help='epochs')
parser.add_argument('--only_one', type=bool, default=True, help='only one')
parser.add_argument('--testRoot', type=str, default=r'')
parser.add_argument('--testSave', type=str, default=r'./results/D2019')
parser.add_argument('--ganLoss', type=bool, default=True)
args = parser.parse_args()


def cv_imwrite(filename, src):
    cv2.imencode('.tiff', src)[1].tofile(filename)


def init_result_Dir():
    work_dir = os.path.join(os.getcwd(), 'Training')
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    path = os.path.join(work_dir, str(max_model))
    os.mkdir(path)
    return path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1028)

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.enable = True
    cudnn.benchmark = True

batchSize = args.batchSize
dataRoot = args.dataRoot
loadSize = (args.loadSize, args.loadSize)
num_epochs = args.num_epochs
if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

Doc_data = DocData(dataRoot, loadSize, training=True, only_one=args.only_one)
Do_data = DataLoader(Doc_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False,
                     pin_memory=True)
binarization = RA_GAN(5)
discriminator = Discriminator(4)
loss_doc = Loss_Doc(binarization)
B_optimizer = optim.Adam(binarization.parameters(), lr=0.00003)
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.000001)
if args.pretrained != '':
    print('loaded ')
    binarization.load_state_dict(torch.load(args.pretrained))
if cuda:
    generator = binarization.to('cuda')
if cuda:
    loss_doc = loss_doc.to('cuda')
if cuda:
    discriminator = discriminator.to('cuda')
print('Datasets:', len(Doc_data))
path1 = init_result_Dir()
for epoch in range(1, num_epochs + 1):
    binarization.train()
    for num, (inputImg, ostu, sobel, gt, gt_Sobel, gray, img_x, img_y, name) in enumerate(Do_data):
        if cuda:
            inputImg = inputImg.cuda()
            ostu = ostu.cuda()
            sobel = sobel.cuda()
            gt = gt.cuda()
            gray = gray.cuda()
        D_optimizer.zero_grad()

        bin_out = binarization(inputImg, ostu, sobel)
        D_real_c = discriminator(gt, inputImg)
        D_real_c = D_real_c.mean().sum() * -1
        D_fake_c = discriminator(bin_out.detach(), inputImg)
        D_fake_c = D_fake_c.mean().sum() * 1
        D_loss = torch.mean(F.relu(1. + D_real_c)) + torch.mean(F.relu(1. + D_fake_c))

        D_loss.backward()
        D_optimizer.step()

        binarization.zero_grad()
        fin_l1_loss, fin_BCEloss, fin_dice_loss, fin_jaccard_loss = loss_doc(inputImg, bin_out, gt)
        fake_preds = discriminator(bin_out, inputImg)
        adversarial_loss = torch.nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))
        G_loss = 10 * fin_l1_loss + fin_BCEloss + fin_dice_loss + fin_jaccard_loss + 0.1 * adversarial_loss
        G_loss.backward()
        B_optimizer.step()
        print(
            '[{}/{}] Epoch {}: G Loss: {:.4f}, D Loss: {:.4f}, Adversarial Loss: {:.4f}, D_real_loss: {:.4f}, D_fake_loss: {:.4f}'.format(
                num, len(Doc_data) // batchSize, epoch, G_loss.item(), D_loss.item(), adversarial_loss.item(), D_real_c.item(), D_fake_c.item())
        )

        if num % 200 == 0:
            outt = torch.cat((gray, gt, sobel, bin_out), 3)  # Use bin_out instead of mid_out
            outt = outt.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
            outt = outt[0].astype(np.float32)
            cv_imwrite(
                os.path.join(path1, str(epoch) + '_' + str(num) + '_' + img_x[0] + '_' + img_y[0] + '_' + name[0]),
                outt)
    if epoch % 1 == 0:
        torch.save(binarization.state_dict(), args.modelsSavePath + '/Net_{}.pth'.format(epoch))
