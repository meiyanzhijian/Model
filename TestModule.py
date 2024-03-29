from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
import os
from utility import to_variable
##################加入ssim包
import cv2
import pssim.pytorch_ssim as pytorch_ssim
from torch.autograd import Variable
import numpy as np

def ssim_(img1,img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0
    img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
    img2 = Variable( img2, requires_grad = False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value
###########################

class Middlebury_eval:
    def __init__(self, input_dir='./evaluation'):
        self.im_list = ['Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Mequon', 'Schefflera', 'Teddy', 'Urban']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame11.png')).unsqueeze(0)))

    def Test(self, model, output_dir='./evaluation/output', output_name='frame10i11.png'):
        with torch.no_grad():
            model.eval()
            for idx in range(len(self.im_list)):
                if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                    os.makedirs(output_dir + '/' + self.im_list[idx])
                frame_out = model(self.input0_list[idx], self.input1_list[idx])
                imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))
####添加ssim
        self.ssim_list=[]
        for item_ssim in self.im_list:
            self.ssim_list.append(cv2.imread(gt_dir + '/' + item_ssim + '/frame10i11.png'))
#############################
    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
##加入with torch.no_grad()
        with torch.no_grad():
#######################
            model.eval()
##加入av_ssim

            av_ssim=0
#############################
            av_psnr = 0
            if logfile is not None:
                logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
#########修改
            print('psnr:')
            if logfile is not None:
                logfile.write('psnr:' + '\n')
##################
            for idx in range(len(self.im_list)):
                if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                    os.makedirs(output_dir + '/' + self.im_list[idx])
                frame_out = model(self.input0_list[idx], self.input1_list[idx])
                gt = self.gt_list[idx]
                psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())  #求psnr值
                av_psnr += psnr
                imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
                msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
                print(msg, end='')
                if logfile is not None:
                    logfile.write(msg)
            av_psnr /= len(self.im_list)
            msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
#########修改
            print('ssim:')
            if logfile is not None:
                logfile.write('ssim:' + '\n')
##################
###加入ssim
            for idx in range(len(self.im_list)):
                frame1 = cv2.imread(output_dir + '/' + self.im_list[idx] + '/' + output_name)  #读取中间帧
                gt = self.ssim_list[idx]    #真实中间帧
                ssim = ssim_(frame1, gt)    #求ssim值

                av_ssim += ssim
                msg_ssim = '{:<15s}{:<20.16f}'.format(self.im_list[idx]+': ', ssim) + '\n'
                print(msg_ssim, end='')
                if logfile is not None:
                    logfile.write(msg_ssim)
            av_ssim /= len(self.im_list)
            msg_ssim = '{:<15s}{:<20.16f}'.format('Average: ', av_ssim) + '\n'
            print(msg_ssim, end='')
            if logfile is not None:
                logfile.write(msg_ssim)
##################################
###添加返回
        return str(av_psnr),str(av_ssim)
##################################

class Davis:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['bike-trial', 'boxing', 'burnout', 'choreography', 'demolition', 'dive-in', 'dolphins', 'e-bike', 'grass-chopper', 'hurdles', 'inflatable', 'juggle', 'kart-turn', 'kids-turning', 'lions', 'mbike-santa', 'monkeys', 'ocean-birds', 'pole-vault', 'running', 'selfie', 'skydive', 'speed-skating', 'swing-boy', 'tackle', 'turtle', 'varanus-tree', 'vietnam', 'wings-turn']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))  #将图片转换为张量
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))
####添加ssim
        self.ssim_list = []
        for item_ssim in self.im_list:
            self.ssim_list.append(cv2.imread(gt_dir + '/' + item_ssim + '/frame10i11.png'))  #图像读取
#############################

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
##加入with torch.no_grad()
        with torch.no_grad():   ###在测试函数前加了装饰器，解决了cuda out of memory（内存不足）
#######################
            model.eval()
            ##加入av_ssim
            av_ssim = 0
#############################
            av_psnr = 0
            if logfile is not None:
                logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
#########修改
            print('psnr:')
            if logfile is not None:
                logfile.write('psnr:' + '\n')
##################
            for idx in range(len(self.im_list)):  #len(self.im_list) = 29
                if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                    os.makedirs(output_dir + '/' + self.im_list[idx])
                frame_out = model(self.input0_list[idx], self.input1_list[idx])  #输出的帧
                gt = self.gt_list[idx]  #真实帧
                psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())  #求出psnr
                av_psnr += psnr
                imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))  #输出图像到文件中 需要写入的文件名，要保存的图像，表示为特定格式保存的参数编码
                msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
                print(msg, end='')
                if logfile is not None:
                    logfile.write(msg)
            av_psnr /= len(self.im_list)
            msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
#########修改
            print('ssim:')
            if logfile is not None:
                logfile.write('ssim:' + '\n')
##################
###加入ssim
            for idx in range(len(self.im_list)):
                frame1 = cv2.imread(output_dir + '/' + self.im_list[idx] + '/' + output_name)  #读取中间帧
                gt = self.ssim_list[idx]    #真实中间帧
                ssim = ssim_(frame1, gt)    #求ssim值

                av_ssim += ssim
                msg_ssim = '{:<15s}{:<20.16f}'.format(self.im_list[idx]+': ', ssim) + '\n'
                print(msg_ssim, end='')
                if logfile is not None:
                    logfile.write(msg_ssim)
            av_ssim /= len(self.im_list)
            msg_ssim = '{:<15s}{:<20.16f}'.format('Average: ', av_ssim) + '\n'
            print(msg_ssim, end='')
            if logfile is not None:
                logfile.write(msg_ssim)
##################################
###添加返回
        return str(av_psnr),str(av_ssim)
##################################

class ucf:
    def __init__(self, input_dir):
        self.im_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame0.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame2.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame1.png')).unsqueeze(0)))
####添加ssim
        self.ssim_list = []
        for item_ssim in self.im_list:
            self.ssim_list.append(cv2.imread(input_dir + '/' + item_ssim +'/frame1.png'))

#############################
    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
##加入with torch.no_grad()
        with torch.no_grad():
#######################
            model.eval()
##加入av_ssim
            av_ssim = 0
#############################
            av_psnr = 0
            if logfile is not None:
                logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
#########修改
            print('psnr:')
            if logfile is not None:
                logfile.write('psnr:' + '\n')
##################
            for idx in range(len(self.im_list)):
                if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                    os.makedirs(output_dir + '/' + self.im_list[idx])
                frame_out = model(self.input0_list[idx], self.input1_list[idx])
                gt = self.gt_list[idx]
                psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
                av_psnr += psnr
                imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
                msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
                print(msg, end='')
                if logfile is not None:
                    logfile.write(msg)
            av_psnr /= len(self.im_list)
            msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
 #########修改
            print('ssim:')
            if logfile is not None:
                logfile.write('ssim:' + '\n')
 ##################
 ###加入ssim
            for idx in range(len(self.im_list)):
                frame1 = cv2.imread(output_dir + '/' + self.im_list[idx] + '/' + output_name)  # 读取中间帧
                gt = self.ssim_list[idx]  # 真实中间帧
                ssim = ssim_(frame1, gt)  # 求ssim值

                av_ssim += ssim
                msg_ssim = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', ssim) + '\n'
                print(msg_ssim, end='')
                if logfile is not None:
                    logfile.write(msg_ssim)
            av_ssim /= len(self.im_list)
            msg_ssim = '{:<15s}{:<20.16f}'.format('Average: ', av_ssim) + '\n'
            print(msg_ssim, end='')
            if logfile is not None:
                logfile.write(msg_ssim)
##################################
###添加返回
        return str(av_psnr), str(av_ssim)
##################################

class ucf_our:
    def __init__(self):

        self.transform = transforms.Compose([transforms.ToTensor()])

    def Test(self, path, model, to_path, logfile=None,output_name='output.png'):
        with torch.no_grad():
            # 将输入帧和输出帧的路径都保存到list
            frame0 = []  # 存储所有frame0.png的路径
            frame2 = []  # 存储所有frame2.png的路径
            frame1_dir = []  # 存储所有frame1.png所在文件夹的路径
            frame1 = []  # 存储所有frame1.png的路径，image.open
            frame1_ssim = []  # 存储所有frame1.png的路径,cv2.read
            frameout = []  # 输出结果的路径

            # 判断是不是文件夹  path D:\UCF101  to_path D:\a
            a = os.path.split(path)
            # print(a[1])
            # print(type(a))
            is_dir = os.path.isdir(path)
            # 如果是文件夹，则在指定目录下新建一个同名文件夹
            if is_dir == True:
                creat = os.path.join(to_path, a[1])  # D:\a\UCF101
                # print(creat)
                if not os.path.exists(creat):
                    os.mkdir(creat)
                # 得到文件夹下的第一层子目录名称dir1 如：ApplyEyeMakeup
                dirs1 = os.listdir(path)
                for dir1 in dirs1:
                    # 输出文件夹下的第一层子目录名字
                    insert1 = os.path.join(path, dir1)  # D:\UCF101\v_ApplyEyeMakeup_g11_c02
                    # print(insert1,'insert1')
                    if os.path.isdir(insert1) == True:
                        creat1 = os.path.join(creat, dir1)  # D:\a\UCF101\v_ApplyEyeMakeup_g11_c02
                        # print(creat1,'creat1')
                        # 子文件夹下的文件目录
                        if not os.path.exists(creat1):
                            os.mkdir(creat1)
                        dirs2 = os.listdir(insert1)
                        for dir2 in dirs2:
                            insert2 = os.path.join(insert1, dir2)  # D:\UCF101\v_ApplyEyeMakeup_g01_c01\1
                            # print(insert2)
                            creat2 = os.path.join(creat1, dir2)  # D:\a\UCF101\v_ApplyEyeMakeup_g01_c01\1
                            # print(creat2)
                            # 将frame0,frame1,frame2分别存入一个list
                            if not os.path.exists(creat2):
                                os.mkdir(creat2)
                            dirframes = os.listdir(insert2)
                            # list0 = []
                            # list2 = []
                            for dirframe in dirframes:
                                if dirframe == 'frame0.png':
                                    framelist0 = os.path.join(insert2, dirframe)  # 将所有frame0.png路径写入一个list
                                    # print(framelist0,'000')
                                    frame0.append(framelist0)
                                    # list0.append(framelist0)
                                elif dirframe == 'frame2.png':
                                    framelist2 = os.path.join(insert2, dirframe)  # 将所有frame2.png路径写入一个list
                                    # print(framelist2, '222')
                                    frame2.append(framelist2)
                                    # list2.append(framelist2)
                                else:
                                    framelist1 = os.path.join(insert2,
                                                              dirframe)  # gt路径：D:\UCF101\v_ApplyEyeMakeup_g01_c01\1\frame1.png
                                    # print(framelist1)
                                    frame1_dir.append(insert2)
                                    frame1.append(framelist1)
                                    frame1_ssim.append(cv2.imread(framelist1))
                                    dirframe=output_name
                                    to_framelist1 = os.path.join(creat2,
                                                                 dirframe)  # 输出路径：D:\a\UCF101\v_ApplyEyeMakeup_g01_c01\1\frame1.png
                                    # print(to_framelist1)
                                    frameout.append(to_framelist1)
                            # # print(list0[0],'000')
                            # # print(list2[0],'222')
                            # print(list0,'aaaaa')

        # 开始测试

            model.eval()
            ##加入av_ssim
            av_ssim = 0
            av_psnr = 0
            if logfile is not None:
                logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
            # 求psnr
            print('psnr:')
            if logfile is not None:
                logfile.write('psnr:' + '\n')
            for item in range(len(frame0)):
                frame_out = model(to_variable(self.transform(Image.open(frame0[item])).unsqueeze(0)), to_variable(self.transform(Image.open(frame2[item])).unsqueeze(0)))
                gt=to_variable(self.transform(Image.open(frame1[item])).unsqueeze(0))
                psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
                av_psnr += psnr
                imwrite(frame_out, frameout[item], range=(0, 1))
                msg = '{:<15s}{:<20.16f}'.format(frame1_dir[item] + ': ', psnr) + '\n'
                print(msg, end='')
                if logfile is not None:
                    logfile.write(msg)
            av_psnr /= len(frame0)
            msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
            print('ssim:')
            if logfile is not None:
                logfile.write('ssim:' + '\n')
            ####求ssim
            for item in range(len(frame0)):
                frame1_out = cv2.imread(frameout[item])
                gt = frame1_ssim[item]
                ssim = ssim_(frame1_out, gt)  # 求ssim值

                av_ssim += ssim
                msg_ssim = '{:<15s}{:<20.16f}'.format(frame1_dir[item] + ': ', ssim) + '\n'
                print(msg_ssim, end='')
                if logfile is not None:
                    logfile.write(msg_ssim)

            av_ssim /= len(frame0)
            msg_ssim = '{:<15s}{:<20.16f}'.format('Average: ', av_ssim) + '\n'
            print(msg_ssim, end='')
            if logfile is not None:
                logfile.write(msg_ssim)

        return str(av_psnr), str(av_ssim)
