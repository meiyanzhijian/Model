import argparse
import torch
import models
import os
import TestModule

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='F:/lwj/Adacof-R2Unet/output_adacof_train/checkpoint/model_epoch050.pth')
parser.add_argument('--config', type=str, default='F:/lwj/Adacof-R2Unet/output_adacof_train/config.txt')
parser.add_argument('--out_dir', type=str, default='F:/lwj/Ablation Experiment/output_adacof_test')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)




def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(args.gpu_id)

    if args.config is not None:
        config_file = open(args.config, 'r')   ###打开文件，读
        while True:
            line = config_file.readline()   ###每次读取一行内容
            if not line:
                break
            if line.find(':') == 0:
                continue
            else:
                tmp_list = line.split(': ')
                if tmp_list[0] == 'kernel_size':
                    args.kernel_size = int(tmp_list[1])
                if tmp_list[0] == 'dilation':
                    args.dilation = int(tmp_list[1])
        config_file.close()

    model = models.Model(args)

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])
    current_epoch = checkpoint['epoch']

    print('Test: Middlebury_eval')
    test_dir = args.out_dir + '/middlebury_eval'
    test_db = TestModule.Middlebury_eval('F:/lwj/test_input/middlebury_eval')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir)

    print('Test: Middlebury_others')
    test_dir = args.out_dir + '/middlebury_others'
    test_db = TestModule.Middlebury_other('F:/lwj/test_input/middlebury_others/input', 'F:/lwj/test_input/middlebury_others/gt')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, current_epoch, output_name='adacof-R2Unet-frame1.png')

    print('Test: DAVIS')
    test_dir = args.out_dir + '/davis'
    test_db = TestModule.Davis('F:/lwj/test_input/davis/input', 'F:/lwj/test_input/davis/gt')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, output_name='adacof-R2Unet-frame1.png')

    print('Test: UCF101')
    test_dir = args.out_dir + '/ucf101'
    test_db = TestModule.ucf('F:/lwj/test_input/ucf101')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, output_name='adacof-R2Unet-frame1.png')

    print('Test:Ourdataset UCF101')
    test_dir = args.out_dir
    test_db = TestModule.ucf_our()
    if not os.path.exists(test_dir):
       os.makedirs(test_dir)
    test_db.Test('F:/lwj/test_input/our_ucf101', model, test_dir,output_name='adacof-R2Unet-frame1.png')

if __name__ == "__main__":
    main()
