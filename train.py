import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_options import TrainOptions
import os
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from model import CreateBranch
from model import CreateDiscriminator
from model import CreatePatchDiscriminator
from utils.timer import Timer
import tensorboardX
# import pickle
import math
import torch.nn as nn
from PIL import Image
from PIL import ImageFile
from pathlib import Path

# TODO
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO
# CHANGE START
# copy from data/__init__.py
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def recover_ori_img(load):
    load = load.cpu().numpy()
    load = load.transpose(1, 2, 0)
    load += IMG_MEAN
    load = load.astype(np.uint8)
    load = Image.fromarray(load.astype(np.uint8)).convert('RGB')
    load = np.array(load).transpose((2, 0, 1))
    return load


def colorize_mask(mask):
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def plot_seg(seg):
    """
    seg: 4-dim tensor with size: (batch size, num of classes, height, width)
    output: 3-dim color segmentation result map, currently confirmed work for batch size = 1
    """
    output = nn.functional.softmax(seg, dim=1)
    # the below line drop the first dimension correspond to batch size
    output = nn.functional.upsample(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
    # output = nn.functional.interpolate(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data.numpy()
    output = output.transpose(1, 2, 0)
    output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    output_col = colorize_mask(output_nomask)
    output_col = output_col.convert('RGB')
    return np.array(output_col).transpose((2, 0, 1))


def plot_src_lbl(lbl):
    lbl = lbl.cpu().numpy()
    lbl = lbl.astype(np.uint8)
    output_col = colorize_mask(lbl)
    output_col = output_col.convert('RGB')
    return np.array(output_col).transpose((2, 0, 1))


# def entropy_seg(seg):
#     """
#     seg: 4-dim tensor with size: (batch size, num of classes, height, width)
#     return ent_seg: (batch size, height, width)
#     """
#     seg = F.softmax(seg, dim=1)
#     num_class = seg.size()[1]
#     ent_seg = -torch.sum(torch.mul(seg, torch.log(seg + 1e-30)), dim=1, keepdim=True) / math.log(num_class)
#     return ent_seg

# CHANGE END


def main():
    
    opt = TrainOptions()
    args = opt.initialize()
    
    _t = {'iter time': Timer()}

    opt.print_options(args)
    
    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)

    # TODO: check model code
    # TODO: if-else to determine the needed of instantiation
    model, optimizer = CreateModel(args)
    model_H, optimizer_H = CreateBranch(args)
    model_D, optimizer_D = CreateDiscriminator(args)
    model_Dp, optimizer_Dp = CreatePatchDiscriminator(args)

    assert (args.lambda_adv_fake == 0 and args.lambda_adv_real == 0) or (
            args.lambda_adv_fake != 0 and args.lambda_adv_real != 0), "D adversarial training weights inconsistent!"
    assert (args.lambda_adv_fake_p == 0 and args.lambda_adv_real_p == 0) or (
                args.lambda_adv_fake_p != 0 and args.lambda_adv_real_p != 0), "Dp adversarial training weights inconsistent!"
    
    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    # # TODO: to comment out?
    # bce_loss = torch.nn.BCEWithLogitsLoss()
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    model_D.train()
    model_D.cuda()

    snapshot_dir = os.path.join(args.checkpoint_dir, args.create_snapshot_folder)

    logdir = os.path.join(args.checkpoint_dir, "logs", args.source + '_to_' + args.target, args.tb_create_exp_folder)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    train_writer = tensorboardX.SummaryWriter(logdir)

    # for tensorboard
    # TODO: whether the elements in the list below exist in all possible argument settings
    loss = ['loss_seg_src', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real',
            'loss_patch_src', 'loss_Dp_trg_fake', 'loss_Dp_src_real', 'loss_Dp_trg_real']
    color_map = ['src_img_vis', 'src_lbl_vis', 'src_color_map', 'trg_img_vis', 'trg_color_map']

    _t['iter time'].tic()

    print("start_iter:", start_iter, "args.num_steps:", args.num_steps)
    for i in range(start_iter, args.num_steps):

        model.adjust_learning_rate(args, optimizer, i) # TODO: study
        model_H.adjust_learning_rate(args, optimizer_H, i)
        model_D.adjust_learning_rate(args, optimizer_D, i)
        model_Dp.adjust_learning_rate(args, optimizer_Dp, i)

        optimizer.zero_grad()
        optimizer_H.zero_grad()
        optimizer_D.zero_grad()
        optimizer_Dp.zero_grad()

        ### model M forward & backward

        ## source domain segmentation loss

        src_img, src_lbl, _, _ = sourceloader_iter.next()
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
        src_seg_score = model(src_img, lbl=src_lbl)

        # for tensorboard visualization
        if (i + 1) % args.add_tb_image_every == 0:
            src_img_vis = recover_ori_img(src_img[0])
            src_lbl_vis = plot_src_lbl(src_lbl[0])
            src_color_map = plot_seg(src_seg_score)

        loss_seg_src = model.loss
        loss_seg_src.backward()


        # source domain patch representation and loss
        if args.lambda_p or args.lambda_adv_fake_p:

            src_patch_score = model_H(src_seg_score, lbl=src_lbl) ########## TODO: patch GT, need another loader above

            loss_patch_src = model_H.loss * args.lambda_p
            loss_patch_src.backward()


        ## target domain data

        trg_img, _, name = targetloader_iter.next()
        trg_img = Variable(trg_img).cuda()
        trg_seg_score = model(trg_img, lbl=None)

        # for tensorboard visualization
        if (i + 1) % args.add_tb_image_every == 0:
            trg_img_vis = recover_ori_img(trg_img[0])
            trg_color_map = plot_seg(trg_seg_score)


        if args.lambda_adv_fake:
            # loss: fooling D

            for param in model_D.parameters():
                param.requires_grad = False

            outD_trg = model_D(F.softmax(trg_seg_score, dim=1), 0)

            loss_D_trg_fake = model_D.loss * args.lambda_adv_fake
            loss_D_trg_fake.backward()


        if args.lambda_adv_fake_p:
            # target domain patch representation
            trg_patch_score = model_H(trg_seg_score, lbl=None)  # TODO: w/o self-training by now

            # loss: fooling Dp

            for param in model_Dp.parameters():
                param.requires_grad = False

            outD_trg = model_Dp(F.softmax(trg_patch_score, dim=1), 0)

            loss_Dp_trg_fake = model_Dp.loss * args.lambda_adv_fake_p
            loss_Dp_trg_fake.backward()


        ### discriminator forward & backward

        if args.lambda_adv_real:

            for param in model_D.parameters():
                param.requires_grad = True

            src_seg_score, trg_seg_score = src_seg_score.detach(), trg_seg_score.detach()

            outD_src = model_D(F.softmax(src_seg_score, dim=1), 0)
            loss_D_src_real = (model_D.loss / 2) * args.lambda_adv_real
            loss_D_src_real.backward()

            outD_trg = model_D(F.softmax(trg_seg_score, dim=1), 1)
            loss_D_trg_real = (model_D.loss / 2) * args.lambda_adv_real
            loss_D_trg_real.backward()


        if args.lambda_adv_real_p:

            for param in model_Dp.parameters():
                param.requires_grad = True

            src_patch_score, trg_patch_score = src_patch_score.detach(), trg_patch_score.detach()

            outDp_src = model_Dp(F.softmax(src_patch_score, dim=1), 0)
            loss_Dp_src_real = (model_Dp.loss / 2) * args.lambda_adv_real_p
            loss_Dp_src_real.backward()

            outDp_trg = model_Dp(F.softmax(trg_patch_score, dim=1), 1)
            loss_Dp_trg_real = (model_Dp.loss / 2) * args.lambda_adv_real_p
            loss_Dp_trg_real.backward()


        ### update all networks

        optimizer.step()

        if args.lambda_p:
            optimizer_H.step()

        if args.lambda_adv_fake:
            optimizer_D.step()

        if args.lambda_adv_fake_p:
            optimizer_Dp.step()


        ### saving and printing
        
        for m in loss:
            train_writer.add_scalar(m, eval(m), i + 1)

        if (i + 1) % args.add_tb_image_every == 0:
            for m in color_map:
                train_writer.add_image(m, eval(m), i + 1)
            
        if (i + 1) % args.save_checkpoint_every == 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(snapshot_dir, str(i + 1) + '.pth'))
            torch.save(model_H.state_dict(), os.path.join(snapshot_dir, str(i + 1) + '_H.pth'))
            torch.save(model_D.state_dict(), os.path.join(snapshot_dir, str(i + 1) + '_D.pth'))
            torch.save(model_Dp.state_dict(), os.path.join(snapshot_dir, str(i + 1) + '_Dp.pth'))
            
        if (i + 1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))

        if i + 1 == args.num_steps_stop:
        # if i + 1 == args.num_steps:
            print('finish training')
            break
        _t['iter time'].tic()


if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
    main()
        