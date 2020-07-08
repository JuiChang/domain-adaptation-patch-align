import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))).convert('RGB')
        # # CHANGE START
        # ori_image = np.array(image)
        # # print("csp, ori_image.shape:", ori_image.shape)
        # ori_image = ori_image.transpose((2, 0, 1))
        # # print("csp, ori_image.shape:", ori_image.shape)
        # # image = Image.fromarray(ori_image)
        # # image.save('tmp_output/image_c_ds.png')
        # # CHANGE END
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        # return ori_image, image.copy(), np.array(size), name
        return image.copy(), np.array(size), name
