import numpy as np
import cv2
import random
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from aug import augmentationData

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, imgs, masks, aug_data=False, test=False):#txt_path

        # fh = open(txt_path, 'r')
        # imgs = []
        # masks = []
        # for line in fh:
        #     line = line.rstrip()
        #     words = line.split()
        #     imgs.append(words[0])
        #     masks.append(words[1])

        self.args = args
        self.img_paths = imgs
        self.mask_paths = masks
        self.aug_data = aug_data
        self.test = test
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        '''
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.
        nplabel = np.empty((160, 160, 3))
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))


        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        #npedge = npedge.astype("float32")
        npedge = mask_to_edges(nplabel)
        return npimage,nplabel,npedge
        '''

        #读图片（如jpg、png）的代码

        #image = Image.open(img_path)
        #mask = Image.open(mask_path)
        #normMean = [0.706591, 0.580208, 0.534363]#[180.18121, 147.95314, 136.26263]
        #normStd = [0.157334, 0.163555, 0.178653]#[40.12028, 41.706684, 45.556572]

        #image = image.astype('float32') / 255
        #image = (image - normMean) / normStd
        #image = image.astype('float32')
        #mask = mask.astype('float32') / 255

        if self.aug_data:
            img, mask= augmentationData(img_path, mask_path, option=[1,2,3])
        else:
            img, mask= augmentationData(img_path, mask_path,)

        '''
        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
        '''
        # image = color.gray2rgb(img)
        # image = image[:,:,np.newaxis]
        # image = image.transpose((2, 0, 1))
        # mask = mask[:,:,np.newaxis]

        # _edgemap = mask.numpy()
        # _edgemap = mask_to_onehot(_edgemap, 1)
        #
        # _edgemap = onehot_to_binary_edges(_edgemap, 1, 1)
        # _edgemap = _edgemap.transpose(1, 2, 0)
        # tran = transforms.ToTensor()
        # edge = tran(_edgemap)
        return img, mask


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    # input_tensor = input_tensor.clone().detach()
    # 到cpu
    # input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    # input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.transpose(1, 2, 0)   # RGB转BRG
    input_tensor *= 255
    # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask
    mask = mask.squeeze(axis=0)
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


def mask_to_edges(mask):
    _edge = mask
    _edge = onehot_to_binary_edges(_edge)
    return torch.from_numpy(_edge).float()

