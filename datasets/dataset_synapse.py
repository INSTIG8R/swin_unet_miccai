import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from bert_embedding import BertEmbedding


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label.long(), 'text': text}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, row_text,transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.rowtext = row_text
        self.bert_embedding = BertEmbedding(max_seq_length=75)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            text = self.rowtext[slice_name+'.png']
            text = text.split('\n')
            text_token = self.bert_embedding(text)
            text = np.array(text_token[0][1])
            if text.shape[0] > 40:
                text = text[:40, :]
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            text = []
            for j in range(len(image)):
                text_tmp = self.rowtext[f"{vol_name}_slice_{j}.png"]
                text_tmp = text_tmp.split('\n')
                text_tmp_token = self.bert_embedding(text_tmp)
                text_tmp = np.array(text_tmp_token[0][1])
                if text_tmp.shape[0] > 40:
                    text_tmp = text_tmp[:40, :]
                text.append(text_tmp)

        # text = self.rowtext[slice_name+'.png']
        # text = text.split('\n')
        # text_token = self.bert_embedding(text)
        # text = np.array(text_token[0][1])
        # if text.shape[0] > 10:
        #     text = text[:10, :]

        sample = {'image': image, 'label': label, 'text': text}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
