# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Mapillary(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=66,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Mapillary, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]
        # print(self.img_list)
        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: 1, 
                              3: 4, 4: ignore_label, 
                              5: ignore_label, 6: 3, 
                              7: ignore_label, 8: ignore_label, 9: ignore_label, 
                              10: ignore_label, 11: ignore_label, 12: ignore_label, 
                              13: 0, 14: ignore_label, 15: 1, 
                              16: ignore_label, 17: 2, 18: ignore_label, 
                              19: 11, 20: 12, 21: 12, 22: 12, 23: ignore_label, 24: 0,
                              25: ignore_label, 26: ignore_label, 27: 10, 28: ignore_label, 
                              29: 9, 30: 8, 
                              31: ignore_label, 32: ignore_label, 33: ignore_label, 34:ignore_label, 35: ignore_label,36: ignore_label, 37: ignore_label, 38: ignore_label, 39: ignore_label, 40: ignore_label, 41:0, 42: ignore_label, 43: ignore_label, 44: ignore_label, 45:5, 46: ignore_label,47:5 ,48:6,49: ignore_label, 50:7, 51: ignore_label, 52:18, 53: ignore_label,54:15,55:13,56: ignore_label,57:17,58:16,59: ignore_label,60: ignore_label,61:14,62: ignore_label,63: ignore_label,64: ignore_label,65: ignore_label}
        self.class_weights = torch.FloatTensor([1.0] * num_classes).cuda()

        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843, 
        #                                 1.1116, 0.9037, 1.0865, 1.0955, 
        #                                 1.0865, 1.1529, 1.0507]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                # print(item)
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        # print(item)

        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'mapillary',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'mapillary',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
