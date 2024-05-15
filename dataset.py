import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import torch
from PIL import Image

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQ(FacesBase):
    def __init__(self, size, path, keys):
        super().__init__()
        self.size=size
        images_path=os.path.join(path,"images")
        labels_path=os.path.join(path,"labels")

        images=os.listdir(images_path)
        labels=os.listdir(labels_path)

        images.sort(key=lambda x: int(x.split(".")[0]))
        labels.sort(key=lambda x: int(x.split(".")[0]))

        self.images = [os.path.join(images_path, img) for img in images]
        self.labels = [os.path.join(labels_path, lbl) for lbl in labels]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_name=self.images[i]
        label_name=self.labels[i]

        im = Image.open(img_name)
        im = im.resize((self.size,self.size))
        im = np.array(im).astype(np.uint8)
        # im = np.transpose(im, (2, 0, 1))
        im = (im/127.5 - 1.0).astype(np.float32)

        label = Image.open(label_name)
        label = label.resize((self.size,self.size), Image.NEAREST)
        label = np.array(label).astype(np.uint8)
        label = label[...]
        label = torch.from_numpy(label).to(torch.int64)

        return {"image":im, "label":label}

class DeepFashion(FacesBase):
    def __init__(self, size, path, train=True):
        super().__init__()
        self.size=size
        if train:
            prefix="train"
        else:
            prefix="test"
        images_path=os.path.join(path,f"{prefix}_img")
        labels_path=os.path.join(path,f"{prefix}_label")

        images=os.listdir(images_path)

        self.images = [os.path.join(images_path, img) for img in images]
        self.labels = [os.path.join(labels_path, img.replace(".jpg", ".png")) for img in images]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_name=self.images[i]
        label_name=self.labels[i]

        im = Image.open(img_name)
        im = im.resize((self.size,self.size))
        im = np.array(im).astype(np.uint8)
        # im = np.transpose(im, (2, 0, 1))
        im = (im/127.5 - 1.0).astype(np.float32)

        label = Image.open(label_name)
        label = label.resize((self.size,self.size), Image.NEAREST)
        label = np.array(label).astype(np.uint8)
        label = label[:,:,0]
        label = torch.from_numpy(label).to(torch.int64)

        return {"image":im, "label":label}

class Cityscape(FacesBase):
    def __init__(self, size, path, prefix="train"):
        super().__init__()
        self.size=size
        images_path=os.path.join(path,"leftImg8bit", prefix)
        labels_path=os.path.join(path,"gtFine", prefix)

        #images path contains subfolders
        images=[]
        labels=[]
        for root, dirs, files in os.walk(images_path):
            for file in files:
                images.append(os.path.join(root, file)) 
                file_prefix = file.split("_leftImg8bit.png")[0]
                labels.append(os.path.join(labels_path, root.split("/")[-1], file_prefix+"_gtFine_labelIds.png"))

        self.images = images
        self.labels = labels
        
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_name=self.images[i]
        label_name=self.labels[i]

        im = Image.open(img_name)
        im = im.resize((self.size*2,self.size))
        im = np.array(im).astype(np.uint8)
        # im = np.transpose(im, (2, 0, 1))
        im = (im/127.5 - 1.0).astype(np.float32)

        label = Image.open(label_name)
        label = label.resize((self.size*2,self.size), Image.NEAREST) #width, height
        label = np.array(label).astype(np.uint8)
        label = label[...]
        label = torch.from_numpy(label).to(torch.int64)

        return {"image":im, "label":label}

class Ade20K(FacesBase):
    def __init__(self, size, path, prefix="training"):
        super().__init__()
        self.size=size
        images_path=os.path.join(path,"images", prefix)
        labels_path=os.path.join(path,"annotations", prefix)

        #images path contains subfolders
        images=os.listdir(images_path)
        labels=os.listdir(labels_path)
    
        images.sort(key = lambda x: int(x.split("_")[-1].split(".")[0]))
        labels.sort(key = lambda x: int(x.split("_")[-1].split(".")[0]))

        self.images = [os.path.join(images_path, img) for img in images]
        self.labels = [os.path.join(labels_path, lbl) for lbl in labels]
        
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_name=self.images[i]
        label_name=self.labels[i]

        im = Image.open(img_name)
        im = im.resize((self.size,self.size))
        im = np.array(im).astype(np.uint8)
        # im = np.transpose(im, (2, 0, 1))
        im = (im/127.5 - 1.0).astype(np.float32)

        label = Image.open(label_name)
        label = label.resize((self.size,self.size), Image.NEAREST) #width, height
        label = np.array(label).astype(np.uint8)

        label = label[...]
        label = torch.from_numpy(label).to(torch.int64)

        return {"image":im, "label":label}


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
