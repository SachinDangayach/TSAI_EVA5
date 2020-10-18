from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile
import requests
from io import StringIO,BytesIO
import albumentations.pytorch as AP
import albumentations as A
import cv2
"""
This is used to download the tiny imagenet data set, load them, split to train test , convert to data set format
TinyImageNetDataSet - This is the main function which calls all all other.
Parameters :
train_split : The percent of train data. Default it is 70%
test_transforms :Transformations to apply for test data
train_transforms : Transformations to apply for train data
Return Value : train_set, test_set of type dataset. Which are ready to go in Dataloader
Description of How it is implemented :
TinyImageNetDataSet is the function which intern calls many funtions
1. Download_images - It dowloads the images from the given url and exact the zip file.
2. class_names - Derives the classes of tiny- Imagenet.
3. TinyImageNet - This returns the complete data of type data set.
4. Then we split the data we got from TinyImageNet class into train and test.
5. DatasetFromSubset - takes train or test data set and apply given transformations
Finaly trainset, testset are returned.
"""

# -----------------------------------------------------Main Function which calls everything--------------------------------------------------------------
def TinyImageNetDataSet(train_split = 70,test_transforms = None,train_transforms = None):

  down_url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  download_images(down_url)
  classes = class_names(url = "tiny-imagenet-200/wnids.txt")
  dataset = TinyImageNet(classes,url="tiny-imagenet-200")
  train_len = len(dataset)*train_split//100
  test_len = len(dataset) - train_len
  train_set, val_set = random_split(dataset, [train_len, test_len])
  train_dataset = DatasetFromSubset(train_set, transform=train_transforms)
  test_dataset = DatasetFromSubset(val_set, transform=test_transforms)

  return train_dataset, test_dataset,classes



# --------------------------------------------------------------Custom data set-------------------------------------------------------------------------

class TinyImageNet(Dataset):
    def __init__(self,classes,url):
        self.data = []
        self.target = []
        self.classes = classes
        self.url = url

        wnids = open(f"{url}/wnids.txt", "r")

        for wclass in notebook.tqdm(wnids,desc='Loading Train Folder', total = 200):
          wclass = wclass.strip()
          for i in os.listdir(url+'/train/'+wclass+'/images/'):
            img = Image.open(url+"/train/"+wclass+"/images/"+i)
            npimg = np.asarray(img)

            if(len(npimg.shape) ==2):

               npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
            self.data.append(npimg)
            self.target.append(self.classes.index(wclass))

        val_file = open(f"{url}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(val_file,desc='Loading Test Folder',total =10000 ):
          split_img, split_class = i.strip().split("\t")[:2]
          img = Image.open(f"{url}/val/images/{split_img}")
          npimg = np.asarray(img)
          if(len(npimg.shape) ==2):

                npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
          self.data.append(npimg)
          self.target.append(self.classes.index(split_class))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data
        return data,target



# ----------------------------------------------------Data subset which comes after splitting--------------------------------------------------

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# -------------------------------------------------------------------------------------------------------------------------------------------------------

def class_names(url = "tiny-imagenet-200/wnids.txt"):
  f = open(url, "r")
  classes = []
  for line in f:
    classes.append(line.strip())
  return classes

# ---------------------------------------------------Download Images---------------------------------------------------------------------------


def download_images(url):

    if (os.path.isdir("tiny-imagenet-200")):
        print ('Images already downloaded...')
        return
    r = requests.get(url, stream=True)
    print ('Downloading TinyImageNet Data' )
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
      zip_ref.extract(member = file)
    zip_ref.close()

class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, rimages, labels, transform=None):
        self.rimages = rimages
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rimages)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.rimages[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


def get_album_transforms(norm_mean,norm_std):
    """get the train and test transform by albumentations"""
    import cv2
    album_train_transform = A.Compose([
                                          A.PadIfNeeded(min_height=70, min_width=70, border_mode = cv2.BORDER_REFLECT, always_apply=True,),
                                          A.RandomCrop(height=64, width=64, always_apply=True),
                                          A.HorizontalFlip(p=0.5),
                                          A.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                                          A.Normalize(
                                             mean=norm_mean,
                                              std=norm_std, ),
                                          A.Cutout(1, 32, 32, p=0.4),
                                          AP.transforms.ToTensor()
                                        ])

    album_test_transform = A.Compose([   A.Normalize(
                                             mean=norm_mean,
                                              std=norm_std, ),
                                          AP.transforms.ToTensor()
                                        ])
    return(album_train_transform,album_test_transform)

# def get_datasets():
#     """Extract and transform the data"""
#     train_set = torchvision.datasets.CIFAR10(root='./data', train=True,download=True)
#     test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,download=True)
#     return(train_set,test_set)

# def trasnform_datasets(train_set, test_set, train_transform, test_transform):
#     """Transform the data"""
#     train_set = AlbumentationsDataset(
#                                     rimages= train_set.data,
#                                     labels=train_set.targets,
#                                     transform=train_transform,
#                                     )
#
#     test_set = AlbumentationsDataset(
#                                     rimages= test_set.data,
#                                     labels=test_set.targets,
#                                     transform=test_transform,
#                                     )
#     return(train_set,test_set)

def get_dataloaders(train_set,test_set,batch_size):
    """ Dataloader Arguments & Test/Train Dataloaders - Load part of ETL"""
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64, num_workers=1)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    # test dataloader
    test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
    return(train_loader,test_loader)
