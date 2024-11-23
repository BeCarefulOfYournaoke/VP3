from .getData import MyDataSet
from torchvision import datasets, transforms
from .One_Level import ReadOne
from .Multi_Level import ReadTwo
from PIL import ImageFilter
import random
from .RandAugment import RandAugment


# train_dataset = datasets.ImageFolder(root=data_folder,
#                                             transform=TwoCropTransform(train_transform))
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform_01, transform_02):
        self.transform_01 = transform_01
        self.transform_02 = transform_02
    def __call__(self, x):
        return [self.transform_01(x), self.transform_02(x)]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



def get_dataset(data_path, name, image_size=224, train=True, pattern='multi'):
    
    # train_transforms = transforms.Compose([
    #                                     transforms.RandomResizedCrop((image_size, image_size), (0.75, 1.)),
    #                                     # transforms.Resize((image_size, image_size)),
    #                                     transforms.RandomGrayscale(p=0.1),
    #                                     # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     #transforms.RandomVerticalFlip(),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                   ])
    # test_transforms = transforms.Compose([
    #                                     transforms.Resize((image_size, image_size)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                   ])

    strong_transforms = transforms.Compose([
                            transforms.RandomResizedCrop(image_size, scale=(0.25, 1.)),
                            # transforms.Resize((image_size, image_size)),
                            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                            transforms.RandomApply([
                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    weak_transforms = transforms.Compose([
                            # transforms.RandomResizedCrop(256, scale=(0.5, 1.)),
                            # transforms.CenterCrop(image_size),

                            # transforms.RandomResizedCrop(image_size, scale=(0.25, 1.)),
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # RandAugment(3, 8),

                            transforms.RandomResizedCrop(image_size, scale=(0.25, 1.)),
                            transforms.RandomHorizontalFlip(p=0.5),

                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

    test_transforms = transforms.Compose([
                            # transforms.Resize(256),
                            # transforms.CenterCrop(image_size),
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
    
    rand_transforms = transforms.Compose([
                            # transforms.RandomResizedCrop(256, scale=(0.25, 1.)),
                            # # transforms.RandomHorizontalFlip(p=0.5),
                            # RandAugment(3, 10),
                            # transforms.CenterCrop(image_size),

                            transforms.RandomResizedCrop(image_size, scale=(0.25, 1.)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            RandAugment(3, 8),

                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
    
    if train==True:
        if pattern == 'multi':
            if name == 'CIFAR20':
                dataset = ReadTwo(data_path, transform = TwoCropTransform(weak_transforms, rand_transforms))
            elif name == 'Travel20':
                dataset = ReadOne(data_path, transform = TwoCropTransform(weak_transforms, rand_transforms))
            elif name == 'ILSVRC20':
                dataset = ReadOne(data_path, transform = TwoCropTransform(weak_transforms, rand_transforms))
            elif name == 'Place20':
                dataset = ReadOne(data_path, transform = TwoCropTransform(weak_transforms, rand_transforms))
            elif name == 'VOC':
                dataset = ReadOne(data_path, transform = TwoCropTransform(weak_transforms, rand_transforms))
                

    elif train==False:
        if pattern == 'multi':
            if name == 'CIFAR20':
                dataset = ReadTwo(data_path, transform = TwoCropTransform(test_transforms, rand_transforms))
            elif name == 'Travel20':
                dataset = ReadOne(data_path, transform = TwoCropTransform(test_transforms, rand_transforms))
            elif name == 'ILSVRC20':
                dataset = ReadOne(data_path, transform = TwoCropTransform(test_transforms, rand_transforms))
            elif name == 'Place20':
                dataset = ReadOne(data_path, transform = TwoCropTransform(test_transforms, rand_transforms))
            elif name == 'VOC':
                dataset = ReadOne(data_path, transform = TwoCropTransform(test_transforms, rand_transforms))
    return dataset
    
