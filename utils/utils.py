from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self,root,transforms=None):
        super(ImageFolderDataset).__init__()
        self.root = root
        self.transforms = transforms
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.png','.jpg','.jpeg'))]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        image_path = os.path.join(self.root,self.files[idx])
        image = Image.open(image_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        return image
    
def get_transforms(img_size,crop,final_size):
    transforms_list=[]

    if img_size>0:
        transforms_list.append(
            transforms.Resize((img_size,img_size))
        )

    if crop:
        transforms_list.append(
            transforms.RandomCrop(final_size)
        )

    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)

def adaptive_instance_normalization(content_feat, style_feat):
   
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
   
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std