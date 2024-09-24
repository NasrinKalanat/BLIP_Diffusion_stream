import os
import glob
from PIL import Image
from torchvision import transforms
class DatasetEval(object):
    def __init__(self, imgs_gt_path, imgs_gen_path, image_size=256):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.imgs_gt_path = imgs_gt_path
        self.imgs_gen_path = imgs_gen_path
        self.imgs_gt = {}
        self.imgs_gen = {}
        self.labels = {}
        self.ids = []
        for file in glob.glob(f"{imgs_gt_path}/*.png"):
            print(file)
            _, idx, _, lbl = file.split(".png")[0].split("_")
            self.imgs_gt[idx]=file
            self.labels[idx]=lbl
            self.ids.append(idx)
        for file in glob.glob(f"{imgs_gen_path}/*.png"):
            _, idx, _, _ = file.split("_")
            self.imgs_gen[idx]=file

    def __getitem__(self, id):
        img_gt_path = os.path.join(self.imgs_gt_path, self.imgs_gt[self.ids[id]])
        img_gen_path = os.path.join(self.imgs_gen_path, self.imgs_gen[self.ids[id]])
        lbl = self.labels[self.ids[id]]

        img_gt = self.transform(Image.open(img_gt_path))
        img_gen = self.transform(Image.open(img_gen_path))
        return img_gt, img_gen, lbl

