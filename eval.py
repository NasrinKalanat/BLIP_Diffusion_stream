
from dataset_eval import DatasetEval

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import transforms
import os

from transformers import CLIPProcessor, CLIPModel
from torchmetrics import FID

class ThirdStageModel(nn.Module):
    def __init__(self, ckptdir="", device="cuda"):
        super(ThirdStageModel, self).__init__()

        self.device = device
        self.wlabels = ["Sunny/Clear", "Cloudy/Overcast", "Rainy", "Snowy", "Foggy/Misty", "Windy", "Stormy/Severe",
                        "Hot/Heatwave", "Cold/Cold Wave", "Mixed/Variable"]
        self.num_classes = len(self.wlabels)

        model_id = "openai/clip-vit-base-patch32"
        self.clip = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        label_tokens = self.processor(text=self.wlabels, padding=True, images=None, return_tensors='pt').to(self.device)
        self.label_features = self.clip.get_text_features(**label_tokens)

        hid_dim = 512
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, hid_dim, device=self.device), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(hid_dim, self.num_classes, device=self.device))
        # self.resnet = self.resnet.to(self.device)
        self.resnet = self.resnet.eval().to(self.device)
        self.fc_w = nn.Sequential(nn.Linear(self.resnet.fc.in_features, hid_dim, device=self.device), nn.ReLU(),
                                  nn.Dropout(p=0.1), nn.Linear(hid_dim, self.num_classes, device=self.device))
        self.fc_f = nn.Sequential(nn.Linear(self.resnet.fc.in_features, hid_dim, device=self.device), nn.ReLU(),
                                  nn.Dropout(p=0.1), nn.Linear(hid_dim, 1, device=self.device))
        self.resnet.fc = nn.Identity()
        # self.resnet_feat = FeatureExtractionModel(self.resnet)

        for name, module in self.named_children():
            if name in ['fc_f', 'fc_w']:  # Skip 'model' and 'resnet'
                print(name)
                module.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


    def save_checkpoint(self, model):
        if not os.path.exists(self.ckptdir):
            os.makedirs(self.ckptdir, exist_ok=True)
        print("Summoning checkpoint.")
        ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
        torch.save(model, ckpt_path)


    def load_checkpoint(self, model):
        print("Loading checkpoint.")
        ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
        pl_sd = torch.load(ckpt_path)  # , map_location="cpu")
        sd = pl_sd["state_dict"]
        model.load_state_dict(sd)


    # Function to compute reconstruction loss
    def compute_reconstruction_loss(self, decoded_images, original_images):
        loss = nn.L1Loss()(decoded_images, original_images)
        return loss


    def compute_mse_loss(self, pred_labels, gt_labels):
        loss = nn.MSELoss()(pred_labels, gt_labels)
        return loss


    # Function to compute class label loss
    def compute_entropy_loss(self, pred_labels, gt_labels):
        # Implement your class label loss computation
        # This could involve using a classification loss like CrossEntropyLoss
        loss = nn.BCEWithLogitsLoss()(pred_labels, gt_labels)
        # loss = F.binary_cross_entropy_with_logits(pred_labels, gt_labels)
        return loss

    def f1_score(self, y_pred, y_true, threshold=0.8):
        """
        Calculate F1 Score for multilabel classification.
        Args:
        y_pred (torch.Tensor): Predictions from the model (probabilities).
        y_true (torch.Tensor): Actual labels.
        threshold (float): Threshold for converting probabilities to binary output.

        Returns:
        float: F1 Score
        """
        # Binarize predictions and labels
        y_pred = (y_pred > threshold).int()
        y_true = y_true.int()

        # True positives, false positives, and false negatives
        tp = (y_true * y_pred).sum(dim=1).float()  # Element-wise multiplication for intersection
        fp = ((1 - y_true) * y_pred).sum(dim=1).float()
        fn = (y_true * (1 - y_pred)).sum(dim=1).float()

        # Precision, recall, and F1 for each label
        epsilon = 1e-7  # To avoid division by zero
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        # Average F1 score across all labels
        avg_f1 = f1.mean().item()
        return avg_f1


    def on_val_start(self):
        # ckpt_path = os.path.join(ckptdir, "last.ckpt")
        # print(f"Loading model from {ckpt_path}")
        # pl_sd = torch.load(ckpt_path)#, map_location="cpu")
        # sd = pl_sd["state_dict"]
        # model = instantiate_from_config(config.model)
        # m, u = model.load_state_dict(sd, strict=False)
        # self.model.to(self.device)
        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # for name, module in self.named_children():
        #     module.eval()
        #     for param in module.parameters():
        #         param.requires_grad = False

        # for name, param in model.named_parameters():
        #     print(f'{name}: requires_grad={param.requires_grad}')
        return


    def on_train_start(self):
        # ckpt_path = os.path.join(ckptdir, "last.ckpt")
        # print(f"Loading model from {ckpt_path}")
        # pl_sd = torch.load(ckpt_path)#, map_location="cpu")
        # sd = pl_sd["state_dict"]
        # model = instantiate_from_config(config.model)
        # m, u = model.load_state_dict(sd, strict=False)
        # self.model.to(self.device)
        for name, module in self.named_children():
            if name in ['fc_f', 'fc_w']:  # Skip 'model' and 'resnet'
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
            else:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            # for name, param in model.named_parameters():
        #     print(f'{name}: requires_grad={param.requires_grad}')
        return


    def test(self, loader):
        global_step = 0
        epoch = 0
        total_fid = 0
        total_acc = 0
        total_acc_clip = 0
        flabel_error = 0
        # refine = Refinement(x.shape[1:], self.device).to(self.device)
        # self.load_checkpoint(self.model)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                images, decoded_images, wlabels = batch
                images = images.to(self.device)
                decoded_images = decoded_images.to(self.device)
                wlabels = wlabels.to(self.device)

                fid = FID().cuda(device=self.device)
                fid.update(((images.clamp(-1., 1.) + 1.0) / 2.0 * 255).type(torch.uint8).cuda(device=self.device),
                           real=True)
                fid.update(((decoded_images.clamp(-1., 1.) + 1.0) / 2.0 * 255).type(torch.uint8).cuda(device=self.device),
                           real=False)
                # total_fid+=fid.compute()

                to_pil = transforms.ToPILImage()
                images_pil = [to_pil(((img.clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)) for img in images]
                inputs = self.processor(text=self.wlabels, images=images_pil, padding=True, return_tensors='pt')
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                images_clip = self.clip(**inputs)
                decoded_images_pil = [to_pil(((img.clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)) for img in
                                      decoded_images]
                inputs = self.processor(text=self.wlabels, images=decoded_images_pil, padding=True, return_tensors='pt')
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                decoded_images_clip = self.clip(**inputs)

                total_acc_clip += (torch.argmax(decoded_images_clip.logits_per_image, dim=1) == torch.argmax(
                    images_clip.logits_per_image, dim=1)).float().mean()
                # total_acc_clip+=(((decoded_images_clip.logits_per_image>0.8).int() * wlabels.int()).sum(dim=1)/wlabels.int().sum(dim=1)).mean()
                # total_acc_clip+=self.f1_score(decoded_images_clip.logits_per_image, torch.sigmoid(images_clip.logits_per_image, dim=1))
                resnet_out = self.resnet(decoded_images)
                pred_wlabels = self.fc_w(resnet_out)
                # total_acc+=(((pred_wlabels>0.8).int() * wlabels.int()).sum(dim=1)/wlabels.int().sum(dim=1)).mean()
                # total_acc+=(torch.argmax(pred_wlabels, dim=1)==torch.argmax(wlabels, dim=1)).float().mean()
                total_acc += self.f1_score(pred_wlabels, wlabels)

                pred_flabels = self.fc_f(resnet_out)
                flabel_error += self.compute_mse_loss(pred_flabels, flabels)

            total_fid = fid.compute()

        print(f'Total test w label accuracy: {total_acc / len(loader)}')
        print(f'Total test clip accuracy: {total_acc_clip / len(loader)}')
        print(f'Total test accuracy: {flabel_error.item() / len(loader)}')
        print(f'Total test FID: {total_fid / len(loader)}')
        return

    def run(self, imgs_gt_path, imgs_gen_path):
        print("**************start third_stage**************")
        batch_frequency = 16
        max_images = 32

        print("********** test **********")
        data_ft = DatasetEval(imgs_gt_path, imgs_gen_path)
        loader = torch.utils.data.DataLoader(data_ft, batch_size=4, shuffle=False)
        self.test(loader)

        # print("********** train **********")
        # data_ft = DatasetEval(root_data, split="train")
        # loader = torch.utils.data.DataLoader(data_ft, batch_size=4, shuffle=True)
        # self.train(loader, image_logger)

        print("********** test **********")

        data_ft = DatasetEval(imgs_gt_path, imgs_gen_path)
        loader = torch.utils.data.DataLoader(data_ft, batch_size=4, shuffle=False)
        self.test(loader)

if __name__ == '__main__':
    model = ThirdStageModel()
    model.run(imgs_gt_path='/data/nak168/spatial_temporal/stream_img/BLIP_Diffusion_stream/images', imgs_gen_path='/data/nak168/spatial_temporal/stream_img/BLIP_Diffusion_stream/output')

