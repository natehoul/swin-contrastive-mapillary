# from mapillary.mapillary_sls.msls import MSLS
from turtle import forward
from mapillary.mapillary_sls.datasets.msls import MSLS
from swin.models import build_model as swin_build_model
from swin.models.swin_transformer import SwinTransformer
from swin.config import get_config
from torchsummary import summary
from mmcv import Config, DictAction
from video_swin.mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
# Modify number of negatives
# dataset = MSLS('datasets/mapillary', seq_length=3, nNeg=1)

# negatives / positives into this one
config = "./swin/configs/swin/swin_base_patch4_window7_224_22k.yaml"
checkpoint = "./swin/checkpoints/swin_base_patch4_window7_224_22k.pth"

cfg = get_config(config)
swin_model = model = SwinTransformer(img_size=cfg.DATA.IMG_SIZE,
                                patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
                                in_chans=cfg.MODEL.SWIN.IN_CHANS,
                                num_classes=cfg.MODEL.NUM_CLASSES,
                                embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
                                depths=cfg.MODEL.SWIN.DEPTHS,
                                num_heads=cfg.MODEL.SWIN.NUM_HEADS,
                                window_size=cfg.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
                                qk_scale=cfg.MODEL.SWIN.QK_SCALE,
                                drop_rate=cfg.MODEL.DROP_RATE,
                                drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
                                ape=cfg.MODEL.SWIN.APE,
                                patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=cfg.TRAIN.USE_CHECKPOINT)

# Call forward_features to get feature vector
swin_model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

# config = "./video_swin/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py"
# checkpoint = "./video_swin/checkpoints/swin_base_patch244_window1677_sthv2.pth"

# cfg = Config.fromfile(config)
# video_swin_model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
# load_checkpoint(video_swin_model, checkpoint, map_location='cpu')

# video_swin_model = video_swin_model.backbone

# summary(swin_model, (3, 224, 224))
# summary(video_swin_model, (3, 3, 224, 224))

# add layer to project features to the same dimension

# play with nNeg, if > 1 use video swin
nNeg = 1

dataset = MSLS('C:\\Users\\Nate\\Desktop\\mapillary-sls', nNeg=nNeg, task='im2im')
dataset.update_subcache()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def euclidean_dist(self, x, y):
        return (x - y).pow(2).sum(1)

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        d_pos = self.euclidean_dist(a, p)
        d_neg = self.euclidean_dist(a, n)
        loss = torch.relu(d_pos - d_neg + self.margin)

        return loss.mean()



# triplet loss / e.g. 
# - a, p, n = dataset.__get_item__()
# - triplet between (video_swin(a), swin(p), swin/video_swin(n))

device = torch.device('cpu')

swin_model = nn.Sequential(
    swin_model,
    nn.Linear(1024, 128)
)

swin_model = swin_model.to(device)
optimizer = optim.Adam(swin_model.parameters(), lr=0.001)
criterion = TripletLoss()

swin_model.train()

batch_size = 100
shuffle = True
num_workers = 4

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

from tqdm.notebook import tqdm
epochs = 5

for epoch in tqdm(range(epochs), desc="Epochs"):
    running_loss = []

    for step, (a, p, n) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
        a = torchvision.transforms.Resize((224, 224))(a)
        p = torchvision.transforms.Resize((224, 224))(p)
        n = torchvision.transforms.Resize((224, 224))(n)

        a = a.to(device)
        p = p.to(device)
        n = n.to(device)

        optimizer.zero_grad()

        a_out = swin_model(a)
        p_out = swin_model(p)
        n_out = swin_model(n)

        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.cpu().detach().numpy())

    print('Epoch: {} / {} - Loss: {:.4f}'.format(epoch + 1, epochs, np.mean(running_loss)))
