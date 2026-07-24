import os.path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

from gstk.models.gqgan import GQGAN
from gstk.data.dataset import MiniImagenet
from gstk.data.imagenet import ImageNetValidation

print("available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
torch.cuda.set_device(0)
L.seed_everything(0)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True  # True
torch.backends.cudnn.benchmark = False  # False


def load_gqgan(config, ckpt_path=None):
    model = GQGAN(**config.model.init_args)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)

    model = model.cuda()

    return model.eval()


def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)

    return parser.parse_args()


def main(args):
    config = OmegaConf.load(args.config_file)
    model = load_gqgan(config, ckpt_path=args.ckpt_path)

    dataset = MiniImagenet(split='val', transform='val')
    # MyCIFAR()
    # dataset = ImageNetValidation(config={'size': 256, 'subset': None})
    # indices = np.arange(len(dataset))
    print("dataset size:", len(dataset))
    exit()
    np.random.seed(42)
    indices = np.random.randint(0, len(dataset) + 1, size=1000)
    save_path = args.config_file.replace("config.yaml", "tmp_rec2")
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for i in indices:
            images = dataset[i]["image"]
            images = images.unsqueeze(0).cuda()

            save_image(images, os.path.join(save_path, f"{i}_ori.png"), nrow=4, normalize=True, value_range=(-1, 1))

            if model.use_ema:
                with model.ema_scope():
                    reconstructed_images, indices, loss_commit, gaussians = model(images)
            else:
                reconstructed_images, indices, loss_commit, gaussians = model(images)

            save_image(reconstructed_images, os.path.join(save_path, f"{i}_rec.png"), nrow=4, normalize=True,
                       value_range=(-1, 1))


if __name__ == "__main__":
    args = get_args()
    main(args)