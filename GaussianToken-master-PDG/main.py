from lightning.pytorch.cli import LightningCLI
import torch
import matplotlib
matplotlib.use("Agg")  # 无GUI后端，适合训练/服务器


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False


def main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
