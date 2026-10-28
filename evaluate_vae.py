import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore

from dataset import XRayDataset, transform
from model_vae import XrayVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_grid(rows, path, row_labels=None, title=None, size=2.0):
    """Save a list of (N, 1, H, W) tensors in [-1, 1], one tensor per row, as a grayscale image grid."""
    nrow, ncol = len(rows), len(rows[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow), squeeze=False)
    for r, images in enumerate(rows):
        for c, img in enumerate(images.detach().cpu().clamp(-1, 1).squeeze(1).numpy()):
            axes[r, c].imshow(img, cmap="gray", vmin=-1, vmax=1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    if row_labels is not None:
        for ax, label in zip(axes[:, 0], row_labels):
            ax.set_ylabel(label)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_grid(reals, recons, path, n=4):
    """4x2 grid randomly reconstruction vs. real"""
    idx = torch.randperm(len(reals))[:n]
    save_grid([reals[idx], recons[idx]], path, row_labels=["real", "recon"], title="Test set reconstructions")


def plot_samples_grid(samples, path, n=4):
    """4x1 grid randomly sampled"""
    save_grid([samples[:n]], path, title="Samples, z ~ N(0, I)")


def plot_training_curves(training_curve, path):
    ep = [t["epoch"] for t in training_curve]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(ep, [t["train_loss"] for t in training_curve], label="train")
    ax.plot(ep, [t["val_loss"] for t in training_curve], label="val")
    ax.set_title("Total loss")

    ax = axes[1]
    ax.plot(ep, [t["train_recon"] for t in training_curve], label="train")
    ax.plot(ep, [t["val_recon"] for t in training_curve], label="val")
    ax.set_title("Reconstruction (summed MSE)")

    ax = axes[2]
    ax.plot(ep, [t["train_kl"] for t in training_curve], label="train")
    ax.plot(ep, [t["val_kl"] for t in training_curve], label="val")
    ax.set_title("KL")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def to_inception(x):
    """[-1, 1] grayscale -> [0, 1] RGB at 299x299"""
    x = (x.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x.repeat(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoint", help="directory written by train_vae.py")
    parser.add_argument("--out-dir", default="eval_outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint)
    model_path = checkpoint_dir / "xray_vae.pt"
    config_path = checkpoint_dir / "config.json"
    training_curve_path = checkpoint_dir / "training_curve.json"

    with open(config_path) as f:
        config = json.load(f)

    with open(training_curve_path) as f:
        training_curve = json.load(f)

    plot_training_curves(training_curve, out / "training_curves.png")

    latent_dim = config["latent_dim"]
    model = XrayVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = XRayDataset(split="test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    n = len(test_ds)
    metrics = {"n_test_images": n}

    with torch.no_grad():
        mse_sum = 0.0
        fid_recon = FID(device=device)
        fid_sample = FID(device=device)
        inception_recon = InceptionScore(device=device)
        inception_sample = InceptionScore(device=device)

        reals = []
        recons = []

        for x, _ in test_loader:
            x = x.to(device)
            recon = model.decoder(model.fc_mu(model.encoder(x)))
            samples = model.decoder(torch.randn(x.size(0), latent_dim, device=device))
            reals.append(x.cpu())
            recons.append(recon.cpu())

            mse_sum += ((recon - x) ** 2).mean(dim=(1, 2, 3)).sum().item()

            real = to_inception(x)
            fake_recon = to_inception(recon)
            fake_sample = to_inception(samples)
            fid_recon.update((fake_recon, real))
            fid_sample.update((fake_sample, real))
            inception_recon.update(fake_recon)
            inception_sample.update(fake_sample)

        metrics["test_recon_mse"] = mse_sum / n
        metrics["fid_recon"] = float(fid_recon.compute())
        metrics["inception_recon"] = float(inception_recon.compute())
        metrics["fid_sample"] = float(fid_sample.compute())
        metrics["inception_sample"] = float(inception_sample.compute())

        reals = torch.cat(reals)
        recons = torch.cat(recons)
        plot_reconstruction_grid(reals, recons, out / "reconstructions.png")
        plot_samples_grid(samples, out / "samples.png")

    print(metrics)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
