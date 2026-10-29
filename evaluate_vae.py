import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from dataset import XRayDataset, transform
from model_vae import XrayVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

TEXT_WIDTH = 7.0  # inches, two-column paper text width
COLUMN_WIDTH = 3.3  # inches, single column width


def save_figure(fig, path):
    """Save as both vector PDF (for LaTeX) and PNG (for quick viewing)."""
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_grid(rows, path, row_labels=None, col_labels=None):
    """Save a list of (N, 1, H, W) tensors in [-1, 1], one tensor per row, as a borderless grayscale grid."""
    nrow, ncol = len(rows), len(rows[0])
    cell = TEXT_WIDTH / ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_WIDTH, cell * nrow), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for r, images in enumerate(rows):
        for c, img in enumerate(images.detach().cpu().clamp(-1, 1).squeeze(1).numpy()):
            axes[r, c].imshow(img, cmap="gray", vmin=-1, vmax=1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if row_labels is not None:
        for ax, label in zip(axes[:, 0], row_labels):
            ax.set_ylabel(label)
    if col_labels is not None:
        for ax, label in zip(axes[-1], col_labels):
            ax.set_xlabel(label)
    save_figure(fig, path)


def plot_reconstructions(reals, recons, path):
    """8x2 grid: inputs on top, reconstructions below."""
    save_grid([reals, recons], path, row_labels=["Input", "Recon."])


def plot_interpolation(model, x0, x1, path, steps=8):
    """8x1 row decoding the straight line between the posterior means of two test images."""
    mu = model.fc_mu(model.encoder(torch.stack([x0, x1]).to(device)))
    t = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    images = model.decoder((1 - t) * mu[0] + t * mu[1])
    save_grid([images], path, col_labels=[f"$t={v:.2f}$" for v in t.squeeze(1).tolist()])


def plot_samples(model, latent_dim, path, n=8):
    """8x1 row of samples decoded from z ~ N(0, I)."""
    save_grid([model.decoder(torch.randn(n, latent_dim, device=device))], path)


def plot_training_curves(training_curve, out, beta_warmup_epochs):
    """One single-column figure per loss term (total, reconstruction, KL), train vs. val."""
    ep = [t["epoch"] for t in training_curve]
    for key, ylabel in [("loss", "Negative ELBO"), ("recon", "MSE"), ("kl", "KL divergence (nats)")]:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
        ax.plot(ep, [t[f"train_{key}"] for t in training_curve], label="Train", lw=1.2)
        ax.plot(ep, [t[f"val_{key}"] for t in training_curve], label="Val", lw=1.2)
        if beta_warmup_epochs:
            ax.axvline(beta_warmup_epochs, color="0.6", ls="--", lw=0.8, label=r"End of $\beta$ warm-up")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, ep[-1])
        ax.legend(frameon=False)
        save_figure(fig, out / f"training_curve_{key}")


def encode_means(model, loader):
    """Posterior means mu(x) and labels for every image in loader."""
    mus, labels = [], []
    for x, y in loader:
        mus.append(model.fc_mu(model.encoder(x.to(device))).cpu())
        labels.append(y)
    return torch.cat(mus).numpy(), torch.cat(labels).numpy()


def linear_probe(train_z, train_y, test_z, test_y, seed):
    """Logistic regression on frozen posterior means. Classes are imbalanced (~3:1 pneumonia), so
    classes are reweighted and balanced accuracy / AUC are reported alongside accuracy."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed))
    clf.fit(train_z, train_y)
    pred = clf.predict(test_z)
    return {
        "accuracy": accuracy_score(test_y, pred),
        "balanced_accuracy": balanced_accuracy_score(test_y, pred),
        "roc_auc": roc_auc_score(test_y, clf.predict_proba(test_z)[:, 1]),
        "majority_class_accuracy": max(np.mean(test_y), 1 - np.mean(test_y)),
    }


def to_inception(x):
    x = (x.clamp(-1, 1) + 1) / 2
    return x.repeat(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint")
    parser.add_argument("--out-dir", default="eval_outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint)
    model_path = checkpoint_dir / "xray_vae.pt"
    config_path = checkpoint_dir / "config.json"
    training_curve_path = checkpoint_dir / "training_curve.json"

    with open(config_path) as f:
        config = json.load(f)

    with open(training_curve_path) as f:
        training_curve = json.load(f)

    plot_training_curves(training_curve, fig_dir, config.get("beta_warmup_epochs", 0))

    latent_dim = config["latent_dim"]
    model = XrayVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    train_ds = XRayDataset(split="train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_ds = XRayDataset(split="test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    n = len(test_ds)
    metrics = {"n_test_images": n}

    with torch.no_grad():
        mse_sum = 0.0
        fid_recon = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        fid_sample = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        inception_real = InceptionScore(normalize=True).to(device)
        inception_recon = InceptionScore(normalize=True).to(device)
        inception_sample = InceptionScore(normalize=True).to(device)

        reals = []
        recons = []
        labels = []

        for x, y in test_loader:
            x = x.to(device)
            recon = model.decoder(model.fc_mu(model.encoder(x)))
            samples = model.decoder(torch.randn(x.size(0), latent_dim, device=device))
            reals.append(x.cpu())
            recons.append(recon.cpu())
            labels.append(y)

            mse_sum += ((recon - x) ** 2).mean(dim=(1, 2, 3)).sum().item()

            real = to_inception(x)
            fake_recon = to_inception(recon)
            fake_sample = to_inception(samples)
            fid_recon.update(real, real=True)
            fid_recon.update(fake_recon, real=False)
            fid_sample.update(real, real=True)
            fid_sample.update(fake_sample, real=False)
            inception_real.update(real)
            inception_recon.update(fake_recon)
            inception_sample.update(fake_sample)

        metrics["test_recon_mse"] = mse_sum / n
        metrics["fid_recon"] = float(fid_recon.compute())
        metrics["fid_sample"] = float(fid_sample.compute())
        metrics["inception_real"] = float(inception_real.compute()[0])
        metrics["inception_recon"] = float(inception_recon.compute()[0])
        metrics["inception_sample"] = float(inception_sample.compute()[0])

        reals = torch.cat(reals)
        recons = torch.cat(recons)
        labels = torch.cat(labels)

        # Figures use a fixed generator so they are reproducible independently of the metric loop above
        g = torch.Generator().manual_seed(args.seed)
        for label, cls in enumerate(XRayDataset.classes):
            cls_idx = torch.nonzero(labels == label).squeeze(1)
            idx = cls_idx[torch.randperm(len(cls_idx), generator=g)[:args.n_show]]
            plot_reconstructions(reals[idx], recons[idx], fig_dir / f"reconstructions_{cls.lower()}")

        i, j = torch.randperm(n, generator=g)[:2].tolist()
        plot_interpolation(model, reals[i], reals[j], fig_dir / "interpolation", steps=args.n_show)

        torch.manual_seed(args.seed)
        plot_samples(model, latent_dim, fig_dir / "samples", n=args.n_show)

        train_z, train_y = encode_means(model, train_loader)
        test_z, test_y = encode_means(model, test_loader)

    metrics["linear_probe"] = linear_probe(train_z, train_y, test_z, test_y, args.seed)

    print(json.dumps(metrics, indent=1))

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
