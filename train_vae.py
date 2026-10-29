import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from dataset import XRayDataset, transform
from model_vae import XrayVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def elbo_loss(mse_fn, x, recon, mu, logvar, beta):
    recon_loss = mse_fn(recon, x) / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default="checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Using device:", device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=1)

    train_ds = XRayDataset(split="train", transform=transform)
    val_ds = XRayDataset(split="val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = XrayVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    mse_loss = nn.MSELoss(reduction="sum")

    warmup_steps = max(1, args.beta_warmup_epochs * len(train_loader))
    step = 0
    training_curve = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            mu, logvar, recon = model(x)
            beta = args.beta * min(1.0, step / warmup_steps)
            loss, recon_loss, kl = elbo_loss(mse_loss, x, recon, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            train_loss += loss.item() * x.size(0)
            train_recon += recon_loss.item() * x.size(0)
            train_kl += kl.item() * x.size(0)

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        n = len(train_ds)
        train_loss = train_loss / n
        train_recon = train_recon / n
        train_kl = train_kl / n

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                mu, logvar, recon = model(x)
                loss, recon_loss, kl = elbo_loss(mse_loss, x, recon, mu, logvar, beta=args.beta)
                val_loss += loss.item() * x.size(0)
                val_recon += recon_loss.item() * x.size(0)
                val_kl += kl.item() * x.size(0)
                
        n = len(val_ds)
        val_loss = val_loss / n
        val_recon = val_recon / n
        val_kl = val_kl / n

        print(f"epoch {epoch + 1}/{args.epochs} train loss {train_loss:.1f} "
              f"(recon {train_recon:.1f}, kl {train_kl:.1f})  val loss {val_loss:.1f} (recon {val_recon:.1f}, kl {val_kl:.1f})")

        training_curve.append({
            "epoch": epoch + 1,
            "lr": lr,
            "beta": beta,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_kl": train_kl,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kl": val_kl,
        })

    torch.save(model.state_dict(), out / "xray_vae.pt")
    with open(out / "training_curve.json", "w") as f:
            json.dump(training_curve, f, indent=1)


if __name__ == "__main__":
    main()
