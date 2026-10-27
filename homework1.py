from pathlib import Path

import kagglehub
import torch
import torchvision.transforms as T
from PIL import Image

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

class XRayDataset(torch.utils.data.Dataset):
  """Chest X-ray pneumonia dataset. Labels: 0 = NORMAL, 1 = PNEUMONIA."""

  classes = ["NORMAL", "PNEUMONIA"]

  def __init__(self, root=path, split="train", transform=None):
      assert split in ("train", "val", "test")
      self.transform = transform
      self.samples = []
      split_dir = Path(root) / "chest_xray" / split
      for label, cls in enumerate(self.classes):
          cls_dir = split_dir / cls
          for file in sorted(cls_dir.iterdir()):
              if file.suffix.lower() in (".jpeg", ".jpg", ".png"):
                  self.samples.append((file, label))

  def __len__(self):
      return len(self.samples)

  def __getitem__(self, idx):
      img_path, label = self.samples[idx]
      image = Image.open(img_path).convert("L")
      if self.transform is not None:
          image = self.transform(image)
      return image, label


# Images vary in size, so resize to a fixed shape before batching.
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

train_ds = XRayDataset(split="train", transform=transform)
val_ds = XRayDataset(split="val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

