'''
Training script for the first CLIP+MSI model.
This uses MSI_Embedder 1 to preprocess the input MSI images.
'''

# IMPORTS ---------------------------------------------------------------------
import argparse
import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import gc
import warnings
import rasterio
import pickle
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.preprocessing import LabelEncoder

# SET DEVICE ------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore")

# CONFIG ----------------------------------------------------------------------
data_path = '../EuroSAT_MSI_data/' # the script will run inside the thesis/ folder
checkpoint_dir = './checkpoints/'
log_dir = './logs/'
tb_log_dir = './tb_logs/'
batch_size = 32
num_workers = 4 #2
#max_epochs = 10 # pass parameter from command line
in_channels = 13
out_channels = 3
learning_rate = 5e-3 #1e-3

# LABEL MAPPING ---------------------------------------------------------------
labels_map = {
    'SeaLake': 'Sea or Lake',
    'Pasture': 'Pasture',
    'PermanentCrop': 'Permanent Crop',
    'Residential': 'Residential Buildings',
    'Industrial': 'Industrial Buildings',
    'HerbaceousVegetation': 'Herbaceous Vegetation',
    'Highway': 'Highway',
    'Forest': 'Forest',
    'River': 'River',
    'AnnualCrop': 'Annual Crop'
}
labels = list(labels_map.values())

# DATA SPLIT ------------------------------------------------------------------
def build_split(seed=42):
    random.seed(seed)
    test_images, val_images, train_images = [], [], []

    for folder in os.listdir(data_path):
        imgs = os.listdir(os.path.join(data_path, folder))
        test = random.sample(imgs, 500)
        remain = [img for img in imgs if img not in test]
        val = random.sample(remain, 500)
        train = random.sample([img for img in remain if img not in val], 1000)
        test_images += test
        val_images += val
        train_images += train

    def to_df(image_list):
        data = []
        for img_id in image_list:
            folder = img_id.split('_')[0]
            data.append({
                'image_id': img_id,
                'filename': os.path.join(data_path, folder, img_id),
                'label': labels_map[folder]
            })
        return pd.DataFrame(data)

    return to_df(train_images), to_df(val_images), to_df(test_images)

# NORMALIZATION ---------------------------------------------------------------
def normalize(array):
    array = array.astype(np.float32)
    return (array - array.min()) / (array.max() - array.min())

# DATASET ---------------------------------------------------------------------
class EuroSATMSIDataset(Dataset):
    def __init__(self, dataframe, label2idx, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filename"]
        label = self.label2idx[self.df.loc[idx, "label"]]
        with rasterio.open(img_path) as src:
            img = src.read() # shape: [C, H, W] == [13, 64, 64]
        img = normalize(img) # put image pixel values in [0, 1]
        img = torch.tensor(img, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

# DATALOADER MODULE -----------------------------------------------------------
class EuroSATDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, label2idx, batch_size):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label2idx = label2idx
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EuroSATMSIDataset(self.train_df, self.label2idx)
        self.val_dataset = EuroSATMSIDataset(self.val_df, self.label2idx)
        self.test_dataset = EuroSATMSIDataset(self.test_df, self.label2idx)

    def _dataloader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=(device == 'cuda'))

    def train_dataloader(self): return self._dataloader(self.train_dataset, True)
    def val_dataloader(self): return self._dataloader(self.val_dataset, False)
    def test_dataloader(self): return self._dataloader(self.test_dataset, False)

# EMBEDDER MODEL --------------------------------------------------------------
class MSIEmbedder2(nn.Module):
    def __init__(self, in_channels):
        super(MSIEmbedder2, self).__init__()
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 1), # 32, 128 possible other choices
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size = 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.proj2(x)


# MAIN MODEL ------------------------------------------------------------------
class CLIPWithMSIEmbedder2(L.LightningModule):
    def __init__(self, in_channels, class_names, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.clip_model, _ = clip.load("ViT-B/32", device=device, download_root=os.path.expanduser("~/.cache/clip"))
        
        # Freeze CLIP parameters 
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = self.clip_model.encode_text
        self.embedder = MSIEmbedder2(in_channels)

        prompts = [f"a satellite photo of {name.lower()}" for name in class_names]
        tokenized = clip.tokenize(prompts)
        with torch.no_grad():
            text_features = self.text_encoder(tokenized.to(device)).detach()
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.MSI_to_CLIP_preprocess = T.Compose([
            T.Resize((224, 224), antialias=False),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, x):
        x = self.embedder(x)
        x = self.MSI_to_CLIP_preprocess(x)
        image_features = self.image_encoder(x)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def predict_logits(self, x):
        image_features = self(x) 
        return 100.0 * image_features @ self.text_features.T # return logits

    def _step(self, batch, stage):
        x, y = batch
        logits = self.predict_logits(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): self._step(batch, "val")
    def test_step(self, batch, batch_idx): self._step(batch, "test")

    def configure_optimizers(self):
        return Adam(self.embedder.parameters(), lr=self.hparams.learning_rate)

# MAIN EXECUTION --------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    max_epochs = args.max_epochs

    train_df, val_df, test_df = build_split()
    le = LabelEncoder().fit(train_df['label'])
    label2idx = {label: idx for idx, label in enumerate(le.classes_)}
    class_names = list(label2idx.keys())

    data_module = EuroSATDataModule(train_df, val_df, test_df, label2idx, batch_size)

    model = CLIPWithMSIEmbedder2(in_channels, class_names, learning_rate)
    #model = torch.compile(model)

    checkpoint_callback_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="clip-msi2-eurosat-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        save_last = True
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=10,
    #     mode="min"
    # )

    logger = CSVLogger(save_dir=log_dir, name="clip-msi2-eurosat")
    logger_tb = TensorBoardLogger(tb_log_dir, name="clip-msi2-eurosat")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_callback_best], #, early_stop_callback
        logger=[logger, logger_tb]
    )

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
