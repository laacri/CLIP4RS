# Load libraries
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
import h5py
import json
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


device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore")

#torch.set_float32_matmul_precision('medium' | 'high')

data_path = '../m-so2sat/' # the script will run inside the thesis/ folder
checkpoint_dir = './checkpoints/'
log_dir = './logs/'
tb_log_dir = './tb_logs/'
batch_size = 32
num_workers = 4
learning_rate = 1e-3 #5e-3


so2sat_labels = ['Compact high rise',
                  'Compact mid rise',
                  'Compact low rise',
                  'Open high rise',
                  'Open mid rise',
                  'Open low rise',
                  'Lightweight low rise',
                  'Large low rise',
                  'Sparsely built',
                  'Heavy industry',
                  'Dense trees',
                  'Scattered trees',
                  'Bush, scrub',
                  'Low plants',
                  'Bare rock or paved',
                  'Bare soil or sand',
                  'Water']
#n_bands = 18


# Normalization function
def compute_band_stats_up(df, bands):
    ''' Compute mean and std using two-pass definition-based method. '''
    n_bands = len(bands)
    band_pixel_counts = np.zeros(n_bands, dtype=np.int64)
    band_sum = np.zeros(n_bands, dtype=np.float64)

    # First pass: compute mean
    for filename in df['filename']:
        try:
            with h5py.File(filename, 'r') as f:
                for i, b in enumerate(bands):
                    band = f[b][:]
                    band_sum[i] += band.sum()
                    band_pixel_counts[i] += band.size
        except Exception as e:
            print(f"Failed reading {filename}: {e}")

    band_means = band_sum / band_pixel_counts

    # Second pass: compute variance
    band_sq_diff_sum = np.zeros(n_bands, dtype=np.float64)
    for filename in df['filename']:
        try:
            with h5py.File(filename, 'r') as f:
                for i, b in enumerate(bands):
                    band = f[b][:]
                    diff = band - band_means[i]
                    band_sq_diff_sum[i] += np.square(diff).sum()
        except Exception as e:
            print(f"Failed reading {filename}: {e}")

    band_vars = band_sq_diff_sum / band_pixel_counts
    band_stds = np.sqrt(band_vars)

    band_stats = pd.DataFrame({
        'Band': bands,
        'Mean': band_means,
        'Std': band_stds,
        'Pixels': band_pixel_counts
    })

    return band_stats


# Dataset and DataModule class
class GEOBenchMSIDataset(Dataset):
    def __init__(self, dataframe, label2idx, band_stats, selected_bands):
        self.df = dataframe.reset_index(drop=True)
        self.label2idx = label2idx
        self.band_stats = band_stats
        self.selected_bands = selected_bands  # new variable

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filename"]
        label = self.label2idx[self.df.loc[idx, "label"]]

        with h5py.File(img_path, 'r') as f:
            band_data = []
            for band_name in self.selected_bands:
                if band_name in f.keys():
                    band = f[band_name][:]
                    stats = self.band_stats[band_name]
                    band = (band - stats["mean"]) / (stats["std"] + 1e-6)
                    band_data.append(band)
                else:
                    raise KeyError(f"Band {band_name} not found in file {img_path}")

        img = np.stack(band_data, axis=0)  # [C, H, W]
        img = torch.tensor(img, dtype=torch.float32)
        return img, label


class GEOBenchMSIDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, label2idx, band_stats, batch_size, selected_bands):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label2idx = label2idx
        self.band_stats = band_stats # train_df band stats
        self.batch_size = batch_size
        self.selected_bands = selected_bands

    def setup(self, stage=None):
        self.train_dataset = GEOBenchMSIDataset(self.train_df, self.label2idx, self.band_stats, self.selected_bands)
        self.val_dataset   = GEOBenchMSIDataset(self.val_df, self.label2idx, self.band_stats, self.selected_bands)
        self.test_dataset  = GEOBenchMSIDataset(self.test_df, self.label2idx, self.band_stats, self.selected_bands)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == 'cuda'))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda'))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda'))
    
    
class MSIEmbedder1(nn.Module):
    def __init__(self, in_channels: int):
        super(MSIEmbedder1, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 1)
        # stride=1, padding=0 by default
        # 1x1 convolution, 1 pixel -> 1 pixel (kernel_size = 1)

    def forward(self, x):
        return self.proj(x) 
    

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


class CLIPWithMSIEmbedder1(L.LightningModule):
    def __init__(self, in_channels, class_names, learning_rate, class_weights):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.class_weights = class_weights

        # Load CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=device, download_root=os.path.expanduser("~/.cache/clip"))
        # _ is because we can't use clip preprocess, as it outputs a tensor
        # but only accepts PIL images as inputs and that's not what MSIEmbedder gives us
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = self.clip_model.encode_text

        # MSI embedder
        self.embedder = MSIEmbedder1(in_channels)

        # Compute text embeddings once for all
        prompts = [f"a satellite photo of {name.lower()}" for name in class_names]
        tokenized = clip.tokenize(prompts)
        with torch.no_grad():
            self.text_features = self.text_encoder(tokenized.to(device)).detach()  # [num_classes, 512]
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            #self.register_buffer("text_features", self.text_features) # good practice for constant tensors (like text_features).

        self.learning_rate = learning_rate

        # Transforms to preprocess images after MSIEmbedder, before CLIP
        self.MSI_to_CLIP_preprocess = T.Compose([
            T.Resize((224, 224), antialias=False),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, x):
        x = self.embedder(x)  # [B, 3, 64, 64]
        x = self.MSI_to_CLIP_preprocess(x)
        image_features = self.image_encoder(x)  # [B, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features  # just return the features, compute similarity later

    def predict_logits(self, x):
        image_features = self(x)  # [B, 512]
        logits = 100.0 * image_features @ self.text_features.T  # [B, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = F.cross_entropy(logits, y, weight=weights) # weights to balance loss function
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = F.cross_entropy(logits, y, weight=weights) # weights to balance loss function
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = F.cross_entropy(logits, y, weight=weights) # weights to balance loss function
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.embedder.parameters(), lr=self.learning_rate)
    

class CLIPWithMSIEmbedder2(L.LightningModule):
    def __init__(self, in_channels, class_names, learning_rate, class_weights):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.class_weights = class_weights

        # Load CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=device, download_root=os.path.expanduser("~/.cache/clip"))
        # _ is because we can't use clip preprocess, as it outputs a tensor
        # but only accepts PIL images as inputs and that's not what MSIEmbedder gives us
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = self.clip_model.encode_text

        # MSI embedder
        self.embedder = MSIEmbedder2(in_channels)

        # Compute text embeddings once for all
        prompts = [f"a satellite photo of {name.lower()}" for name in class_names]
        tokenized = clip.tokenize(prompts)
        with torch.no_grad():
            self.text_features = self.text_encoder(tokenized.to(device)).detach()  # [num_classes, 512]
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            #self.register_buffer("text_features", self.text_features) # good practice for constant tensors (like text_features).

        self.learning_rate = learning_rate

        # Transforms to preprocess images after MSIEmbedder, before CLIP
        self.MSI_to_CLIP_preprocess = T.Compose([
            T.Resize((224, 224), antialias=False),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, x):
        x = self.embedder(x)  # [B, 3, 64, 64]
        x = self.MSI_to_CLIP_preprocess(x)
        image_features = self.image_encoder(x)  # [B, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features  # just return the features, compute similarity later

    def predict_logits(self, x):
        image_features = self(x)  # [B, 512]
        logits = 100.0 * image_features @ self.text_features.T  # [B, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = F.cross_entropy(logits, y, weight=weights) # weights to balance loss function
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = F.cross_entropy(logits, y, weight=weights) # weights to balance loss function
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = F.cross_entropy(logits, y, weight=weights) # weights to balance loss function
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.embedder.parameters(), lr=self.learning_rate)
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model', type=int, choices=[1, 2], required=True, help='Model version: 1 or 2')
    args = parser.parse_args()
    max_epochs = args.max_epochs


    # Build TEST, VALID and TRAIN dataframes
    # images_folder = 'm-brick-kiln'
    # images_path = os.path.join(data_path, images_folder)
    images_path = data_path

    # Read default partition
    with open(os.path.join(images_path, 'default_partition.json'), 'r') as file:
        def_partition = json.load(file)
        print("\nSplits:")
        print(list(def_partition.keys()))
        print(len(def_partition['train']), len(def_partition['valid']), len(def_partition['test']))

    # Read test, val and train dataframe from files
    print("\nTEST set: -------------------------------------------------------------")
    test_df = pd.read_csv('../test_df_m-so2sat.csv')
    print(test_df.head())

    print("\nVALIDATION set: -------------------------------------------------------")
    val_df = pd.read_csv('../val_df_m-so2sat.csv')
    print(val_df.head())

    print("\nTRAIN set: ------------------------------------------------------------")
    train_df = pd.read_csv('../train_df_m-so2sat.csv')
    print(train_df.head())

    path_prefix = "/content/drive/MyDrive/Thesis CLIP4EO/GEO-Bench classification data/m-so2sat/"
    for df in [train_df, val_df, test_df]:
        df["filename"] = df["filename"].str.replace(path_prefix, data_path, regex=False)


    # Extract band stats for normalization
    selected_bands = [#'01 - VH.Real',
                      '02 - Blue',
                      #'02 - VH.Imaginary',
                      '03 - Green',
                      #'03 - VV.Real',
                      '04 - Red',
                      #'04 - VV.Imaginary',
                      '05 - VH.LEE Filtered',
                      '05 - Vegetation Red Edge',
                      '06 - VV.LEE Filtered',
                      '06 - Vegetation Red Edge',
                      #'07 - VH.LEE Filtered.Real',
                      '07 - Vegetation Red Edge',
                      '08 - NIR',
                      #'08 - VV.LEE Filtered.Imaginary',
                      '08A - Vegetation Red Edge',
                      '11 - SWIR',
                      '12 - SWIR']


    band_stats = compute_band_stats_up(train_df, selected_bands)
    stats_dict = {f"{row['Band']}": {
                    "mean": row["Mean"],
                    "std": row["Std"]}
                for _, row in band_stats.iterrows()}

    print(stats_dict)
    

    # 1. Encode string labels into integers
    le = LabelEncoder()
    le.fit(train_df["label"])
    label2idx = {label: idx for idx, label in enumerate(le.classes_)}
    class_names = list(label2idx.keys())

    # 2. Load data module
    data_module = GEOBenchMSIDataModule(train_df, val_df, test_df, label2idx, stats_dict, batch_size, selected_bands)

    class_counts = train_df['label'].value_counts()
    total = sum(class_counts)
    class_weights = {label2idx[label]: total / count for label, count in class_counts.items()}
    norm_factor = sum(class_weights.values()) # normalize for stability
    class_weights = {k: v / norm_factor for k, v in class_weights.items()}
    # {0: 0.058, 1: 0.058, 2: 0.058, ...} the dataset is balanced

    # Convert to tensor for use in model
    weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32)
    # tensor([0.58, 0.58, 0.58, ...]) the dataset is balanced

    # 3. Create the model -- edit for binary classification
    if args.model == 1:
        model = CLIPWithMSIEmbedder1(
            in_channels = len(selected_bands),
            class_names = class_names,
            learning_rate = learning_rate,
            class_weights = weights_tensor
        )

        # 4. Specify a checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clip-msi1-geobench-so2sat-{epoch:02d}-{val_acc:.4f}", # not a format string, values will be filled at runtime
            save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
            monitor="val_acc",
            mode="max",
            save_last = True
        )

        # # 5. Specify logger in csv format
        logger = CSVLogger(save_dir=log_dir, name="clip-msi1-geobench-so2sat")

        # define the logger object
        logger_tb = TensorBoardLogger(tb_log_dir, name = "clip-msi1-geobench-so2sat", log_graph = True)

    elif args.model == 2:
        model = CLIPWithMSIEmbedder2(
            in_channels = len(selected_bands),
            class_names = class_names,
            learning_rate = learning_rate,
            class_weights = weights_tensor
        )

        # 4. Specify a checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clip-msi2-geobench-so2sat-{epoch:02d}-{val_acc:.4f}", # not a format string, values will be filled at runtime
            save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
            monitor="val_acc",
            mode="max",
            save_last = True
        )

        # # 5. Specify logger in csv format
        logger = CSVLogger(log_dir, name="clip-msi2-geobench-so2sat")

        # define the logger object
        logger_tb = TensorBoardLogger(tb_log_dir, name = "clip-msi2-geobench-so2sat", log_graph = True)

    else:
        raise ValueError("Unsupported model type. Choose --model 1 or 2.")


    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision = 32,
        gradient_clip_val=1.0,
        #amp_backend=None,
        #precision="16-mixed", # enable AMP (Automatic Mixed Precision) # -> error after first epoch
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        logger=[logger, logger_tb]
    )

    # 7. Training
    trainer.fit(model, datamodule=data_module)

    trainer.validate(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)




if __name__ == "__main__":
    main()

