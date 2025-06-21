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

checkpoint_dir = './checkpoints/'
log_dir = './logs/'
tb_log_dir = './tb_logs/'
batch_size = 32
num_workers = 4
learning_rate = 1e-3 #1e-4, 1e-5


# Loading and so2sat labels
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


# Loading and mapping EuroSAT labels
eurosat_labels_map = {'SeaLake' : 'Sea or Lake',
                      'Pasture': 'Pasture',
                      'PermanentCrop': 'Permanent Crop',
                      'Residential': 'Residential Buildings',
                      'Industrial': 'Industrial Buildings',
                      'HerbaceousVegetation': 'Herbaceous Vegetation',
                      'Highway': 'Highway',
                      'Forest': 'Forest',
                      'River': 'River',
                      'AnnualCrop': 'Annual Crop'}

# Define labels
eurosat_labels = list(eurosat_labels_map.values())


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


# Global per-channel normalization
def compute_eu_band_stats(df, n_bands=13):
    ''' Compute global per-channel mean and std for EuroSAT. '''
    band_means = [[] for _ in range(n_bands)]
    band_stds = [[] for _ in range(n_bands)]

    for idx, row in df.iterrows():
        img_path = row["filename"]
        try:
            with rasterio.open(img_path) as src:
                for i in range(n_bands):
                    stats = src.stats()[i]
                    band_means[i].append(stats.mean)
                    band_stds[i].append(stats.std)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    global_mean = [np.mean(m) for m in band_means]
    global_std = [np.mean(s) for s in band_stds]
    global_stats = {"mean": global_mean, "std": global_std}
    
    return global_stats


# Dataset and DataModule class
class GEOBenchMSIDataset(Dataset):
    def __init__(self, dataframe, label2idx, band_stats, selected_bands):
        self.df = dataframe.reset_index(drop=True)
        self.label2idx = label2idx
        self.band_stats = band_stats
        self.selected_bands = selected_bands  # nuova variabile

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
                    raise KeyError(f"Banda {band_name} non trovata nel file {img_path}")

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


class EuroSATMSIDataset(Dataset):
    def __init__(self, dataframe, label2idx, band_stats):
        self.df = dataframe.reset_index(drop=True) # ensure input df has continuous index
        self.label2idx = label2idx # mapping between string labels and associated integers
        self.band_stats = band_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filename"]
        label_str = self.df.loc[idx, "label"]
        label = self.label2idx[label_str]

        with rasterio.open(img_path) as src:
            img = src.read()  # shape: [C, H, W] == [13, 64, 64]
            img = img.astype(np.float32)  # necessary or the script will fail for TypeError

        # Per-band normalization based on global stats
        means = self.band_stats["mean"]
        stds = self.band_stats["std"]

        for c in range(img.shape[0]):
            img[c] = (img[c] - means[c]) / (stds[c] + 1e-6)

        img = torch.tensor(img, dtype=torch.float32)
        
        return img, label
    

class EuroSATDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, label2idx, batch_size, band_stats):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.band_stats = band_stats # train_df band stats

    def setup(self, stage=None):
        self.train_dataset = EuroSATMSIDataset(self.train_df, self.label2idx, self.band_stats)
        self.val_dataset = EuroSATMSIDataset(self.val_df, self.label2idx, self.band_stats)
        self.test_dataset = EuroSATMSIDataset(self.test_df, self.label2idx, self.band_stats)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == 'cuda')) # optimized


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda')) # optimized


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda')) # optimized


    
# A SINGLE EMBEDDER TO RULE THEM ALL - transfer learning
# Define a fixed number of input channels and put to zero the ones in excess
class MSIEmbedder3(nn.Module):
    def __init__(self, max_in_channels: int = 13): # mazimum number of allowed channels
        super(MSIEmbedder3, self).__init__()
        self.max_in_channels = max_in_channels

        # The input to the model is always expected to have max_in_channels -> the forward is different
        self.proj3 = nn.Sequential(
            nn.Conv2d(max_in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        ''' x: Tensor of shape [B, C, H, W] where C == self.max_in_channels '''
        B, C, H, W = x.shape
        if C < self.max_in_channels:
            # Pad with zeros the channel dimension to make it
            # [B, c < C, H, W] -> [B, max_in_channels, H, W]
            pad_size = self.max_in_channels - C
            padding = torch.zeros((B, pad_size, H, W), device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif C > self.max_in_channels:
            raise ValueError(f"Input has {C} channels, but max_in_channels is {self.max_in_channels}")

        return self.proj3(x)


class CLIPWithMSIEmbedder3(L.LightningModule):
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
        self.embedder = MSIEmbedder3(in_channels)

        # Compute text embeddings once for all
        prompts = [f"a satellite photo of a{'n' if name[0].lower() in 'aeiou' else ''} {name.lower()} area" for name in class_names]
        # "a ... area" are added to the standard prompt
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
    parser.add_argument('--dataset', type=str, choices=['eurosat', 'm-so2sat'], required=True, help='Training data: EuroSAT or m-so2sat')
    args = parser.parse_args()
    max_epochs = args.max_epochs
    dataset = args.dataset

    # Build TEST, VALID and TRAIN dataframes
    #images_path = data_path

    if dataset == "m-so2sat":

        data_path = '../m-so2sat/'

        # Read default partition
        with open(os.path.join(data_path, 'default_partition.json'), 'r') as file:
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

        print("Band stats for m-so2sat:")
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

        model = CLIPWithMSIEmbedder3(
            in_channels = len(selected_bands) + 1, # now we're using 12 bands, but other datasets have 13
            class_names = class_names,
            learning_rate = learning_rate,
            class_weights = weights_tensor
        )

        # 4. Specify a checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clip-msi3-transfer-so2sat-{epoch:02d}-{val_acc:.4f}", # not a format string, values will be filled at runtime
            save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
            monitor="val_acc",
            mode="max",
            save_last = True
        )

        # # 5. Specify logger in csv format
        logger = CSVLogger(save_dir=log_dir, name="clip-msi3-transfer-so2sat")

        # define the logger object
        logger_tb = TensorBoardLogger(tb_log_dir, name = "clip-msi3-transfer-so2sat", log_graph = True)

        # 6. Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            precision = 32,
            gradient_clip_val=1.0,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback],
            logger=[logger, logger_tb]
        )

        # 7. Training
        trainer.fit(model, datamodule=data_module)

        trainer.validate(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module)
        

    elif dataset == "eurosat":

        data_path = '../EuroSAT_MSI_data/'

        # TEST dataframe - 5000 images randomly sampled with seed = 42 -----------------
        print("\nTEST set: -------------------------------------------------------------")
        test_images = []

        for image_folder in os.listdir(data_path):
            random.seed(42)
            random_samples = random.sample(os.listdir(data_path + image_folder), 500)
            test_images += random_samples
        print(len(test_images), 'images\n')

        data = []
        for img_id in test_images:
            folder = img_id.split('_')[0]
            filename = os.path.join(data_path, folder, img_id)
            data.append({
                'image_id': img_id,
                'filename': filename,
                'label': eurosat_labels_map[folder]
            })

        test_df = pd.DataFrame(data)
        print(test_df.head())


        # VALIDATION dataframe - 5000 images randomly sampled with seed = 42 -----------
        print("\nVALIDATION set: -------------------------------------------------------")
        val_images = []

        for image_folder in os.listdir(data_path):
            possible_images = [img for img in os.listdir(data_path + image_folder) if img not in test_images]
            random.seed(42)
            random_samples = random.sample(possible_images, 500)
            val_images += random_samples
        print(len(val_images), 'images\n')

        data = []
        for img_id in val_images:
            folder = img_id.split('_')[0]
            filename = os.path.join(data_path, folder, img_id)
            data.append({
                'image_id': img_id,
                'filename': filename,
                'label': eurosat_labels_map[folder]
            })

        val_df = pd.DataFrame(data)
        print(val_df.head())


        # TRAIN dataframe - 10000 remaining images randomly sampled with seed = 42 -----
        print("\n\nTRAIN set: ----------------------------------------------------------")
        train_images = []

        for image_folder in os.listdir(data_path):
            possible_images = [img for img in os.listdir(data_path + image_folder) if (img not in test_images and img not in val_images)]
            random.seed(42)
            random_samples = random.sample(possible_images, 1000)
            train_images += random_samples
        print(len(train_images), 'images\n')

        data = []
        for img_id in train_images:
            folder = img_id.split('_')[0]
            filename = os.path.join(data_path, folder, img_id)
            data.append({
                'image_id': img_id,
                'filename': filename,
                'label': eurosat_labels_map[folder]
            })

        train_df = pd.DataFrame(data)
        print(train_df.head())

        
        # Compute global band stats on training dataset
        band_stats = compute_eu_band_stats(train_df)
        print(band_stats)


        # 1. Encode string labels into integers
        le = LabelEncoder()
        le.fit(train_df["label"])
        label2idx = {label: idx for idx, label in enumerate(le.classes_)}
        class_names = list(label2idx.keys())

        # 2. Load data module
        data_module = EuroSATDataModule(train_df, val_df, test_df, label2idx, batch_size, band_stats)
        
        # Add class weights (actually this dataset is balanced, so they will all be equal)
        class_counts = train_df['label'].value_counts()
        total = sum(class_counts)
        class_weights = {label2idx[label]: total / count for label, count in class_counts.items()}
        norm_factor = sum(class_weights.values()) # normalize for stability
        class_weights = {k: v / norm_factor for k, v in class_weights.items()}

        # Convert to tensor for use in model
        weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32)

        # 3. Create the model
        model = CLIPWithMSIEmbedder3(
            in_channels = 13, # input EuroSAT channels
            class_names = class_names,
            learning_rate = learning_rate,
            class_weights = weights_tensor
        )

        # 4. Specify a checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clip-msi-transfer-eurosat-{epoch:02d}-{val_acc:.4f}", # not a format string, values will be filled at runtime
            save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
            monitor="val_acc",
            mode="max",
            save_last = True
        )

        # # 5. Specify logger in csv format
        logger = CSVLogger(save_dir=log_dir, name="clip-msi-transfer-eurosat")

        # define the logger object
        logger_tb = TensorBoardLogger(tb_log_dir, name = "clip-msi-transfer-eurosat", log_graph = True)

        # 6. Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            #precision="16-mixed", # enable AMP (Automatic Mixed Precision) # -> error after first epoch
            log_every_n_steps=5,
            callbacks=[checkpoint_callback],
            logger=[logger, logger_tb]
        )

        # 7. Training
        trainer.fit(model, datamodule=data_module)

        trainer.validate(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module)

    
    else:
        raise ValueError("Unsupported model type. Choose --model 'm-so2sat' or 'eurosat'.")







if __name__ == "__main__":
    main()

