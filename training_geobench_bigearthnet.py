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
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.classification import MultilabelAccuracy, MultilabelRankingAveragePrecision
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.preprocessing import LabelEncoder


device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore")


data_path = '../m-bigearthnet/' # the script will run inside the thesis/ folder
checkpoint_dir = './checkpoints/'
log_dir = './logs/'
tb_log_dir = './tb_logs/'
batch_size = 32
num_workers = 4
learning_rate = 1e-3 #5e-3


bigearthnet_labels = ['Agro-forestry areas',
                      'Airports',
                      'Annual crops associated with permanent crops',
                      'Bare rock',
                      'Beaches, dunes, sands',
                      'Broad-leaved forest',
                      'Burnt areas',
                      'Coastal lagoons',
                      'Complex cultivation patterns',
                      'Coniferous forest',
                      'Construction sites',
                      'Continuous urban fabric',
                      'Discontinuous urban fabric',
                      'Dump sites',
                      'Estuaries',
                      'Fruit trees and berry plantations',
                      'Green urban areas',
                      'Industrial or commercial units',
                      'Inland marshes',
                      'Intertidal flats',
                      'Land principally occupied by agriculture, with significant areas of natural vegetation',
                      'Mineral extraction sites',
                      'Mixed forest',
                      'Moors and heathland',
                      'Natural grassland',
                      'Non-irrigated arable land',
                      'Olive groves',
                      'Pastures',
                      'Peatbogs',
                      'Permanently irrigated land',
                      'Port areas',
                      'Rice fields',
                      'Road and rail networks and associated land',
                      'Salines',
                      'Salt marshes',
                      'Sclerophyllous vegetation',
                      'Sea and ocean',
                      'Sparsely vegetated areas',
                      'Sport and leisure facilities',
                      'Transitional woodland/shrub',
                      'Vineyards',
                      'Water bodies',
                      'Water courses']
n_bands = 12


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
    def __init__(self, dataframe, band_stats): #label2idx, 
        self.df = dataframe.reset_index(drop=True)
        #self.label2idx = label2idx
        self.band_stats = band_stats  # needed for normalization with train_df band stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filename"]
        #label = torch.tensor(eval(self.df.loc[idx, "ground_truth_binary_vector"], dtype=torch.float32))
        # binary_vector = ast.literal_eval(self.df.loc[idx, "ground_truth_binary_vector"])
        # label = torch.tensor(binary_vector, dtype=torch.float32)
        label = torch.tensor(self.df.loc[idx, "ground_truth_binary_vector"], dtype=torch.float32)

        with h5py.File(img_path, 'r') as f:
            bands = []
            for i, band_name in enumerate(f.keys()):
                band = f[band_name][:]
                stats = self.band_stats[band_name]
                band = (band - stats["mean"]) / (stats["std"] + 1e-6)
                bands.append(band)

        img = np.stack(bands, axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        return img, label


class GEOBenchMSIDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, band_stats, batch_size):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        #self.label2idx = label2idx
        self.band_stats = band_stats # train_df band stats
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = GEOBenchMSIDataset(self.train_df, self.band_stats) #self.label2idx, 
        self.val_dataset   = GEOBenchMSIDataset(self.val_df, self.band_stats) #self.label2idx, 
        self.test_dataset  = GEOBenchMSIDataset(self.test_df, self.band_stats) #self.label2idx, 

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
    def __init__(self, in_channels, num_classes, class_names, learning_rate, pos_weight):
        super().__init__()
        # Instantiate mAP metrics for each stage
        self.num_classes = num_classes
        self.train_map = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')
        self.val_map = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')
        self.test_map = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')

        # self.train_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5)
        # self.val_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5)
        # self.test_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5)

        # self.train_lrap = MultilabelRankingAveragePrecision(num_labels=num_classes)
        # self.val_lrap = MultilabelRankingAveragePrecision(num_labels=num_classes)
        # self.test_lrap = MultilabelRankingAveragePrecision(num_labels=num_classes)

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

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.train_map.update(probs, y.int())
        # self.train_acc.update(probs, y)
        # self.train_lrap.update(probs, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("logits_mean", logits.mean(), on_step=False, on_epoch=True)
        self.log("probs_std", probs.std(), on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        map_score = self.train_map.compute()
        self.log("train_mAP", map_score, on_step=False, on_epoch=True)
        # self.log("train_acc", self.train_acc.compute())
        # self.log("train_LRAP", self.train_lrap.compute())
        self.train_map.reset()
        # self.train_acc.reset()
        # self.train_lrap.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.val_map.update(probs, y.int())
        # self.val_acc.update(probs, y)
        # self.val_lrap.update(probs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        map_score = self.val_map.compute()
        self.log("val_mAP", map_score, on_step=False, on_epoch=True)
        # self.log("val_acc", self.val_acc.compute())
        # self.log("val_LRAP", self.val_lrap.compute())
        self.val_map.reset()
        # self.val_acc.reset()
        # self.val_lrap.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.test_map.update(probs, y.int())
        # self.test_acc.update(probs, y)
        # self.test_lrap.update(probs, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        map_score = self.test_map.compute()
        self.log("test_mAP", map_score, on_step=False, on_epoch=True)
        # self.log("test_acc", self.test_acc.compute())
        # self.log("test_LRAP", self.test_lrap.compute())
        self.test_map.reset()
        # self.test_acc.reset()
        # self.test_lrap.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.embedder.parameters(), lr=self.learning_rate)
    

class CLIPWithMSIEmbedder2(L.LightningModule):
    def __init__(self, in_channels, num_classes, class_names, learning_rate, pos_weight):
        super().__init__()
        # Instantiate mAP metrics for each stage
        self.num_classes = num_classes
        self.train_map = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')
        self.val_map = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')
        self.test_map = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')


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

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.train_map.update(probs, y.int())
        # self.train_acc.update(probs, y)
        # self.train_lrap.update(probs, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("logits_mean", logits.mean(), on_step=False, on_epoch=True)
        self.log("probs_std", probs.std(), on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        map_score = self.train_map.compute()
        self.log("train_mAP", map_score, on_step=False, on_epoch=True)
        # self.log("train_acc", self.train_acc.compute())
        # self.log("train_LRAP", self.train_lrap.compute())
        self.train_map.reset()
        # self.train_acc.reset()
        # self.train_lrap.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.val_map.update(probs, y.int())
        # self.val_acc.update(probs, y)
        # self.val_lrap.update(probs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        map_score = self.val_map.compute()
        self.log("val_mAP", map_score, on_step=False, on_epoch=True)
        # self.log("val_acc", self.val_acc.compute())
        # self.log("val_LRAP", self.val_lrap.compute())
        self.val_map.reset()
        # self.val_acc.reset()
        # self.val_lrap.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.test_map.update(probs, y.int())
        # self.test_acc.update(probs, y)
        # self.test_lrap.update(probs, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("logits_mean", logits.mean(), on_step=False, on_epoch=True)
        self.log("probs_std", probs.std(), on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        map_score = self.test_map.compute()
        self.log("test_mAP", map_score, on_step=False, on_epoch=True)
        # self.log("test_acc", self.test_acc.compute())
        # self.log("test_LRAP", self.test_lrap.compute())
        self.test_map.reset()
        # self.test_acc.reset()
        # self.test_lrap.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.embedder.parameters(), lr=self.learning_rate)
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model', type=int, choices=[1, 2], required=True, help='Model version: 1 or 2')
    args = parser.parse_args()
    max_epochs = args.max_epochs


    # Build TEST, VALID and TRAIN dataframes
    images_path = data_path

    # Read default partition
    with open(os.path.join(images_path, 'default_partition.json'), 'r') as file:
        def_partition = json.load(file)
        print("Splits:")
        print(list(def_partition.keys()))
        print(len(def_partition['train']), len(def_partition['valid']), len(def_partition['test']))

    # Read label map
    with open(os.path.join(images_path, 'label_stats.json'), 'r') as file:
        label_map = json.load(file)


    # Building TEST dataframe ------------------------------------------------------
    print("\nTEST set: -------------------------------------------------------------")
    data = []
    split = 'test'
    for img_id in def_partition[split]:
        vector = label_map.get(img_id, None)  # None in case any image is missing in label_map
        labels_idx = [i for i, x in enumerate(label_map[img_id]) if x == 1]
        labels = [bigearthnet_labels[i] for i in labels_idx]
        data.append({
            'image_id': img_id,
            'filename': os.path.join(images_path, img_id + '.hdf5'),
            'bands': n_bands,
            'split': split,
            'labels': labels,
            'ground_truth_binary_vector': vector
        })

    test_df = pd.DataFrame(data)
    print(test_df.head())

    # Building VALIDATION dataframe ------------------------------------------------
    print("\nVALIDATION set: -------------------------------------------------------")
    data = []
    split = 'valid'
    for img_id in def_partition[split]:
        vector = label_map.get(img_id, None)  # None in case any image is missing in label_map
        labels_idx = [i for i, x in enumerate(label_map[img_id]) if x == 1]
        labels = [bigearthnet_labels[i] for i in labels_idx]
        data.append({
            'image_id': img_id,
            'filename': os.path.join(images_path, img_id + '.hdf5'),
            'bands': n_bands,
            'split': split,
            'labels': labels,
            'ground_truth_binary_vector': vector
        })

    val_df = pd.DataFrame(data)
    print(val_df.head())

    # Building TRAIN dataframe -----------------------------------------------------
    print("\nTRAIN set: ------------------------------------------------------------")
    data = []
    split = 'train'
    for img_id in def_partition[split]:
        vector = label_map.get(img_id, None)  # None in case any image is missing in label_map
        labels_idx = [i for i, x in enumerate(label_map[img_id]) if x == 1]
        labels = [bigearthnet_labels[i] for i in labels_idx]
        data.append({
            'image_id': img_id,
            'filename': os.path.join(images_path, img_id + '.hdf5'),
            'bands': n_bands,
            'split': split,
            'labels': labels,
            'ground_truth_binary_vector': vector
        })

    train_df = pd.DataFrame(data)
    print(train_df.head())

    # Extract band stats for normalization
    bands = ['01 - Coastal aerosol',
            '02 - Blue',
            '03 - Green',
            '04 - Red',
            '05 - Vegetation Red Edge',
            '06 - Vegetation Red Edge',
            '07 - Vegetation Red Edge',
            '08 - NIR',
            '08A - Vegetation Red Edge',
            '09 - Water vapour',
            '11 - SWIR',
            '12 - SWIR']

    band_stats = compute_band_stats_up(train_df.head(5000), bands)
    stats_dict = {f"{row['Band']}": {
                    "mean": row["Mean"],
                    "std": row["Std"]}
                for _, row in band_stats.iterrows()}

    print(stats_dict)


    # Computing weights for labels based on frequency
    all_vectors = np.stack(train_df["ground_truth_binary_vector"].values)
    label_freq = all_vectors.sum(axis=0)
    label_pos_ratio = label_freq / len(train_df)

    # Print number of positive examples per class
    for idx, (label, count, ratio) in enumerate(zip(bigearthnet_labels, label_freq, label_pos_ratio)):
        print(f"{idx:2d} - {label:<60} | Positives: {int(count):5d} | Ratio: {ratio:.4f}")

    # Total number of samples
    num_samples = len(train_df)

    # Number of positives per class
    pos_counts = label_freq
    neg_counts = num_samples - pos_counts

    # Avoid divide-by-zero
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
        
    # # 1. Encode string labels into integers
    # le = LabelEncoder()
    # le.fit(train_df["label"])
    # label2idx = {label: idx for idx, label in enumerate(le.classes_)}
    # class_names = list(label2idx.keys())

    # 2. Load data module
    data_module = GEOBenchMSIDataModule(train_df, val_df, test_df, stats_dict, batch_size)

    # class_counts = train_df['label'].value_counts()
    # total = sum(class_counts)
    # class_weights = {label2idx[label]: total / count for label, count in class_counts.items()}
    # norm_factor = sum(class_weights.values()) # normalize for stability
    # class_weights = {k: v / norm_factor for k, v in class_weights.items()}
    # # {0: 0.1, 1: 0.1, 2: 0.1, ...} the dataset is balanced

    # # Convert to tensor for use in model
    # weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32)
    # tensor([1.0, 1.0, 1.0, ...]) the dataset is balanced

    # 3. Create the model -- edit for binary classification
    if args.model == 1:
        model = CLIPWithMSIEmbedder1(
            in_channels = n_bands,
            num_classes = len(bigearthnet_labels),
            class_names = bigearthnet_labels,
            learning_rate = learning_rate,
            pos_weight = pos_weight
        )

        # 4. Specify a checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clip-msi1-geobench-bigearthnet-{epoch:02d}-{val_mAP:.4f}", # not a format string, values will be filled at runtime
            save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
            monitor="val_mAP",
            mode="max",
            save_last = True
        )

        # # 5. Specify logger in csv format
        logger = CSVLogger(save_dir=log_dir, name="clip-msi1-geobench-bigearthnet")

        # define the logger object
        logger_tb = TensorBoardLogger(tb_log_dir, name = "clip-msi1-geobench-bigearthnet", log_graph = True)

    elif args.model == 2:
        model = CLIPWithMSIEmbedder2(
            in_channels = n_bands,
            num_classes = len(bigearthnet_labels),
            class_names = bigearthnet_labels,
            learning_rate = learning_rate,
            pos_weight = pos_weight
        )

        # 4. Specify a checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="clip-msi2-geobench-bigearthnet-{epoch:02d}-{val_mAP:.4f}", # not a format string, values will be filled at runtime
            save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
            monitor="val_mAP",
            mode="max",
            save_last = True
        )

        # # 5. Specify logger in csv format
        logger = CSVLogger(log_dir, name="clip-msi2-geobench-bigearthnet")

        # define the logger object
        logger_tb = TensorBoardLogger(tb_log_dir, name = "clip-msi2-geobench-bigearthnet", log_graph = True)

    else:
        raise ValueError("Unsupported model type. Choose --model 1 or 2.")


    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision = 32,
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

