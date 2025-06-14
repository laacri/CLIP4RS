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


data_path = '../EuroSAT_MSI_data/' # the script will run inside the thesis/ folder
checkpoint_dir = './checkpoints/'
log_dir = './logs/'
tb_log_dir = './tb_logs/'
batch_size = 32
num_workers = 4
learning_rate = 5e-3 #1e-3

# Loading and mapping labels
# (in order to correspond to the ones in m-bigearthnet from GEO-Bench)
labels_map = {'SeaLake' : 'Sea or Lake',
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
labels = list(labels_map.values())



# Global normalization over all 13 channels
def normalize(array):
    array = array.astype(np.float32)
    return (array - array.min()) / (array.max() - array.min())


class EuroSATMSIDataset(Dataset):
    def __init__(self, dataframe, label2idx, transform=None):
        self.df = dataframe.reset_index(drop=True) # ensure input df has continuous index
        self.transform = transform
        self.label2idx = label2idx # mapping between string labels and associated integers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filename"]
        label_str = self.df.loc[idx, "label"]
        label = self.label2idx[label_str]

        with rasterio.open(img_path) as src:
            img = src.read()  # shape: [C, H, W] == [13, 64, 64]

        img = normalize(img) # GLOBAL NORMALIZATION -> all pixel values in [0, 1]
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label
    

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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == 'cuda')) # optimized


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda')) # optimized


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda')) # optimized



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
    


class CLIPWithMSIEmbedder2(L.LightningModule):
    def __init__(self, in_channels, class_names, learning_rate): # was running with 5e-3!!!
        super().__init__()
        self.save_hyperparameters()

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
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.embedder.parameters(), lr=self.learning_rate)
    



# MAIN EXECUTION --------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    max_epochs = args.max_epochs

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
            'label': labels_map[folder]
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
            'label': labels_map[folder]
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
            'label': labels_map[folder]
        })

    train_df = pd.DataFrame(data)
    print(train_df.head())


    # 1. Encode string labels into integers
    le = LabelEncoder()
    le.fit(train_df["label"])
    label2idx = {label: idx for idx, label in enumerate(le.classes_)}
    class_names = list(label2idx.keys())

    # 2. Load data module
    data_module = EuroSATDataModule(train_df, val_df, test_df, label2idx, batch_size) # batch_size = 8, 16, 32, 64(not sure)

    # 3. Create the model
    model_2 = CLIPWithMSIEmbedder2(
        in_channels = 13,
        class_names = class_names,
        learning_rate = learning_rate,
    )

    # 3.1 Compile the model - can result in significant speedups
    #model_2 = torch.compile(model_2)  # W0509: not enough SMs to use max_autotune_gemm mode -> doesn't work, turn it off

    # 4. Specify a checkpoint callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="clip-msi2-eurosat-lr5e-3-{epoch:02d}-{val_acc:.4f}", # not a format string, values will be filled at runtime
        save_top_k=1, # save only the checkpoint with the highest performance (here, val_acc)
        monitor="val_acc",
        mode="max",
        save_last = True
    )

    # Note:
    # - save_top_k=3 would keep the top 3 best-performing checkpoints
    # - save_top_k=-1 saves every epoch's checkpoint
    # - save_top_k=0 disables saving entirely

    # # 5. Specify logger in csv format
    logger = CSVLogger(save_dir=log_dir, name="clip-msi2-eurosat-lr5e-3")

    # define the logger object
    logger_tb = TensorBoardLogger(tb_log_dir, name="clip-msi2-eurosat-lr5e-3", log_graph=True)

    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        #precision="16-mixed", # enable AMP (Automatic Mixed Precision) # -> error after first epoch
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=[logger, logger_tb]
    )

    # 7. Training
    trainer.fit(model_2, datamodule=data_module)

    trainer.validate(model_2, datamodule=data_module)

    trainer.test(model_2, datamodule=data_module)



if __name__ == "__main__":
    main()

