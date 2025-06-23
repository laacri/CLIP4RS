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
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import f1_score
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
# checkpoint_dir = './checkpoints/'
# log_dir = './logs/'
# tb_log_dir = './tb_logs/'
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




class MSIEmbedder1(nn.Module):
    def __init__(self, in_channels: int):
        super(MSIEmbedder1, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 1)

    def forward(self, x):
        return self.proj(x)


class MSIEmbedder2(nn.Module):
    def __init__(self, in_channels):
        super(MSIEmbedder2, self).__init__()
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 1),
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

        # Load CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=device, download_root=os.path.expanduser("~/.cache/clip"))

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
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("logits_mean", logits.mean(), on_step=False, on_epoch=True)
        self.log("probs_std", probs.std(), on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        map_score = self.train_map.compute()
        self.log("train_mAP", map_score, on_step=False, on_epoch=True)
        self.train_map.reset()


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.val_map.update(probs, y.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        map_score = self.val_map.compute()
        self.log("val_mAP", map_score, on_step=False, on_epoch=True)
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.test_map.update(probs, y.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        map_score = self.test_map.compute()
        self.log("test_mAP", map_score, on_step=False, on_epoch=True)
        self.test_map.reset()

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
            self.text_features = self.text_encoder(tokenized.to(device)).detach()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

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
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("logits_mean", logits.mean(), on_step=False, on_epoch=True)
        self.log("probs_std", probs.std(), on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        map_score = self.train_map.compute()
        self.log("train_mAP", map_score, on_step=False, on_epoch=True)
        self.train_map.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.val_map.update(probs, y.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        map_score = self.val_map.compute()
        self.log("val_mAP", map_score, on_step=False, on_epoch=True)
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.test_map.update(probs, y.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("logits_mean", logits.mean(), on_step=False, on_epoch=True)
        self.log("probs_std", probs.std(), on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        map_score = self.test_map.compute()
        self.log("test_mAP", map_score, on_step=False, on_epoch=True)
        self.test_map.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.embedder.parameters(), lr=self.learning_rate)



def add_top_predictions_with_batches_multilabel_MSI(input_df, model, device,
                                                    labels, output_path, band_stats, tokenizer = 'default'):

    ''' Process images from input dataframe, using the specified model and labels.
        - 'bands' is a list of strings identifying the RGB channels for the current dataset
        - 'tokenizer' needs to be specified when not using the standard CLIP
        - additional parameters coming from the MSI model '''

    # Set chunk and batch size
    chunk_size = 250
    batch_size = 8

    new_columns = [l.replace(' ', '_') + "_dotprod" for l in labels] + \
                  [l.replace(' ', '_') + "_softmax" for l in labels] + \
                  [l.replace(' ', '_') + "_sigmoid" for l in labels] + \
                  ["preds"] + ["pred_binary_vector"]

    # Process images in chunks
    for chunk_start in tqdm(range(0, input_df.shape[0], chunk_size)):
        chunk_df = input_df.iloc[
            chunk_start : chunk_start + chunk_size
        ].copy()

        # Initialize new columns for each chunk
        # for col in new_columns:
        #     chunk_df[col] = ""
        init_data = {col: [""] * len(chunk_df) for col in new_columns}
        chunk_df = pd.concat([chunk_df, pd.DataFrame(init_data)], axis=1)

        batch = []
        batch_indices = []  # stores row positions within chunk_df

        for index, filename in enumerate(chunk_df["filename"]):
            # index is relative to chunk_df
            try:
                #####EDITS: PUT THE GETITEM CODE TO BUILD THE IMAGE AND NORMALIZE IT

                f = h5py.File(filename)
                bands = []
                for i, band_name in enumerate(f.keys()):
                    band = f[band_name][:]
                    stats = band_stats[band_name]
                    band = (band - stats["mean"]) / (stats["std"] + 1e-6)
                    bands.append(band)

                img = np.stack(bands, axis=0)
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # -> shape [1, 12, H, W]

                # Compose RGB image
                # f = h5py.File(filename)
                # r = f[bands[0]][:]    # shape (120, 120)
                # g = f[bands[1]][:]    # shape (120, 120)
                # b = f[bands[2]][:]    # shape (120, 120)
                # rgb_image = np.stack([r, g, b], axis = -1) # shape (120, 120, 3)
                # true_color = normalize(rgb_image)

                ####EDITS: PUT THE CODE FROM THE MODEL TO PREPROCESS IMG AND LABELS

                #image = preprocess(Image.fromarray((true_color * 255).astype(np.uint8))).unsqueeze(0).to(device)
                img = model.embedder(img) # -> shape [1, 3, H, W]
                img = model.MSI_to_CLIP_preprocess(img) # -> shape [1, 3, 224, 224]

                # WATCH OUT: hardcoded clip tokenixer here -> change it when the model changes!
                if tokenizer == 'default': # go for standard CLIP tokenizer
                    text = clip.tokenize(['a satellite image of ' + l.lower() for l in labels]).to(device)
                # else: # go for model-specific tokenizer
                #     text = tokenizer(['a satellite image of ' + l.lower() for l in labels]).to(device)

                batch.append(img)
                batch_indices.append(index)  # store relative index

                # Process batch if full
                if len(batch) == batch_size:

                    ####EDITS: TAKE CODE FROM CLIPMSI MODEL TO ENCODE AND COMPUTE SIMILARITY

                    with torch.no_grad():
                        batch_images = torch.cat(batch, dim=0)
                        # image_features = model.encode_image(batch_images)
                        # text_features = model.encode_text(text)
                        image_features = model.image_encoder(batch_images)
                        text_features = model.text_encoder(text.to(device)).detach()

                    #image_features /= image_features.norm(dim=-1, keepdim=True)
                    #text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    for i, img_feats in enumerate(image_features):
                        similarity = img_feats @ text_features.T # normalized dot product -> cosine similarity
                        softmax = (100.0 * img_feats @ text_features.T).softmax(dim=-1)
                        sigmoid = (img_feats @ text_features.T).sigmoid()
                        for j, label in enumerate(labels):
                            chunk_df.iloc[
                                batch_indices[i], chunk_df.columns.get_loc(new_columns[j])
                            ] = similarity[j].item()
                            chunk_df.iloc[
                                batch_indices[i], chunk_df.columns.get_loc(new_columns[len(labels) + j]) # 2 + j
                            ] = softmax[j].item()
                            chunk_df.iloc[
                                batch_indices[i], chunk_df.columns.get_loc(new_columns[2 * len(labels) + j]) # 4 + j
                            ] = sigmoid[j].item()
                        # predicted labels
                        labels_idx = [idx for idx, x in enumerate(sigmoid) if x > 0.5]
                        class_labels = [labels[idx] for idx in labels_idx]
                        chunk_df.iat[
                            batch_indices[i], chunk_df.columns.get_loc(new_columns[-2])
                        ] = class_labels
                        # predicted binary vector
                        pred_binary_vector = [1 if x > 0.5 else 0 for x in sigmoid.tolist()]
                        chunk_df.iat[
                            batch_indices[i], chunk_df.columns.get_loc(new_columns[-1])
                        ] = pred_binary_vector

                    # Clear memory
                    del batch, batch_images, image_features
                    torch.cuda.empty_cache()
                    gc.collect()
                    batch = []
                    batch_indices = []

            except Exception as e:
                print(f"Error processing image {filename}: {e}")


        # Process remaining images in the batch (if any)
        if batch:
            with torch.no_grad():
                batch_images = torch.cat(batch, dim=0)
            #     image_features = model.encode_image(batch_images)
            #     text_features = model.encode_text(text)
                image_features = model.image_encoder(batch_images)
                text_features = model.text_encoder(text.to(device)).detach()

            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for i, img_feats in enumerate(image_features):
                similarity = img_feats @ text_features.T # normalized dot product -> cosine similarity
                softmax = (100.0 * img_feats @ text_features.T).softmax(dim=-1)
                sigmoid = (img_feats @ text_features.T).sigmoid()
                for j, label in enumerate(labels):
                    chunk_df.iloc[
                        batch_indices[i], chunk_df.columns.get_loc(new_columns[j])
                    ] = similarity[j].item()
                    chunk_df.iloc[
                        batch_indices[i], chunk_df.columns.get_loc(new_columns[len(labels) + j])
                    ] = softmax[j].item()
                    chunk_df.iloc[
                        batch_indices[i], chunk_df.columns.get_loc(new_columns[2 * len(labels) + j])
                    ] = sigmoid[j].item()
                # predicted labels
                labels_idx = [i for i, x in enumerate(sigmoid) if x > 0.5]
                class_labels = [labels[i] for i in labels_idx]
                chunk_df.iat[
                    batch_indices[i], chunk_df.columns.get_loc(new_columns[-2]) # 'preds' column
                ] = class_labels
                # predicted binary vector
                pred_binary_vector = [1 if x > 0.5 else 0 for x in sigmoid.tolist()]
                chunk_df.iat[
                    batch_indices[i], chunk_df.columns.get_loc(new_columns[-1]) # 'pred_binary_vector' column
                ] = pred_binary_vector

            del batch, batch_images, image_features
            torch.cuda.empty_cache()
            gc.collect()

        # Append results to CSV
        chunk_df.to_csv(
            f"{output_path}",
            mode = "a",
            index = False,
            header = (chunk_start == 0),
        )

        del chunk_df
        gc.collect()

    return


def unpack_binary_vector(str_vec):
    return [int(x.strip("' ")) for x in str_vec.strip("[]").split(",")]

def prepare_predictions_and_targets(df, columns):
    gt_vectors = list(map(unpack_binary_vector, df['ground_truth_binary_vector'].tolist()))
    y_true = np.array(gt_vectors)
    y_scores = np.array([
        [row[col] for col in columns]
        for _, row in df.iterrows()
    ])
    return y_true, y_scores

def compute_mAP(df):
    similarity_cols = [c for c in df.columns if 'dotprod' in c]
    y_true, y_scores = prepare_predictions_and_targets(df, similarity_cols)
    mAP_score = average_precision_score(y_true, y_scores, average = 'samples')
    return mAP_score


def compute_recall_at_k(df, k):

    ''' Function to iterate a df's wors and compute recall@k. '''

    recalls = []

    for _, row in df.iterrows():

        # Extract indices of true labels
        true_indices = row.ground_truth_binary_vector.strip('[]').split(',')
        true_indices = [int(label.strip("' ")) for label in true_indices]
        true_indices = np.nonzero(true_indices)[0]

        # Extract indices of top k predictions (ranked by similarity == dotprod)
        preds = np.array([row[col] for col in df.columns if col.endswith('_dotprod')])
        topk_indices = preds.argsort()[::-1][:k]

        # Computing recall
        num_correct = len(set(true_indices) & set(topk_indices))
        recall = num_correct / len(true_indices) if len(true_indices) > 0 else 0

        recalls.append(recall)

    return np.mean(recalls)


def compute_precision_at_k(df, k):

    ''' Function to iterate a df's wors and compute precision@k. '''

    precisions = []

    for _, row in df.iterrows():

        # Extract indices of true labels
        true_indices = row.ground_truth_binary_vector.strip('[]').split(',')
        true_indices = [int(label.strip("' ")) for label in true_indices]
        true_indices = np.nonzero(true_indices)[0]

        # Extract indices of top k predictions (ranked by similarity == dotprod)
        preds = np.array([row[col] for col in df.columns if col.endswith('_dotprod')])
        topk_indices = preds.argsort()[::-1][:k]

        # Computing precision
        num_correct = len(set(true_indices) & set(topk_indices))
        precision = num_correct / k if k > 0 else 0

        precisions.append(precision)

    return np.mean(precisions)


def compute_ranking_loss(df):

    gt_vectors = list(map(unpack_binary_vector, df['ground_truth_binary_vector'].tolist()))
    y_true = np.array(gt_vectors)
    similarity_cols = [c for c in df.columns if 'dotprod' in c]
    y_scores = df[similarity_cols].to_numpy()

    return label_ranking_loss(y_true, y_scores)


# Computing micro/macro F1 - sigmoid scores with adaptive thresholds
def compute_f1_scores_adaptive(df):
    gt_vectors = list(map(unpack_binary_vector, df['ground_truth_binary_vector'].tolist()))
    y_true = np.array(gt_vectors)
    similarity_cols = [c for c in df.columns if 'dotprod' in c]
    y_scores = df[similarity_cols].to_numpy()
    thresholds = np.median(y_scores, axis = 0)  # or np.mean (slightly different from the 5th decimal digit)
    y_pred = (y_scores >= thresholds).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    return macro_f1, micro_f1




def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    # parser.add_argument('--model', type=int, choices=[1, 2], required=True, help='Model version: 1 or 2')
    # args = parser.parse_args()
    # max_epochs = args.max_epochs

    # Build TEST dataframe
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


    # Computing band stats (take from file)
    stats_dict = {'01 - Coastal aerosol': {'mean': 381.99246068055555, 'std': 447.3006618101728}, '02 - Blue': {'mean': 486.2384950138889, 'std': 507.6022595613056}, '03 - Green': {'mean': 710.6073047916667, 'std': 546.4906943594993}, '04 - Red': {'mean': 724.8705855416666, 'std': 679.2041801606863}, '05 - Vegetation Red Edge': {'mean': 1096.7521818055557, 'std': 690.954796365387}, '06 - Vegetation Red Edge': {'mean': 1878.8457811111111, 'std': 993.2168811837848}, '07 - Vegetation Red Edge': {'mean': 2153.6003776805555, 'std': 1156.159482697204}, '08 - NIR': {'mean': 2295.249383625, 'std': 1261.722136112209}, '08A - Vegetation Red Edge': {'mean': 2351.9794912083335, 'std': 1240.7231463855621}, '09 - Water vapour': {'mean': 2322.9576192916666, 'std': 1186.8312492923706}, '11 - SWIR': {'mean': 1850.3102373194445, 'std': 1105.6170464498296}, '12 - SWIR': {'mean': 1216.491185861111, 'std': 870.4542402015679}}

    # Computing weights for labels based on frequency
    all_vectors = np.stack(train_df["ground_truth_binary_vector"].values)
    label_freq = all_vectors.sum(axis=0)
    label_pos_ratio = label_freq / len(train_df)

    print()
    print()

    for idx, (label, count, ratio) in enumerate(zip(bigearthnet_labels, label_freq, label_pos_ratio)):
        print(f"{idx:2d} - {label:<60} | Positives: {int(count):5d} | Ratio: {ratio:.4f}")

    print()
    print()

    num_samples = len(train_df)

    pos_counts = label_freq
    neg_counts = num_samples - pos_counts

    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)


    # MODEL 1 EVALUATION
    print("\n### Evaluating CLIP-MSI1 on m-bigearthnet ###\n") # --------------------------------------
    
    ckpt_path = "./checkpoints/clip-msi1-geobench-bigearthnet-last.ckpt"

    model = CLIPWithMSIEmbedder1.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        in_channels = n_bands,
        num_classes = len(bigearthnet_labels),
        class_names = bigearthnet_labels,
        learning_rate = learning_rate,
        pos_weight = pos_weight
    )

    model.eval()
    model.to(device)

    add_top_predictions_with_batches_multilabel_MSI(test_df,
                                                    model,
                                                    device,
                                                    bigearthnet_labels,
                                                    './output/m-bigearthnet_predictions_CLIPwithMSI1.csv',
                                                    stats_dict,
                                                    tokenizer = 'default')
    
    output_df = pd.read_csv('./output/m-bigearthnet_predictions_CLIPwithMSI1.csv')
    print("Saved file shape:", output_df.shape)
    output_df = output_df.dropna().reset_index(drop=True)
    print("After dropna:", output_df.shape)

    print(f"CLIP-MSI1 Mean Average Precision (mAP): {compute_mAP(output_df):.4f}")
    print(f"Recall@3: {compute_recall_at_k(output_df, k=3):.4f}")
    print(f"Recall@5: {compute_recall_at_k(output_df, k=5):.4f}")
    print(f"Recall@10: {compute_recall_at_k(output_df, k=10):.4f}")
    print(f"Precision@3: {compute_precision_at_k(output_df, k=3):.4f}")
    print(f"Precision@5: {compute_precision_at_k(output_df, k=5):.4f}")
    print(f"Precision@10: {compute_precision_at_k(output_df, k=10):.4f}")
    print(f"Ranking Loss: {compute_ranking_loss(output_df):.4f}")
    macro_f1, micro_f1 = compute_f1_scores_adaptive(output_df)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")


    # MODEL 2 EVALUATION
    print("\n### Evaluating CLIP-MSI2 on m-bigearthnet ###\n") # --------------------------------------
    
    ckpt_path = "./checkpoints/clip-msi2-geobench-bigearthnet-last.ckpt"

    model = CLIPWithMSIEmbedder2.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        in_channels = n_bands,
        num_classes = len(bigearthnet_labels),
        class_names = bigearthnet_labels,
        learning_rate = learning_rate,
        pos_weight = pos_weight
    )

    model.eval()
    model.to(device)

    add_top_predictions_with_batches_multilabel_MSI(test_df,
                                                    model,
                                                    device,
                                                    bigearthnet_labels,
                                                    './output/m-bigearthnet_predictions_CLIPwithMSI2.csv',
                                                    stats_dict,
                                                    tokenizer = 'default')
    
    output_df = pd.read_csv('./output/m-bigearthnet_predictions_CLIPwithMSI2.csv')
    print("Saved file shape:", output_df.shape)
    output_df = output_df.dropna().reset_index(drop=True)
    print("After dropna:", output_df.shape)

    print(f"CLIP-MSI2 Mean Average Precision (mAP): {compute_mAP(output_df):.4f}")
    print(f"Recall@3: {compute_recall_at_k(output_df, k=3):.4f}")
    print(f"Recall@5: {compute_recall_at_k(output_df, k=5):.4f}")
    print(f"Recall@10: {compute_recall_at_k(output_df, k=10):.4f}")
    print(f"Precision@3: {compute_precision_at_k(output_df, k=3):.4f}")
    print(f"Precision@5: {compute_precision_at_k(output_df, k=5):.4f}")
    print(f"Precision@10: {compute_precision_at_k(output_df, k=10):.4f}")
    print(f"Ranking Loss: {compute_ranking_loss(output_df):.4f}")
    macro_f1, micro_f1 = compute_f1_scores_adaptive(output_df)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")


    # MODEL TRANSFER SO2SAT EVALUATION
    print("\n### Evaluating CLIP-MSITs on m-bigearthnet ###\n") # --------------------------------------
    
    ckpt_path = "/content/drive/MyDrive/Thesis CLIP4EO/checkpoints/clip-msi-transfer-so2sat-epoch=42-val_acc=0.2110.ckpt"

    model = CLIPWithMSIEmbedder3.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        in_channels = 13, #fixed for CLIPMSI3
        class_names = bigearthnet_labels,
        learning_rate = learning_rate,
        class_weights = pos_weight
    )


    model.eval()
    model.to(device)

    add_top_predictions_with_batches_multilabel_MSI(test_df,
                                                    model,
                                                    device,
                                                    bigearthnet_labels,
                                                    './output/m-bigearthnet_predictions_CLIPwithMSITs.csv',
                                                    stats_dict,
                                                    tokenizer = 'default')
    
    output_df = pd.read_csv('./output/m-bigearthnet_predictions_CLIPwithMSITs.csv')
    print("Saved file shape:", output_df.shape)
    output_df = output_df.dropna().reset_index(drop=True)
    print("After dropna:", output_df.shape)

    print(f"CLIP-MSI2 Mean Average Precision (mAP): {compute_mAP(output_df):.4f}")
    print(f"Recall@3: {compute_recall_at_k(output_df, k=3):.4f}")
    print(f"Recall@5: {compute_recall_at_k(output_df, k=5):.4f}")
    print(f"Recall@10: {compute_recall_at_k(output_df, k=10):.4f}")
    print(f"Precision@3: {compute_precision_at_k(output_df, k=3):.4f}")
    print(f"Precision@5: {compute_precision_at_k(output_df, k=5):.4f}")
    print(f"Precision@10: {compute_precision_at_k(output_df, k=10):.4f}")
    print(f"Ranking Loss: {compute_ranking_loss(output_df):.4f}")
    macro_f1, micro_f1 = compute_f1_scores_adaptive(output_df)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")


     # MODEL TRANSFER SO2SAT EVALUATION
    print("\n### Evaluating CLIP-MSITE on m-bigearthnet ###\n") # --------------------------------------
    
    ckpt_path = "/content/drive/MyDrive/Thesis CLIP4EO/checkpoints/clip-msi-transfer-eurosat-epoch=89-val_acc=0.8254.ckpt"

    model = CLIPWithMSIEmbedder3.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        in_channels = 13, #fixed for CLIPMSI3
        class_names = bigearthnet_labels,
        learning_rate = learning_rate,
        class_weights = pos_weight
    )


    model.eval()
    model.to(device)

    add_top_predictions_with_batches_multilabel_MSI(test_df,
                                                    model,
                                                    device,
                                                    bigearthnet_labels,
                                                    './output/m-bigearthnet_predictions_CLIPwithMSITE.csv',
                                                    stats_dict,
                                                    tokenizer = 'default')
    
    output_df = pd.read_csv('./output/m-bigearthnet_predictions_CLIPwithMSITE.csv')
    print("Saved file shape:", output_df.shape)
    output_df = output_df.dropna().reset_index(drop=True)
    print("After dropna:", output_df.shape)

    print(f"CLIP-MSI2 Mean Average Precision (mAP): {compute_mAP(output_df):.4f}")
    print(f"Recall@3: {compute_recall_at_k(output_df, k=3):.4f}")
    print(f"Recall@5: {compute_recall_at_k(output_df, k=5):.4f}")
    print(f"Recall@10: {compute_recall_at_k(output_df, k=10):.4f}")
    print(f"Precision@3: {compute_precision_at_k(output_df, k=3):.4f}")
    print(f"Precision@5: {compute_precision_at_k(output_df, k=5):.4f}")
    print(f"Precision@10: {compute_precision_at_k(output_df, k=10):.4f}")
    print(f"Ranking Loss: {compute_ranking_loss(output_df):.4f}")
    macro_f1, micro_f1 = compute_f1_scores_adaptive(output_df)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")

    return



if __name__ == "__main__":
    main()











