import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence

# Custom Dataset class for the Flickr8k dataset
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        # Load caption file (image_name#idx <TAB> caption)
        self.df = pd.read_csv(captions_file, delimiter="\t", names=["image", "caption"])
        self.df["image"] = self.df["image"].apply(lambda x: x.split("#")[0])  # Remove #index part

        # Filter out captions for missing images
        existing_images = set(os.listdir(root_dir))
        original_len = len(self.df)
        self.df = self.df[self.df["image"].isin(existing_images)].reset_index(drop=True)
        print(f"Filtered captions: {len(self.df)} / {original_len} rows kept.")

        self.captions = self.df["caption"]
        self.images = self.df["image"]

        self.transform = transform

        # Build vocabulary from training captions
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load image and caption at given index
        caption = self.captions[index]
        img_id = self.images[index]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        # Apply image transforms (resize, normalize, etc.)
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption to numerical format with <SOS> and <EOS> tokens
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)


# Collate function for DataLoader to batch variable-length captions
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Stack images into a batch tensor
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        # Pad captions to the same length
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions
