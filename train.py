import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import FlickrDataset, MyCollate
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

import os

if __name__ == "__main__":
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 5
    batch_size = 16
    freq_threshold = 5

    # Paths
    root_dir = "/Users/alanmb/Desktop/OneSpanProject/imageCaptionGenerator/data/Flickr8k_Dataset"
    captions_file = "/Users/alanmb/Desktop/OneSpanProject/imageCaptionGenerator/data/Flickr8k_text/Flickr8k.token.txt"
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    dataset = FlickrDataset(root_dir, captions_file, transform=transform, freq_threshold=freq_threshold)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    vocab_size = len(dataset.vocab)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    # Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for idx, (imgs, captions) in enumerate(loader):
            imgs, captions = imgs.to(device), captions.to(device)
            optimizer.zero_grad()

            features = encoder(imgs)
            outputs = decoder(features, captions)

            # Align outputs and targets for loss calculation
            outputs = outputs[:, 1:, :]  # Remove the first output (image feature)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}")

        # Save model checkpoints
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'vocab': dataset.vocab.itos
        }, os.path.join(save_dir, f"checkpoint_{epoch+1}.pth"))