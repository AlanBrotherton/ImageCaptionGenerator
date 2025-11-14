import torch
from torchvision import transforms
from PIL import Image

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.vocabulary import Vocabulary

import os

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def predict(image_path, checkpoint_path, vocab, embed_size=256, hidden_size=512, num_layers=1, max_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load models
    vocab_size = len(vocab)
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Encode image
    with torch.no_grad():
        features = encoder(image)
        inputs = features
        caption = []
        states = None

        # Start with <SOS>
        word = torch.tensor([vocab.stoi["<SOS>"]]).unsqueeze(0).to(device)

        for _ in range(max_length):
            outputs = decoder.embed(word)
            outputs = torch.cat((inputs.unsqueeze(1), outputs), dim=1)
            hiddens, states = decoder.lstm(outputs, states)
            output = decoder.linear(hiddens[:, -1, :])
            predicted = output.argmax(1)
            predicted_word = predicted.item()

            if predicted_word == vocab.stoi["<EOS>"]:
                break
            caption.append(vocab.itos[predicted_word])
            word = predicted.unsqueeze(0)

    return " ".join(caption)

if __name__ == "__main__":
    image_path = "example.jpg"  # Change to your test image
    checkpoint_path = "models/checkpoint_5.pth"

    # Rebuild vocab using the original dataset
    from utils.dataset import FlickrDataset
    dummy_transform = transforms.Compose([transforms.ToTensor()])
    captions_file = "data/Flickr8k_text/Flickr8k.token.txt"
    root_dir = "data/Flickr8k_Dataset"

    dummy_dataset = FlickrDataset(root_dir, captions_file, transform=dummy_transform, freq_threshold=5)
    vocab = dummy_dataset.vocab

    caption = predict(image_path, checkpoint_path, vocab)
    print("Generated Caption:", caption)

