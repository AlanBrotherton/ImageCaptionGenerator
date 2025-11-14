import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.vocabulary import Vocabulary
from utils.dataset import FlickrDataset

import os

# Prediction function
def predict(image, checkpoint_path, vocab, embed_size=256, hidden_size=512, num_layers=1, max_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    vocab_size = len(vocab)
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        features = encoder(image_tensor)
        inputs = features
        caption = []
        states = None

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

# Streamlit UI
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        # Load vocab using dataset (same as during training)
        captions_file = "data/Flickr8k_text/Flickr8k.token.txt"
        root_dir = "data/Flickr8k_Dataset"
        dummy_transform = transforms.Compose([transforms.ToTensor()])
        dummy_dataset = FlickrDataset(root_dir, captions_file, transform=dummy_transform, freq_threshold=5)
        vocab = dummy_dataset.vocab

        checkpoint_path = "models/checkpoint_5.pth"
        caption = predict(image, checkpoint_path, vocab)

    st.success("Caption generated!")
    st.markdown(f"**Caption:** {caption}")
