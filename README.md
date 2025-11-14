# ImageCaptionGenerator

A small image captioning project using a ResNet encoder and an LSTM decoder. It extracts image features with a frozen ResNet‑50 backbone, projects them to a learned embedding, and decodes captions token-by-token with an embedding + LSTM + linear classifier. The vocabulary is built with an NLTK tokenizer and supports <PAD>, <SOS>, <EOS>, and <UNK> tokens.

Technical highlights
- Encoder: ResNet-50 (pretrained, backbone frozen) → linear projection + batchnorm to produce image embeddings.
- Decoder: Embedding layer → LSTM (batch_first) → Linear to vocabulary logits. Greedy decoding implemented in `predict.py`.
- Data pipeline: `FlickrDataset` loads images and captions, numericalizes tokens with `Vocabulary` and uses a custom collate function to pad captions for batching.
- Checkpoints: training saves encoder/decoder weights and `vocab.itos` for inference.

Quick start
1. Clone the repo and create/activate a Python virtual environment:

```bash
git clone https://github.com/AlanBrotherton/ImageCaptionGenerator.git
cd ImageCaptionGenerator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data
- Place images under `data/Flickr8k_Dataset/Images/` (or update paths in `train.py`).
- Make sure captions file (Flickr8k.token style) is available and the filenames match the images.

Train (example)
- Edit `train.py` to set absolute `root_dir` (image folder) and `captions_file` paths.
- Run training (recommended in a terminal):

```bash
python train.py
```

- Checkpoints are saved to `models/checkpoint_*.pth` and include `encoder`, `decoder`, `optimizer` and `vocab` (itos mapping).

Predict (single image)
- Update `predict.py` to point to `image_path` and `checkpoint_path` (e.g. `models/checkpoint_5.pth`), or modify it to accept CLI args.
- Run:

```bash
python predict.py
```

- `predict.py` reconstructs a `Vocabulary` from `checkpoint['vocab']`, loads model weights and performs greedy decoding until `<EOS>` or max length.

Run the Streamlit UI
- The repository includes `imageGeneratorApp.py`, a Streamlit-based UI for quick demoing.
- Launch with the Streamlit CLI (do not run with `python` if you want the browser UI):

```bash
streamlit run imageGeneratorApp.py
```

File structure (important files)
- `train.py` — training loop and checkpoint saving
- `predict.py` — inference script for a single image
- `imageGeneratorApp.py` — Streamlit demo UI
- `models/encoder.py` — `EncoderCNN` (ResNet-based)
- `models/decoder.py` — `DecoderRNN` (embedding + LSTM)
- `utils/dataset.py` — `FlickrDataset`, `MyCollate`
- `utils/vocabulary.py` — `Vocabulary` class (tokenize/build/numericalize)
- `models/*.pth` — (ignored by default) saved checkpoints

Notes and troubleshooting
- Use absolute paths to avoid working-directory issues when running scripts.
- If DataLoader raises multiprocessing errors on macOS/Windows, ensure `train.py` execution is guarded with `if __name__ == "__main__":` and/or set `num_workers=0` while debugging.
- If NLTK tokenizer fails the first time, run `python -c "import nltk; nltk.download('punkt')"` in your venv to download punkt.
- `.gitignore` excludes `data/`, `venv/`, and `models/*.pth` by default to avoid pushing large datasets or checkpoints.

License
- Add a license file if you want to publish this repository publicly (MIT is common for small projects).

Contact
- For improvements or changes you want automated (argparse, nicer CLI, visualization), open an issue or request and I can update the project.
