# Image Caption Generator — Usage & Overview

What it does
- Loads a trained Encoder/Decoder checkpoint and a saved vocabulary, then generates a caption for a single image using the trained model.

Brief description of how it works
- Loads a checkpoint that is expected to contain the keys `encoder`, `decoder`, and `vocab` (where `vocab` is an `itos` mapping: int -> token string).
- Reconstructs a minimal `Vocabulary` object from `vocab` and instantiates the models:
  - `EncoderCNN` (ResNet backbone → feature vector)
  - `DecoderRNN` (Embedding + LSTM + Linear)
- Preprocesses the input image with the same transforms used in training (resize → normalize → tensor).
- Performs greedy decoding starting from `<SOS>`, predicting tokens until `<EOS>` or a maximum length is reached.

Files referenced
- `predict.py` — inference script that loads checkpoint + vocab and produces a caption for a single image.
- `models/encoder.py` and `models/decoder.py` — model definitions required by `predict.py`.
- `utils/vocabulary.py` — vocabulary helper class used to reconstruct `stoi`/`itos` mappings.

Dependencies
- Python 3.8+ (project used 3.12 in development but 3.8+ should work)
- torch
- torchvision
- pillow (PIL)
- nltk (only for vocabulary/tokenization if you rebuild vocab)

Install dependencies (example):
```sh
pip install -r requirements.txt
# or at minimum
pip install torch torchvision pillow nltk
```

How to run
1. Make sure you have a checkpoint that contains at least `encoder`, `decoder`, and `vocab` (an `itos` dict). Example: `models/checkpoint_5.pth`.
2. Place or point to an image you want captioned.
3. Edit the bottom of `predict.py` (or modify the variables) to set the paths, for example:
```py
image_path = "path/to/your/image.jpg"
checkpoint_path = "models/checkpoint_5.pth"
```
4. Run the script from the project root:
```sh
python predict.py
```
