# predict.py — Usage & Overview

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

Example (reconstructing vocab from checkpoint):
```py
checkpoint = torch.load(checkpoint_path, map_location="cpu")
vocab = Vocabulary(freq_threshold=5)
vocab.itos = checkpoint['vocab']
vocab.stoi = {v: k for k, v in vocab.itos.items()}
caption = predict(image_path, checkpoint_path, vocab)
print("Predicted Caption:", caption)
```

Notes & troubleshooting
- Checkpoint format: `predict.py` expects `checkpoint['encoder']`, `checkpoint['decoder']`, and `checkpoint['vocab']`. If your checkpoint uses different keys, update the loading code accordingly.
- Use absolute paths if you run from a different working directory.
- If GPU is available, `predict.py` will use it automatically; to force CPU, set the checkpoint load `map_location` to `'cpu'` and run on CPU-only device.
- Ensure `vocab.itos` includes `<SOS>` and `<EOS>` tokens, and rebuild `vocab.stoi` from it before running inference.
- If output is empty or contains `<UNK>` tokens often, the checkpoint/vocab may be incompatible with the current `utils/vocabulary.py` or the model was trained with a different vocabulary.

Optional improvements
- Add argparse to `predict.py` so you can run:
  ```sh
  python predict.py --image path/to.jpg --checkpoint models/checkpoint_5.pth
  ```
- Add a small visualization that displays the image with the predicted caption.

If you want, I can update `predict.py` to add `argparse` and the example visualization.
