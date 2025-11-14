# ImageCaptionGenerator

Setup
```zsh
# activate venv
source venv/bin/activate
# install dependencies
pip install -r requirements.txt
```

Train (optional)
- Edit `train.py` and set absolute paths for `root_dir` (images) and `captions_file`.
- Then run:
```zsh
python train.py
```
- Checkpoints are saved to `models/` (these are ignored by `.gitignore`).

Predict (single image)
- Ensure you have a checkpoint (example: `models/checkpoint_5.pth`).
- Edit `predict.py` to set `image_path` and `checkpoint_path`, or modify the script to accept arguments.
- Run:
```zsh
python predict.py
```
- `predict.py` expects the checkpoint to contain `encoder`, `decoder`, and `vocab` (an `itos` mapping).

Run the Streamlit app (UI)
- Use the Streamlit CLI so the app runs in your browser:
```zsh
streamlit run imageGeneratorApp.py
```
- Optional flags:
```zsh
streamlit run imageGeneratorApp.py --server.port 8502
streamlit run imageGeneratorApp.py --server.headless true
```
