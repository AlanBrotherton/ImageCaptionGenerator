# ImageCaptionGenerator — Quick README

A minimal README with only the essential instructions to run, train, and predict.

Checklist
- Activate virtual environment and install dependencies.
- Train the model (optional).
- Run `predict.py` to generate a caption for an image.
- Run the Streamlit UI with `streamlit run imageGeneratorApp.py`.

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

Notes
- Use absolute paths to avoid working-directory issues.
- If DataLoader raises multiprocessing errors on macOS/Windows, ensure `train.py` is run inside `if __name__ == "__main__":` and/or set `num_workers=0` for debugging.
- Do not commit `data/` or `models/*.pth` (they are in `.gitignore`).

If you want CLI args for `predict.py` or a small launch script for Streamlit, I can add them.

