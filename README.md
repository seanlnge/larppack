# TikTokGen

Local toolkit for generating TikTok-style slideshow images from product markdown. Thank you codex 5.3 + cursor (too broke for 4.6/4.7 or 5.5)

## What it does

- Create slideshow scripts from product briefs (`products/*.md`) using OpenAI
- Select best background images with CLIP embeddings (`clip-ViT-B-16`)
- Render final slide images with TikTok-like styling
- Manage the workflow from a local GUI (products, outputs, `.env`, Drive upload)

## Main scripts

- `generate_tiktok_script.py`  
  Product markdown -> slideshow JSON script
- `generate_slides.py`  
  Slideshow JSON -> rendered slide PNGs
- `run_product_to_slides.py`  
  Full CLI workflow: product markdown -> script -> slide outputs
- `embed_larppack_clip.py`  
  Build / incrementally update `larppack_embeddings.json`
- `workflow_gui.py`  
  Local web UI (Flask)

## Quick start

1. Install dependencies (Python 3.11+ recommended):
   - `pip install flask requests pillow sentence-transformers openai`
2. Add `.env` with at least:
   - `OPENAI_API_KEY=...`
3. Start GUI:
   - `start_gui.bat`
4. Open:
   - [http://127.0.0.1:5050](http://127.0.0.1:5050)

## Google Drive setup

The app supports OAuth for Drive uploads.

Set in `.env`:

- `GOOGLE_DRIVE_OAUTH_CLIENT_ID=...`
- `GOOGLE_DRIVE_OAUTH_CLIENT_SECRET=...`
- optional `GOOGLE_DRIVE_REDIRECT_URI=http://127.0.0.1:5050/auth/google/callback`
- optional `GOOGLE_DRIVE_FOLDER_ID=...`

In Google Cloud OAuth client config, add:

- Redirect URI / Callback URI:  
  `http://127.0.0.1:5050/auth/google/callback`

Then in GUI:

- Home -> **Connect / Reconnect Google Drive**

## Photo embedding workflow

- Add photos via GUI (Home -> **Add New Photos**)
- Run incremental embedding via GUI (Home -> **Embed New Photos**)
- Incremental mode tracks embedded `image_path` and skips already-embedded files

## Safe-to-push notes

This repo ignores local/runtime artifacts:

- `.env`
- `outputs/`
- generated `scripts/*.json`
- `larppack_embeddings.json`
- `__pycache__/` and Python bytecode files

Before pushing:

- verify `.env` is not staged
- ensure no secrets are hardcoded in tracked files
