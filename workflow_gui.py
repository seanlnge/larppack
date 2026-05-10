import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


ROOT_DIR = Path(__file__).resolve().parent
PRODUCTS_DIR = ROOT_DIR / "products"
SCRIPTS_DIR = ROOT_DIR / "scripts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PHOTOS_DIR = ROOT_DIR / "larppack"
ENV_PATH = ROOT_DIR / ".env"
RUN_WORKFLOW_SCRIPT = ROOT_DIR / "run_product_to_slides.py"
EMBED_SCRIPT = ROOT_DIR / "embed_larppack_clip.py"
EMBEDDINGS_JSON = ROOT_DIR / "larppack_embeddings.json"

OUTPUT_LINE_RE = re.compile(r"Done\. Generated \d+ slides in (.+)")
SCRIPT_LINE_RE = re.compile(r"Wrote generated slideshow script to:\s*(.+)")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


def ensure_dirs() -> None:
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "product"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_env_text(content: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def load_env_vars() -> dict[str, str]:
    vars_from_file = parse_env_text(read_text(ENV_PATH))
    merged = dict(vars_from_file)
    for key in ("OPENAI_API_KEY", "GOOGLE_DRIVE_API_KEY", "GOOGLE_DRIVE_ACCESS_TOKEN"):
        if key in os.environ:
            merged[key] = os.environ[key]
    return merged


def list_products() -> list[Path]:
    return sorted(PRODUCTS_DIR.glob("*.md"), key=lambda p: p.name.lower())


def list_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_output_dirs() -> list[Path]:
    return sorted([p for p in OUTPUTS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def list_photos() -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    return sorted([p for p in PHOTOS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in extensions])


def read_embedding_paths() -> set[str]:
    if not EMBEDDINGS_JSON.exists():
        return set()

    try:
        rows = json.loads(EMBEDDINGS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(rows, list):
        return set()

    paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_path = str(row.get("image_path", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        normalized = image_path or file_name
        if normalized:
            paths.add(normalized)
    return paths


def run_embed_new_photos() -> str:
    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", "python"),
        str(EMBED_SCRIPT),
        "--images-dir",
        str(PHOTOS_DIR),
        "--output-json",
        str(EMBEDDINGS_JSON),
        "--incremental",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    logs = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(logs.strip())
    return logs.strip()


def find_preview_image(output_dir: Path) -> str | None:
    for candidate in sorted(output_dir.glob("slide*_final.png")):
        return f"outputs/{output_dir.name}/{candidate.name}"
    return None


def run_workflow(product_path: Path) -> tuple[str, Path | None, Path | None]:
    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", "python"),
        str(RUN_WORKFLOW_SCRIPT),
        "--product-md",
        str(product_path.relative_to(ROOT_DIR)),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    logs = (completed.stdout or "") + "\n" + (completed.stderr or "")

    script_path: Path | None = None
    output_path: Path | None = None

    script_match = SCRIPT_LINE_RE.search(logs)
    if script_match:
        parsed = Path(script_match.group(1).strip())
        script_path = parsed if parsed.is_absolute() else (ROOT_DIR / parsed)

    output_match = OUTPUT_LINE_RE.search(logs)
    if output_match:
        parsed = Path(output_match.group(1).strip())
        output_path = parsed if parsed.is_absolute() else (ROOT_DIR / parsed)

    if completed.returncode != 0:
        raise RuntimeError(logs.strip())
    return logs.strip(), script_path, output_path


def refresh_google_access_token(env_vars: dict[str, str]) -> str:
    refresh_token = env_vars.get("GOOGLE_DRIVE_REFRESH_TOKEN", "")
    client_id = env_vars.get("GOOGLE_DRIVE_CLIENT_ID", "")
    client_secret = env_vars.get("GOOGLE_DRIVE_CLIENT_SECRET", "")
    if not (refresh_token and client_id and client_secret):
        return ""

    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("access_token", "")


def upload_output_to_google_drive(output_dir: Path) -> str:
    env_vars = load_env_vars()
    api_key = env_vars.get("GOOGLE_DRIVE_API_KEY", "")
    access_token = env_vars.get("GOOGLE_DRIVE_ACCESS_TOKEN", "")
    folder_id = env_vars.get("GOOGLE_DRIVE_FOLDER_ID", "")

    if not access_token:
        access_token = refresh_google_access_token(env_vars)

    if not api_key:
        raise RuntimeError("Missing GOOGLE_DRIVE_API_KEY in .env.")
    if not access_token:
        raise RuntimeError(
            "Missing Google auth token. Set GOOGLE_DRIVE_ACCESS_TOKEN, or provide "
            "GOOGLE_DRIVE_REFRESH_TOKEN + GOOGLE_DRIVE_CLIENT_ID + GOOGLE_DRIVE_CLIENT_SECRET."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_base = Path(temp_dir) / output_dir.name
        archive_file = shutil.make_archive(str(zip_base), "zip", root_dir=str(output_dir))
        archive_path = Path(archive_file)

        metadata: dict[str, Any] = {"name": archive_path.name}
        if folder_id:
            metadata["parents"] = [folder_id]

        multipart = {
            "metadata": ("metadata", requests.compat.json.dumps(metadata), "application/json; charset=UTF-8"),
            "file": (archive_path.name, archive_path.read_bytes(), "application/zip"),
        }

        response = requests.post(
            f"https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&key={api_key}",
            headers={"Authorization": f"Bearer {access_token}"},
            files=multipart,
            timeout=120,
        )
        response.raise_for_status()
        file_payload = response.json()
        file_id = file_payload.get("id", "")
        if not file_id:
            raise RuntimeError(f"Upload succeeded but no file id returned: {file_payload}")
        return f"https://drive.google.com/file/d/{file_id}/view"


@app.route("/")
def home() -> str:
    ensure_dirs()
    products = list_products()
    scripts = list_scripts()
    photos = list_photos()
    embedded_paths = read_embedding_paths()
    new_photo_count = sum(
        1 for photo in photos if photo.relative_to(PHOTOS_DIR).as_posix() not in embedded_paths
    )

    output_dirs = []
    for output in list_output_dirs():
        output_dirs.append(
            {
                "name": output.name,
                "path": output,
                "preview": find_preview_image(output),
                "slide_count": len(list(output.glob("slide*_final.png"))),
            }
        )

    return render_template(
        "home.html",
        products=products,
        scripts=scripts,
        output_dirs=output_dirs,
        photo_count=len(photos),
        embedded_count=len(embedded_paths),
        new_photo_count=new_photo_count,
    )


@app.route("/photos/upload", methods=["POST"])
def photos_upload() -> str:
    files = request.files.getlist("photos")
    if not files:
        flash("No photos selected.", "error")
        return redirect(url_for("home"))

    saved = 0
    for file in files:
        if not file or not file.filename:
            continue
        base_name = secure_filename(Path(file.filename).name)
        if not base_name:
            continue

        destination = PHOTOS_DIR / base_name
        stem = destination.stem
        suffix = destination.suffix
        index = 1
        while destination.exists():
            destination = PHOTOS_DIR / f"{stem}_{index}{suffix}"
            index += 1

        file.save(str(destination))
        saved += 1

    if saved == 0:
        flash("No valid photos uploaded.", "error")
    else:
        flash(f"Uploaded {saved} photo(s) to larppack.", "success")
    return redirect(url_for("home"))


@app.route("/photos/embed-new", methods=["POST"])
def photos_embed_new() -> str:
    try:
        logs = run_embed_new_photos()
        flash("Embedding completed.", "success")
        flash(logs, "info")
    except Exception as exc:
        flash(f"Embedding failed: {exc}", "error")
    return redirect(url_for("home"))


@app.route("/product/new", methods=["GET", "POST"])
def product_new() -> str:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        content = request.form.get("content", "")
        if not name:
            flash("Product name is required.", "error")
            return redirect(url_for("product_new"))

        file_name = f"{slugify(name)}.md"
        path = PRODUCTS_DIR / file_name
        write_text(path, content)
        flash(f"Created product: {file_name}", "success")
        return redirect(url_for("product_edit", file_name=file_name))
    return render_template("product_form.html", title="New Product", file_name="", content="")


@app.route("/product/<file_name>", methods=["GET", "POST"])
def product_edit(file_name: str) -> str:
    path = PRODUCTS_DIR / file_name
    if request.method == "POST":
        content = request.form.get("content", "")
        write_text(path, content)
        flash(f"Saved {file_name}", "success")
        return redirect(url_for("product_edit", file_name=file_name))

    if not path.exists():
        flash(f"Product not found: {file_name}", "error")
        return redirect(url_for("home"))

    return render_template("product_form.html", title=f"Edit Product: {file_name}", file_name=file_name, content=read_text(path))


@app.route("/product/<file_name>/delete", methods=["POST"])
def product_delete(file_name: str) -> str:
    path = PRODUCTS_DIR / file_name
    if path.exists():
        path.unlink()
        flash(f"Deleted {file_name}", "success")
    else:
        flash(f"Product not found: {file_name}", "error")
    return redirect(url_for("home"))


@app.route("/generate/<file_name>", methods=["POST"])
def generate_from_product(file_name: str) -> str:
    product_path = PRODUCTS_DIR / file_name
    if not product_path.exists():
        flash(f"Product not found: {file_name}", "error")
        return redirect(url_for("home"))
    try:
        logs, script_path, output_path = run_workflow(product_path)
        flash("Generation completed successfully.", "success")
        flash(logs, "info")
        if script_path is not None:
            flash(f"Script: {script_path.name}", "success")
        if output_path is not None and output_path.exists():
            return redirect(url_for("output_view", output_name=output_path.name))
    except Exception as exc:
        flash(f"Generation failed: {exc}", "error")
    return redirect(url_for("home"))


@app.route("/output/<output_name>")
def output_view(output_name: str) -> str:
    path = OUTPUTS_DIR / output_name
    if not path.exists() or not path.is_dir():
        flash("Output folder not found.", "error")
        return redirect(url_for("home"))

    slides = [f"outputs/{output_name}/{p.name}" for p in sorted(path.glob("slide*_final.png"))]
    manifest_path = path / "manifest.json"
    manifest = read_text(manifest_path) if manifest_path.exists() else ""
    return render_template("output_view.html", output_name=output_name, slides=slides, manifest=manifest)


@app.route("/output/<output_name>/delete", methods=["POST"])
def output_delete(output_name: str) -> str:
    path = OUTPUTS_DIR / output_name
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        flash(f"Deleted output folder: {output_name}", "success")
    else:
        flash(f"Output folder not found: {output_name}", "error")
    return redirect(url_for("home"))


@app.route("/output/<output_name>/upload-gdrive", methods=["POST"])
def output_upload_gdrive(output_name: str) -> str:
    path = OUTPUTS_DIR / output_name
    if not path.exists() or not path.is_dir():
        flash("Output folder not found.", "error")
        return redirect(url_for("home"))
    try:
        drive_link = upload_output_to_google_drive(path)
        flash(f"Uploaded to Google Drive: {drive_link}", "success")
    except Exception as exc:
        flash(f"Google Drive upload failed: {exc}", "error")
    return redirect(url_for("output_view", output_name=output_name))


@app.route("/settings/env", methods=["GET", "POST"])
def settings_env() -> str:
    if request.method == "POST":
        content = request.form.get("content", "")
        write_text(ENV_PATH, content.strip() + "\n")
        flash(".env saved. Restart the app if you changed runtime-sensitive variables.", "success")
        return redirect(url_for("settings_env"))
    return render_template("env_settings.html", content=read_text(ENV_PATH))


@app.route("/files/<path:relative_path>")
def serve_file(relative_path: str):
    safe_root = ROOT_DIR.resolve()
    target = (ROOT_DIR / relative_path).resolve()
    if safe_root not in target.parents and target != safe_root:
        flash("Invalid file path.", "error")
        return redirect(url_for("home"))
    if not target.exists() or not target.is_file():
        flash("File not found.", "error")
        return redirect(url_for("home"))
    return send_from_directory(str(target.parent), target.name)


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=5050, debug=False)
