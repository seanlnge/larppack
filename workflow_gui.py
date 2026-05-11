import json
import mimetypes
import os
import secrets
import re
import shutil
import subprocess
from urllib.parse import urlencode
from pathlib import Path
from typing import Any
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
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
GOOGLE_OAUTH_SCOPE = "https://www.googleapis.com/auth/drive.file"
GOOGLE_AUTH_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
SETTINGS_FIELDS = [
    ("OPENAI_API_KEY", "OpenAI API Key", True),
    ("GOOGLE_DRIVE_API_KEY", "Google Drive API Key (optional)", True),
    ("GOOGLE_DRIVE_OAUTH_CLIENT_ID", "Google Drive OAuth Client ID", True),
    ("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET", "Google Drive OAuth Client Secret", True),
    ("GOOGLE_DRIVE_ACCESS_TOKEN", "Google Drive Access Token", True),
    ("GOOGLE_DRIVE_REFRESH_TOKEN", "Google Drive Refresh Token", True),
    ("GOOGLE_DRIVE_REDIRECT_URI", "Google Drive Redirect URI", False),
    ("GOOGLE_DRIVE_FOLDER_ID", "Google Drive Folder ID (optional)", False),
    ("GOOGLE_DRIVE_FOLDER_NAME", "Google Drive Folder Name (optional)", False),
]

OUTPUT_LINE_RE = re.compile(r"Done\. Generated \d+ slides in (.+)")
SCRIPT_LINE_RE = re.compile(r"Wrote generated slideshow script to:\s*(.+)")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()


def ensure_dirs() -> None:
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def is_async_job_request() -> bool:
    return request.headers.get("X-Async-Job") == "1"


def create_job(title: str) -> str:
    job_id = secrets.token_urlsafe(12)
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": title,
            "status": "queued",
            "progress": 0,
            "logs": [],
            "error": "",
            "result": {},
        }
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(fields)


def append_job_log(job_id: str, message: str) -> None:
    clean = message.rstrip()
    if not clean:
        return
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["logs"].append(clean)
        JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-400:]


def run_background_job(job_id: str, runner) -> None:
    def _worker() -> None:
        update_job(job_id, status="running")
        try:
            result = runner()
            update_job(job_id, status="success", progress=100, result=result or {})
        except Exception as exc:
            append_job_log(job_id, f"ERROR: {exc}")
            update_job(job_id, status="error", error=str(exc), progress=100)

    Thread(target=_worker, daemon=True).start()


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


def upsert_env_vars(path: Path, updates: dict[str, str]) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    key_to_idx: dict[str, int] = {}

    for idx, line in enumerate(existing_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            key_to_idx[key] = idx

    for key, value in updates.items():
        safe_val = value.replace("\n", " ").strip()
        rendered = f'{key}="{safe_val}"'
        if key in key_to_idx:
            existing_lines[key_to_idx[key]] = rendered
        else:
            existing_lines.append(rendered)

    path.write_text("\n".join(existing_lines).strip() + "\n", encoding="utf-8")


def load_env_vars() -> dict[str, str]:
    vars_from_file = parse_env_text(read_text(ENV_PATH))
    merged = dict(vars_from_file)
    for key in (
        "OPENAI_API_KEY",
        "GOOGLE_DRIVE_API_KEY",
        "GOOGLE_DRIVE_ACCESS_TOKEN",
        "GOOGLE_DRIVE_REFRESH_TOKEN",
        "GOOGLE_DRIVE_OAUTH_CLIENT_ID",
        "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET",
        "GOOGLE_DRIVE_REDIRECT_URI",
    ):
        if key in os.environ:
            merged[key] = os.environ[key]
    return merged


def get_drive_connection_status() -> dict[str, Any]:
    env_vars = load_env_vars()
    has_client_id = bool(env_vars.get("GOOGLE_DRIVE_OAUTH_CLIENT_ID") or env_vars.get("GOOGLE_DRIVE_CLIENT_ID"))
    has_client_secret = bool(env_vars.get("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET") or env_vars.get("GOOGLE_DRIVE_CLIENT_SECRET"))
    has_refresh_token = bool(env_vars.get("GOOGLE_DRIVE_REFRESH_TOKEN"))
    has_access_token = bool(env_vars.get("GOOGLE_DRIVE_ACCESS_TOKEN"))
    connected = has_access_token or (has_client_id and has_client_secret and has_refresh_token)
    return {
        "connected": connected,
        "has_client_id": has_client_id,
        "has_client_secret": has_client_secret,
        "has_refresh_token": has_refresh_token,
        "has_access_token": has_access_token,
        "redirect_uri": env_vars.get("GOOGLE_DRIVE_REDIRECT_URI", "http://127.0.0.1:5050/auth/google/callback"),
    }


def get_google_oauth_config() -> tuple[str, str, str]:
    env_vars = load_env_vars()
    client_id = env_vars.get("GOOGLE_DRIVE_OAUTH_CLIENT_ID") or env_vars.get("GOOGLE_DRIVE_CLIENT_ID", "")
    client_secret = env_vars.get("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET") or env_vars.get("GOOGLE_DRIVE_CLIENT_SECRET", "")
    redirect_uri = env_vars.get("GOOGLE_DRIVE_REDIRECT_URI", "http://127.0.0.1:5050/auth/google/callback")
    return client_id.strip(), client_secret.strip(), redirect_uri.strip()


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


def run_command_stream(command: list[str], cwd: Path, on_line=None) -> str:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        lines.append(line)
        if on_line is not None:
            on_line(line)
    return_code = process.wait()
    logs = "\n".join(lines)
    if return_code != 0:
        raise RuntimeError(logs.strip())
    return logs


def run_embed_new_photos(on_line=None) -> str:
    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", "python"),
        str(EMBED_SCRIPT),
        "--images-dir",
        str(PHOTOS_DIR),
        "--output-json",
        str(EMBEDDINGS_JSON),
        "--incremental",
    ]
    logs = run_command_stream(cmd, ROOT_DIR, on_line=on_line)
    return logs.strip()


def find_preview_image(output_dir: Path) -> str | None:
    for candidate in sorted(output_dir.glob("slide*_final.png")):
        return f"outputs/{output_dir.name}/{candidate.name}"
    return None


def run_workflow(product_path: Path, on_line=None) -> tuple[str, Path | None, Path | None]:
    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", "python"),
        str(RUN_WORKFLOW_SCRIPT),
        "--product-md",
        str(product_path.relative_to(ROOT_DIR)),
    ]
    logs = run_command_stream(cmd, ROOT_DIR, on_line=on_line)

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

    return logs.strip(), script_path, output_path


def refresh_google_access_token(env_vars: dict[str, str]) -> str:
    refresh_token = env_vars.get("GOOGLE_DRIVE_REFRESH_TOKEN", "")
    client_id = env_vars.get("GOOGLE_DRIVE_OAUTH_CLIENT_ID") or env_vars.get("GOOGLE_DRIVE_CLIENT_ID", "")
    client_secret = env_vars.get("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET") or env_vars.get("GOOGLE_DRIVE_CLIENT_SECRET", "")
    if not (refresh_token and client_id and client_secret):
        return ""

    response = requests.post(
        GOOGLE_TOKEN_URL,
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


def is_google_auth_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "401" in lowered
        or "unauthorized" in lowered
        or "unauthenticated" in lowered
        or "invalid credentials" in lowered
        or "autherror" in lowered
    )


def upload_output_to_google_drive(output_dir: Path, on_progress=None) -> str:
    env_vars = load_env_vars()
    access_token = env_vars.get("GOOGLE_DRIVE_ACCESS_TOKEN", "").strip()
    folder_id = env_vars.get("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    folder_name = env_vars.get("GOOGLE_DRIVE_FOLDER_NAME", "").strip()
    slide_files = sorted(output_dir.glob("slide*_final.png"))
    if not slide_files:
        raise RuntimeError(f"No slide images found in output folder: {output_dir.name}")

    if not access_token:
        access_token = refresh_google_access_token(env_vars)
        if access_token:
            upsert_env_vars(ENV_PATH, {"GOOGLE_DRIVE_ACCESS_TOKEN": access_token})

    if not access_token:
        raise RuntimeError(
            "Missing Google auth token. Set GOOGLE_DRIVE_ACCESS_TOKEN, or provide "
            "GOOGLE_DRIVE_REFRESH_TOKEN + GOOGLE_DRIVE_OAUTH_CLIENT_ID + GOOGLE_DRIVE_OAUTH_CLIENT_SECRET."
        )

    def _upload_with_token(token: str) -> str:
        resolved_folder_id = resolve_or_create_folder_id(
            access_token=token,
            folder_id=folder_id,
            folder_name=folder_name,
        )

        if resolved_folder_id and resolved_folder_id != folder_id:
            upsert_env_vars(ENV_PATH, {"GOOGLE_DRIVE_FOLDER_ID": resolved_folder_id})

        destination_folder_id = create_drive_folder(
            access_token=token,
            folder_name=output_dir.name,
            parent_folder_id=resolved_folder_id,
        )
        if on_progress is not None:
            on_progress(5, f"Created destination folder: {output_dir.name}")

        total_files = len(slide_files)

        def _upload_one(path: Path) -> str:
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            upload_drive_file(
                access_token=token,
                file_name=path.name,
                file_bytes=path.read_bytes(),
                mime_type=mime_type,
                parent_folder_id=destination_folder_id,
            )
            return path.name

        uploaded_count = 0
        worker_count = min(8, total_files)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_upload_one, slide_file): slide_file.name for slide_file in slide_files}
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed uploading {file_name}: {exc}") from exc
                uploaded_count += 1
                if on_progress is not None:
                    progress = 5 + int((uploaded_count / total_files) * 95)
                    on_progress(progress, f"Uploaded {uploaded_count}/{total_files}: {file_name}")

        return f"https://drive.google.com/drive/folders/{destination_folder_id} (uploaded {uploaded_count} images)"

    try:
        return _upload_with_token(access_token)
    except Exception as first_exc:
        if not is_google_auth_error(str(first_exc)):
            raise

        refreshed = refresh_google_access_token(env_vars)
        if not refreshed:
            raise RuntimeError(
                "Google authentication failed and token refresh is unavailable. "
                "Reconnect Google Drive from Settings."
            ) from first_exc

        upsert_env_vars(ENV_PATH, {"GOOGLE_DRIVE_ACCESS_TOKEN": refreshed})
        if on_progress is not None:
            on_progress(2, "Access token refreshed, retrying upload...")
        return _upload_with_token(refreshed)


def upload_drive_file(
    access_token: str,
    file_name: str,
    file_bytes: bytes,
    mime_type: str,
    parent_folder_id: str,
) -> str:
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    metadata: dict[str, Any] = {"name": file_name}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    create_resp = requests.post(
        "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true&fields=id",
        headers=auth_headers,
        json=metadata,
        timeout=60,
    )
    if create_resp.status_code >= 400:
        raise RuntimeError(
            "Google Drive file create failed. "
            f"{create_resp.status_code} {create_resp.reason}. Response: {create_resp.text[:1000]}"
        )

    file_id = create_resp.json().get("id", "")
    if not file_id:
        raise RuntimeError(f"Google Drive file create returned no id: {create_resp.text[:1000]}")

    media_resp = requests.patch(
        f"https://www.googleapis.com/upload/drive/v3/files/{file_id}?uploadType=media&supportsAllDrives=true",
        headers={**auth_headers, "Content-Type": mime_type},
        data=file_bytes,
        timeout=180,
    )
    if media_resp.status_code >= 400:
        raise RuntimeError(
            "Google Drive media upload failed. "
            f"{media_resp.status_code} {media_resp.reason}. Response: {media_resp.text[:1000]}"
        )
    return file_id


def create_drive_folder(access_token: str, folder_name: str, parent_folder_id: str = "") -> str:
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    metadata: dict[str, Any] = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    create = requests.post(
        "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true&fields=id,name",
        headers=auth_headers,
        json=metadata,
        timeout=30,
    )
    if create.status_code >= 400:
        raise RuntimeError(
            "Failed creating Drive folder. "
            f"{create.status_code} {create.reason}. Response: {create.text[:800]}"
        )
    new_id = str(create.json().get("id", "")).strip()
    if not new_id:
        raise RuntimeError(f"Drive folder creation returned no id: {create.text[:800]}")
    return new_id


def resolve_or_create_folder_id(access_token: str, folder_id: str, folder_name: str) -> str:
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    effective_name = folder_name

    # If folder_id exists, verify it points to an accessible folder.
    if folder_id:
        verify = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{folder_id}?fields=id,mimeType,name&supportsAllDrives=true",
            headers=auth_headers,
            timeout=30,
        )
        if verify.status_code < 400:
            mime = verify.json().get("mimeType", "")
            if mime != "application/vnd.google-apps.folder":
                raise RuntimeError(f"GOOGLE_DRIVE_FOLDER_ID is not a folder: {folder_id}")
            return folder_id

        # Common case: user put folder name into GOOGLE_DRIVE_FOLDER_ID
        if verify.status_code == 404 and not effective_name:
            effective_name = folder_id
        elif verify.status_code != 404:
            raise RuntimeError(
                "Could not verify GOOGLE_DRIVE_FOLDER_ID. "
                f"{verify.status_code} {verify.reason}. Response: {verify.text[:800]}"
            )

    if not effective_name:
        return ""

    # Try finding existing folder by name.
    q_name = effective_name.replace("'", "\\'")
    search = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers=auth_headers,
        params={
            "q": f"mimeType='application/vnd.google-apps.folder' and name='{q_name}' and trashed=false",
            "fields": "files(id,name)",
            "spaces": "drive",
            "pageSize": 10,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        },
        timeout=30,
    )
    if search.status_code >= 400:
        raise RuntimeError(
            "Failed searching Drive folder by name. "
            f"{search.status_code} {search.reason}. Response: {search.text[:800]}"
        )
    files = search.json().get("files", [])
    if files:
        return str(files[0].get("id", "")).strip()

    # Create folder if it does not exist.
    create = requests.post(
        "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true&fields=id,name",
        headers=auth_headers,
        json={"name": effective_name, "mimeType": "application/vnd.google-apps.folder"},
        timeout=30,
    )
    if create.status_code >= 400:
        raise RuntimeError(
            "Failed creating Drive folder. "
            f"{create.status_code} {create.reason}. Response: {create.text[:800]}"
        )
    new_id = str(create.json().get("id", "")).strip()
    if not new_id:
        raise RuntimeError(f"Drive folder creation returned no id: {create.text[:800]}")
    return new_id


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
        drive_status=get_drive_connection_status(),
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
    if is_async_job_request():
        job_id = create_job("Embed New Photos")
        home_redirect_url = url_for("home")

        photos = list_photos()
        embedded_paths = read_embedding_paths()
        total_new = sum(1 for photo in photos if photo.relative_to(PHOTOS_DIR).as_posix() not in embedded_paths)

        def _runner() -> dict[str, Any]:
            update_job(job_id, progress=3)
            done = 0

            def _on_line(line: str) -> None:
                nonlocal done
                append_job_log(job_id, line)
                if line.startswith("Embedded "):
                    done += 1
                    if total_new > 0:
                        progress = 5 + int((done / total_new) * 90)
                        update_job(job_id, progress=min(progress, 95))

            logs = run_embed_new_photos(on_line=_on_line)
            append_job_log(job_id, logs)
            update_job(job_id, progress=100)
            return {"redirect_url": home_redirect_url}

        run_background_job(job_id, _runner)
        return jsonify({"job_id": job_id})

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

    if is_async_job_request():
        job_id = create_job(f"Generate Output: {file_name}")
        home_redirect_url = url_for("home")

        def _runner() -> dict[str, Any]:
            current_progress = 5
            update_job(job_id, progress=current_progress)
            rendered_count = 0

            def _on_line(line: str) -> None:
                nonlocal rendered_count, current_progress
                append_job_log(job_id, line)
                low = line.lower()
                if "step 1/2" in low:
                    current_progress = max(current_progress, 10)
                    update_job(job_id, progress=current_progress)
                elif "wrote generated slideshow script to:" in low:
                    current_progress = max(current_progress, 40)
                    update_job(job_id, progress=current_progress)
                elif "step 2/2" in low:
                    current_progress = max(current_progress, 50)
                    update_job(job_id, progress=current_progress)
                elif "rendered slide" in low:
                    rendered_count += 1
                    current_progress = min(95, 50 + rendered_count * 6)
                    update_job(job_id, progress=current_progress)

            logs, script_path, output_path = run_workflow(product_path, on_line=_on_line)
            append_job_log(job_id, logs)

            result: dict[str, Any] = {"redirect_url": home_redirect_url}
            if output_path is not None and output_path.exists():
                result["redirect_url"] = f"/output/{output_path.name}"
            if script_path is not None:
                result["script_name"] = script_path.name
            return result

        run_background_job(job_id, _runner)
        return jsonify({"job_id": job_id})

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

    if is_async_job_request():
        job_id = create_job(f"Upload to Drive: {output_name}")
        output_redirect_url = url_for("output_view", output_name=output_name)

        def _runner() -> dict[str, Any]:
            update_job(job_id, progress=3)

            def _on_progress(progress: int, message: str) -> None:
                update_job(job_id, progress=max(3, min(99, progress)))
                append_job_log(job_id, message)

            drive_link = upload_output_to_google_drive(path, on_progress=_on_progress)
            append_job_log(job_id, f"Uploaded successfully: {drive_link}")
            return {"redirect_url": output_redirect_url, "drive_link": drive_link}

        run_background_job(job_id, _runner)
        return jsonify({"job_id": job_id})

    try:
        drive_link = upload_output_to_google_drive(path)
        flash(f"Uploaded to Google Drive: {drive_link}", "success")
    except Exception as exc:
        flash(f"Google Drive upload failed: {exc}", "error")
    return redirect(url_for("output_view", output_name=output_name))


@app.route("/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/settings/env", methods=["GET", "POST"])
def settings_env() -> str:
    env_vars = load_env_vars()
    if request.method == "POST":
        updates: dict[str, str] = {}
        for key, _, _ in SETTINGS_FIELDS:
            value = request.form.get(key, "").strip()
            if value:
                updates[key] = value

        if updates:
            upsert_env_vars(ENV_PATH, updates)
            flash("Settings saved. Empty fields are left unchanged.", "success")
        else:
            flash("No changes submitted.", "info")
        return redirect(url_for("settings_env"))

    field_rows: list[dict[str, Any]] = []
    for key, label, is_secret in SETTINGS_FIELDS:
        current = env_vars.get(key, "")
        field_rows.append(
            {
                "key": key,
                "label": label,
                "is_secret": is_secret,
                "configured": bool(current),
                "default_value": current if not is_secret else "",
                "placeholder": "Configured (leave blank to keep current)" if current else "Not set",
            }
        )

    return render_template("env_settings.html", fields=field_rows)


@app.route("/auth/google/start")
def google_auth_start() -> str:
    client_id, _, redirect_uri = get_google_oauth_config()
    if not client_id:
        flash("Missing GOOGLE_DRIVE_OAUTH_CLIENT_ID in .env.", "error")
        return redirect(url_for("settings_env"))

    state = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_OAUTH_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return redirect(f"{GOOGLE_AUTH_BASE_URL}?{urlencode(params)}")


@app.route("/auth/google/callback")
def google_auth_callback() -> str:
    saved_state = session.get("google_oauth_state")
    state = request.args.get("state", "")
    if not saved_state or state != saved_state:
        flash("Invalid OAuth state. Please try connecting Google Drive again.", "error")
        return redirect(url_for("settings_env"))

    if request.args.get("error"):
        flash(f"Google OAuth error: {request.args.get('error')}", "error")
        return redirect(url_for("settings_env"))

    code = request.args.get("code", "")
    if not code:
        flash("Google OAuth callback did not include an authorization code.", "error")
        return redirect(url_for("settings_env"))

    client_id, client_secret, redirect_uri = get_google_oauth_config()
    if not client_id or not client_secret:
        flash("Missing GOOGLE_DRIVE_OAUTH_CLIENT_ID or GOOGLE_DRIVE_OAUTH_CLIENT_SECRET in .env.", "error")
        return redirect(url_for("settings_env"))

    token_resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if token_resp.status_code >= 400:
        flash(f"Token exchange failed: {token_resp.text}", "error")
        return redirect(url_for("settings_env"))

    payload = token_resp.json()
    access_token = payload.get("access_token", "")
    refresh_token = payload.get("refresh_token", "")
    updates: dict[str, str] = {}
    if access_token:
        updates["GOOGLE_DRIVE_ACCESS_TOKEN"] = access_token
    if refresh_token:
        updates["GOOGLE_DRIVE_REFRESH_TOKEN"] = refresh_token

    if updates:
        upsert_env_vars(ENV_PATH, updates)
        flash("Google Drive connected. Tokens saved to .env.", "success")
    else:
        flash("OAuth completed but no tokens were returned.", "error")
    return redirect(url_for("settings_env"))


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
