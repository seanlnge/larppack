import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed all images from a directory using CLIP ViT-B/16 and save as JSON."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("larppack"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="clip-ViT-B-16",
        help="Local path or model identifier for sentence-transformers CLIP model.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("larppack_embeddings.json"),
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only embed images not already present in output JSON by image path.",
    )
    return parser.parse_args()


def get_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def to_rel_posix(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def load_existing_rows(output_json: Path) -> tuple[list[dict[str, object]], set[str]]:
    if not output_json.exists():
        return [], set()

    rows = json.loads(output_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return [], set()

    normalized_rows: list[dict[str, object]] = []
    existing_paths: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        embedding = row.get("embedding")
        if embedding is None:
            continue

        image_path = str(row.get("image_path", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        if not image_path and file_name:
            image_path = file_name
        if not file_name and image_path:
            file_name = Path(image_path).name
        if not image_path or not file_name:
            continue

        normalized_rows.append(
            {
                "file_name": file_name,
                "image_path": image_path,
                "embedding": embedding,
            }
        )
        existing_paths.add(image_path)

    return normalized_rows, existing_paths


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")

    image_paths = get_image_paths(args.images_dir)
    if not image_paths:
        raise ValueError(f"No supported image files found in: {args.images_dir}")

    rows, existing_paths = load_existing_rows(args.output_json)

    if args.incremental:
        images_to_embed = [path for path in image_paths if to_rel_posix(path, args.images_dir) not in existing_paths]
    else:
        images_to_embed = image_paths
        rows = []
        existing_paths = set()

    if not images_to_embed:
        args.output_json.write_text(json.dumps(rows), encoding="utf-8")
        print(f"No new images to embed. Existing embeddings: {len(rows)}")
        return

    model = SentenceTransformer(args.model_path)

    for image_path in images_to_embed:
        try:
            with Image.open(image_path) as image:
                embedding = model.encode(image)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Skipping unreadable image {image_path.name}: {exc}")
            continue

        rel_path = to_rel_posix(image_path, args.images_dir)
        rows.append(
            {
                "file_name": image_path.name,
                "image_path": rel_path,
                "embedding": embedding.tolist(),
            }
        )
        existing_paths.add(rel_path)
        print(f"Embedded {rel_path}")

    args.output_json.write_text(json.dumps(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} embeddings to {args.output_json} ({len(images_to_embed)} processed)")


if __name__ == "__main__":
    main()
