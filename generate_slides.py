import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, UnidentifiedImageError
from sentence_transformers import SentenceTransformer


CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
DEFAULT_NEGATIVE_WEIGHT = 0.35
DEFAULT_SELECTION_TOP_K = 8
DEFAULT_SELECTION_TEMPERATURE = 0.45
DEFAULT_SCORE_NOISE_STD = 0.012
DEFAULT_RECENT_RUN_PENALTY = 0.08
DEFAULT_AVOID_RECENT_RUNS = 4
CARD_WIDTH_RATIO = 0.92
CARD_RADIUS = 24
TOP_CARD_START_Y = 280
CARD_GAP = 44
TOP_CARD_TEXT_PADDING_X = 68
TOP_CARD_TEXT_PADDING_Y = 52
BOTTOM_CARD_TEXT_PADDING_X = 60
BOTTOM_CARD_TEXT_PADDING_Y = 48
OVERLAY_ALPHA = 52
TARGET_SEMIBOLD_WEIGHT = 600.0
STICKER_PADDING_X = 48
STICKER_PADDING_Y = 22
STICKER_RADIUS = 24


@dataclass
class SlideSelection:
    slide_index: int
    slide_type: str
    file_name: str
    image_path: str
    similarity: float
    adjusted_score: float
    recent_penalty_count: int
    query_text: str
    negative_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TikTok-style slideshow PNGs using schema input + CLIP image matching "
            "without repeating background images."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Path to slideshow input JSON.")
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        default=Path("larppack_embeddings.json"),
        help="JSON file containing image embeddings (file_name + embedding).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("larppack"),
        help="Directory containing source image files.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="clip-ViT-B-16",
        help="Local path or model identifier for sentence-transformers CLIP model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory where generated slide folder will be created.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional explicit path to a bold font file (.ttf/.otf).",
    )
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=DEFAULT_NEGATIVE_WEIGHT,
        help="Weight used when subtracting negative prompt embedding from positive query embedding.",
    )
    parser.add_argument(
        "--selection-top-k",
        type=int,
        default=DEFAULT_SELECTION_TOP_K,
        help="Sample from top-K scored candidates per slide for diversity.",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=DEFAULT_SELECTION_TEMPERATURE,
        help="Softmax temperature for top-K sampling (<=0 makes selection deterministic).",
    )
    parser.add_argument(
        "--score-noise-std",
        type=float,
        default=DEFAULT_SCORE_NOISE_STD,
        help="Gaussian noise std added to scores before candidate sampling.",
    )
    parser.add_argument(
        "--recent-run-penalty",
        type=float,
        default=DEFAULT_RECENT_RUN_PENALTY,
        help="Score penalty applied per recent usage count for each image path.",
    )
    parser.add_argument(
        "--avoid-recent-runs",
        type=int,
        default=DEFAULT_AVOID_RECENT_RUNS,
        help="Number of recent runs for this input stem to use for image reuse penalties.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible stochastic image selection.",
    )
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm


def ensure_required_keys(data: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


def load_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ensure_required_keys(
        payload,
        ["slideCount", "imageDescriptor", "productWebsite", "textColor", "textBackgroundColor", "slides"],
        "input payload",
    )
    if not isinstance(payload["slides"], list):
        raise ValueError("`slides` must be a list.")
    if payload["slideCount"] != len(payload["slides"]):
        raise ValueError(
            f"slideCount ({payload['slideCount']}) must match number of slides ({len(payload['slides'])})."
        )

    for index, slide in enumerate(payload["slides"], start=1):
        ensure_required_keys(slide, ["slideType", "title", "backgroundImage"], f"slide #{index}")
        bg = slide["backgroundImage"]
        ensure_required_keys(bg, ["descriptor", "style", "mood", "negativePrompt"], f"slide #{index}.backgroundImage")
        if slide["slideType"] != "title" and "content" not in slide:
            raise ValueError(f"slide #{index} has slideType '{slide['slideType']}' but is missing `content`.")
    return payload


def load_embedding_rows(embeddings_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows = json.loads(embeddings_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Embedding JSON must be a non-empty list.")

    file_names: list[str] = []
    image_paths: list[str] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        file_name = row.get("file_name")
        image_path = row.get("image_path")
        embedding = row.get("embedding")
        if not file_name or embedding is None:
            continue
        if not image_path:
            image_path = file_name
        vector = np.asarray(embedding, dtype=np.float32)
        vectors.append(normalize(vector))
        file_names.append(str(file_name))
        image_paths.append(str(image_path))

    if not file_names:
        raise ValueError("No valid embedding rows found in embeddings JSON.")
    return file_names, image_paths, np.vstack(vectors)


def build_query_text(global_descriptor: str, slide: dict[str, Any]) -> tuple[str, str]:
    bg = slide["backgroundImage"]
    positive_parts = [global_descriptor, bg["descriptor"], bg["style"], bg["mood"]]
    positive_text = ", ".join(part.strip() for part in positive_parts if part and str(part).strip())
    negative_prompt = str(bg.get("negativePrompt", "")).strip()
    return positive_text, negative_prompt


def load_recent_usage_counts(output_root: Path, input_stem: str, avoid_recent_runs: int) -> dict[str, int]:
    if avoid_recent_runs <= 0 or not output_root.exists():
        return {}

    candidate_dirs = sorted(
        [p for p in output_root.glob(f"{input_stem}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:avoid_recent_runs]

    counts: dict[str, int] = {}
    for output_dir in candidate_dirs:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            rel_path = str(row.get("selected_image_path") or row.get("selected_image") or "").strip()
            if not rel_path:
                continue
            counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def sample_candidate_index(
    candidate_indices: list[int],
    adjusted_scores: np.ndarray,
    selection_top_k: int,
    selection_temperature: float,
    rng: np.random.Generator,
) -> int:
    if not candidate_indices:
        raise RuntimeError("No candidate indices available for sampling.")

    top_k = max(1, min(selection_top_k, len(candidate_indices)))
    top_candidates = candidate_indices[:top_k]

    if selection_temperature <= 0 or top_k == 1:
        return top_candidates[0]

    logits = adjusted_scores[top_candidates] / selection_temperature
    logits = logits - np.max(logits)
    exp_vals = np.exp(logits)
    probs = exp_vals / np.sum(exp_vals)
    chosen = int(rng.choice(np.asarray(top_candidates), p=probs))
    return chosen


def select_images_for_slides(
    model: SentenceTransformer,
    payload: dict[str, Any],
    image_file_names: list[str],
    image_rel_paths: list[str],
    image_vectors: np.ndarray,
    images_dir: Path,
    negative_weight: float,
    recent_usage_counts: dict[str, int],
    recent_run_penalty: float,
    selection_top_k: int,
    selection_temperature: float,
    score_noise_std: float,
    rng: np.random.Generator,
) -> list[SlideSelection]:
    used_indices: set[int] = set()
    selections: list[SlideSelection] = []
    global_descriptor = str(payload["imageDescriptor"])

    for slide_index, slide in enumerate(payload["slides"], start=1):
        positive_text, negative_prompt = build_query_text(global_descriptor, slide)
        positive_vec = normalize(np.asarray(model.encode(positive_text), dtype=np.float32))
        if negative_prompt:
            negative_vec = normalize(np.asarray(model.encode(negative_prompt), dtype=np.float32))
            query_vec = normalize(positive_vec - (negative_weight * negative_vec))
        else:
            query_vec = positive_vec

        scores = image_vectors @ query_vec
        adjusted_scores = scores.copy()

        for idx, rel_path in enumerate(image_rel_paths):
            usage_count = recent_usage_counts.get(rel_path, 0)
            if usage_count > 0:
                adjusted_scores[idx] -= recent_run_penalty * usage_count

        if score_noise_std > 0:
            adjusted_scores = adjusted_scores + rng.normal(0, score_noise_std, size=adjusted_scores.shape[0])

        ranked_indices = np.argsort(adjusted_scores)[::-1]

        candidate_indices: list[int] = []
        for idx in ranked_indices:
            if int(idx) in used_indices:
                continue
            candidate = images_dir / image_rel_paths[int(idx)]
            if candidate.exists():
                candidate_indices.append(int(idx))

        if not candidate_indices:
            raise RuntimeError(f"Could not find an unused readable image for slide #{slide_index}.")

        chosen_idx = sample_candidate_index(
            candidate_indices=candidate_indices,
            adjusted_scores=adjusted_scores,
            selection_top_k=selection_top_k,
            selection_temperature=selection_temperature,
            rng=rng,
        )
        used_indices.add(chosen_idx)
        penalty_count = recent_usage_counts.get(image_rel_paths[chosen_idx], 0)
        selections.append(
            SlideSelection(
                slide_index=slide_index,
                slide_type=str(slide["slideType"]),
                file_name=image_file_names[chosen_idx],
                image_path=image_rel_paths[chosen_idx],
                similarity=float(scores[chosen_idx]),
                adjusted_score=float(adjusted_scores[chosen_idx]),
                recent_penalty_count=penalty_count,
                query_text=positive_text,
                negative_prompt=negative_prompt,
            )
        )
    return selections


def choose_font_path(user_font_path: Path | None) -> Path | None:
    if user_font_path is not None:
        return user_font_path if user_font_path.exists() else None

    candidates = [
        Path("font/TikTokSans-VariableFont_opsz,slnt,wdth,wght.ttf"),
        Path("C:/Windows/Fonts/proximanova_bold.otf"),
        Path("C:/Windows/Fonts/Montserrat-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/seguisb.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_font(size: int, font_path: Path | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is not None:
        try:
            font = ImageFont.truetype(str(font_path), size=size)
            apply_medium_weight_if_variable(font, font_path)
            return font
        except OSError:
            pass
    return ImageFont.load_default()


def apply_medium_weight_if_variable(font: ImageFont.ImageFont, font_path: Path) -> None:
    if not isinstance(font, ImageFont.FreeTypeFont):
        return
    if "tiktoksans-variablefont" not in font_path.name.lower():
        return

    try:
        axes = font.get_variation_axes()
    except (AttributeError, OSError):
        return

    axis_values: list[float] = []
    has_weight_axis = False

    for axis in axes:
        axis_name_raw = axis.get("name", b"")
        if isinstance(axis_name_raw, bytes):
            axis_name = axis_name_raw.decode("utf-8", errors="ignore").lower()
        else:
            axis_name = str(axis_name_raw).lower()

        default_value = float(axis.get("default", axis.get("minimum", 0)))
        min_value = float(axis.get("minimum", default_value))
        max_value = float(axis.get("maximum", default_value))

        if "weight" in axis_name or "wght" in axis_name:
            has_weight_axis = True
            default_value = min(max(TARGET_SEMIBOLD_WEIGHT, min_value), max_value)

        axis_values.append(default_value)

    if not has_weight_axis:
        return

    try:
        font.set_variation_by_axes(axis_values)
    except OSError:
        try:
            font.set_variation_by_name("SemiBold")
        except OSError:
            pass


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=6)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width, _ = measure_text(draw, candidate, font)
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def fit_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    max_width: int,
    max_lines: int,
    initial_size: int,
    min_size: int,
) -> tuple[ImageFont.ImageFont, str, tuple[int, int]]:
    size = initial_size
    while size >= min_size:
        font = load_font(size, font_path)
        wrapped = wrap_text(draw, text, font, max_width)
        line_count = wrapped.count("\n") + 1 if wrapped else 0
        if line_count <= max_lines:
            block_size = measure_text(draw, wrapped, font)
            if block_size[0] <= max_width:
                return font, wrapped, block_size
        size -= 2

    fallback_font = load_font(min_size, font_path)
    wrapped = wrap_text(draw, text, fallback_font, max_width)
    return fallback_font, wrapped, measure_text(draw, wrapped, fallback_font)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    text_width, text_height = measure_text(draw, text, font)
    x = left + (right - left - text_width) // 2
    y = top + (bottom - top - text_height) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=6)


def draw_text_card(
    draw: ImageDraw.ImageDraw,
    text: str,
    top_y: int,
    font_path: Path | None,
    max_lines: int,
    initial_font_size: int,
    min_font_size: int,
    card_fill: tuple[int, int, int],
    text_fill: tuple[int, int, int],
    padding_x: int,
    padding_y: int,
) -> tuple[int, int]:
    card_width = int(CANVAS_WIDTH * CARD_WIDTH_RATIO)
    card_left = (CANVAS_WIDTH - card_width) // 2
    card_right = card_left + card_width
    text_max_width = card_width - (padding_x * 2)

    font, wrapped_text, text_size = fit_text_block(
        draw=draw,
        text=text,
        font_path=font_path,
        max_width=text_max_width,
        max_lines=max_lines,
        initial_size=initial_font_size,
        min_size=min_font_size,
    )

    card_height = text_size[1] + (padding_y * 2)
    card_bottom = top_y + card_height

    draw.rounded_rectangle(
        (card_left, top_y, card_right, card_bottom),
        radius=CARD_RADIUS,
        fill=card_fill,
    )
    # Use glyph bbox offsets so top/bottom visual padding is symmetric.
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", spacing=6)
    text_width = bbox[2] - bbox[0]
    text_max_width = card_width - (padding_x * 2)
    text_x = card_left + padding_x + ((text_max_width - text_width) // 2) - bbox[0]
    text_y = top_y + padding_y - bbox[1]
    draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=text_fill, align="center", spacing=6)
    return card_bottom, card_left


def crop_to_cover(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    if src_ratio > tgt_ratio:
        new_h = target_h
        new_w = int(new_h * src_ratio)
    else:
        new_w = target_w
        new_h = int(new_w / src_ratio)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def draw_website_sticker(
    canvas: Image.Image,
    website: str,
    font_path: Path | None,
    center_x: int,
    center_y: int,
) -> None:
    tmp_draw = ImageDraw.Draw(canvas)
    sticker_font = load_font(66, font_path)
    bbox = tmp_draw.multiline_textbbox((0, 0), website, font=sticker_font, align="center", spacing=6)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    sticker_w = int(math.ceil(text_width + (STICKER_PADDING_X * 2)))
    sticker_h = int(math.ceil(text_height + (STICKER_PADDING_Y * 2)))
    sticker = Image.new("RGBA", (sticker_w, sticker_h), (0, 0, 0, 0))
    sticker_draw = ImageDraw.Draw(sticker)
    sticker_draw.rounded_rectangle((0, 0, sticker_w, sticker_h), radius=STICKER_RADIUS, fill=(255, 255, 255, 255))
    text_x = int(STICKER_PADDING_X - bbox[0])
    text_y = int(STICKER_PADDING_Y - bbox[1])
    sticker_draw.multiline_text((text_x, text_y), website, font=sticker_font, fill=(0, 0, 0), align="center")

    rotated = sticker.rotate(-10, expand=True, resample=Image.Resampling.BICUBIC)
    paste_x = center_x - (rotated.width // 2)
    paste_y = center_y - (rotated.height // 2)
    canvas.paste(rotated, (paste_x, paste_y), rotated)


def parse_color(value: str, fallback: str) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(value)
    except ValueError:
        return ImageColor.getrgb(fallback)


def render_slide(
    payload: dict[str, Any],
    slide: dict[str, Any],
    selection: SlideSelection,
    images_dir: Path,
    output_path: Path,
    font_path: Path | None,
) -> None:
    source_image_path = images_dir / selection.image_path
    with Image.open(source_image_path) as source:
        background = crop_to_cover(source.convert("RGB"), CANVAS_WIDTH, CANVAS_HEIGHT)

    canvas = background.convert("RGBA")
    overlay = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, OVERLAY_ALPHA))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    text_color = parse_color(str(payload.get("textColor", "#FFFFFF")), "#FFFFFF")
    card_color = parse_color(str(payload.get("textBackgroundColor", "#000000")), "#000000")

    top_bottom, _ = draw_text_card(
        draw=draw,
        text=str(slide["title"]),
        top_y=TOP_CARD_START_Y,
        font_path=font_path,
        max_lines=4,
        initial_font_size=78,
        min_font_size=46,
        card_fill=card_color,
        text_fill=text_color,
        padding_x=TOP_CARD_TEXT_PADDING_X,
        padding_y=TOP_CARD_TEXT_PADDING_Y,
    )

    content = str(slide.get("content", "")).strip()
    if content:
        bottom_top = top_bottom + CARD_GAP
        bottom_bottom, bottom_left = draw_text_card(
            draw=draw,
            text=content,
            top_y=bottom_top,
            font_path=font_path,
            max_lines=6,
            initial_font_size=66,
            min_font_size=40,
            card_fill=card_color,
            text_fill=text_color,
            padding_x=BOTTOM_CARD_TEXT_PADDING_X,
            padding_y=BOTTOM_CARD_TEXT_PADDING_Y,
        )

        if str(slide["slideType"]) == "cta_final":
            sticker_center_x = CANVAS_WIDTH // 2
            sticker_center_y = min(bottom_bottom + 145, CANVAS_HEIGHT - 220)
            draw_website_sticker(
                canvas=canvas,
                website=str(payload["productWebsite"]),
                font_path=font_path,
                center_x=sticker_center_x,
                center_y=sticker_center_y,
            )
        _ = bottom_left

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG")


def create_output_folder(output_root: Path, input_json_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{input_json_path.stem}_{timestamp}"
    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_manifest(output_dir: Path, selections: list[SlideSelection]) -> None:
    rows: list[dict[str, Any]] = []
    for selection in selections:
        rows.append(
            {
                "slide_index": selection.slide_index,
                "slide_type": selection.slide_type,
                "selected_image": selection.file_name,
                "selected_image_path": selection.image_path,
                "similarity": round(selection.similarity, 6),
                "adjusted_score": round(selection.adjusted_score, 6),
                "recent_penalty_count": selection.recent_penalty_count,
                "query_text": selection.query_text,
                "negative_prompt": selection.negative_prompt,
                "output_file": f"slide{selection.slide_index:02d}_final.png",
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_input(args.input_json)

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    if not args.embeddings_json.exists():
        raise FileNotFoundError(f"Embeddings file does not exist: {args.embeddings_json}")

    image_file_names, image_rel_paths, image_vectors = load_embedding_rows(args.embeddings_json)
    recent_usage_counts = load_recent_usage_counts(
        output_root=args.output_root,
        input_stem=args.input_json.stem,
        avoid_recent_runs=args.avoid_recent_runs,
    )
    rng = np.random.default_rng(args.seed)
    model = SentenceTransformer(args.model_path)
    selections = select_images_for_slides(
        model=model,
        payload=payload,
        image_file_names=image_file_names,
        image_rel_paths=image_rel_paths,
        image_vectors=image_vectors,
        images_dir=args.images_dir,
        negative_weight=args.negative_weight,
        recent_usage_counts=recent_usage_counts,
        recent_run_penalty=args.recent_run_penalty,
        selection_top_k=args.selection_top_k,
        selection_temperature=args.selection_temperature,
        score_noise_std=args.score_noise_std,
        rng=rng,
    )

    output_dir = create_output_folder(args.output_root, args.input_json)
    font_path = choose_font_path(args.font_path)

    for selection, slide in zip(selections, payload["slides"], strict=True):
        output_name = f"slide{selection.slide_index:02d}_final.png"
        output_path = output_dir / output_name
        try:
            render_slide(
                payload=payload,
                slide=slide,
                selection=selection,
                images_dir=args.images_dir,
                output_path=output_path,
                font_path=font_path,
            )
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to render slide #{selection.slide_index} from {selection.file_name}: {exc}") from exc
        print(f"Rendered {output_name} using {selection.file_name}")

    save_manifest(output_dir, selections)
    print(f"Done. Generated {len(selections)} slides in {output_dir}")


if __name__ == "__main__":
    main()
