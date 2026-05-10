import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TikTok slideshow JSON scripts from product markdown using OpenAI, "
            "input schema guidance, and template examples."
        )
    )
    parser.add_argument(
        "--product-md",
        type=Path,
        required=True,
        help="Path to the product description markdown file in products/.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("input_schema.json"),
        help="Path to the schema guidance JSON.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("template_input_schema.json"),
        help="Path to the schema template JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts"),
        help="Directory where generated script JSON will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model to use (default: gpt-5.4).",
    )
    parser.add_argument(
        "--slides",
        type=int,
        default=7,
        help="Target number of slides to ask for.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Model temperature.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt preview and skip API call.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_prompt(
    product_markdown: str,
    schema_text: str,
    template_text: str,
    target_slide_count: int,
) -> str:
    return f"""
You are writing a high-performing TikTok slideshow script in JSON.

Goals:
- Output exactly one JSON object.
- Follow the schema guidance exactly.
- Mimic the quality, rhythm, and structure style of the provided template.
- Keep language punchy, specific, and emotionally vivid.
- Keep each slide concise and skimmable.
- Keep the overall narrative cohesive: hook -> value/tips -> bridge -> final CTA.

Requirements:
- Return valid JSON only (no markdown fences, no commentary).
- Must include root keys:
  slideCount, imageDescriptor, productWebsite, textColor, textBackgroundColor, slides
- slideCount must be {target_slide_count}.
- slides must have exactly {target_slide_count} items.
- slideType can include title, tip, cta_bridge, cta_final.
- Every slide must include title and backgroundImage.
- Non-title slides must include content.
- Every backgroundImage must include descriptor, style, mood, negativePrompt.
- imageDescriptor should preserve visual coherence across all slides.
- Use textColor '#FFFFFF' and textBackgroundColor '#000000' unless product strongly demands a different contrast.
- cta_final should naturally mention the product website.

Schema guidance JSON:
{schema_text}

Template JSON (style/structure reference):
{template_text}

Product description markdown:
{product_markdown}
""".strip()


def parse_llm_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON object.")

    return json.loads(candidate[start : end + 1])


def validate_script_shape(payload: dict[str, Any]) -> None:
    required_root = [
        "slideCount",
        "imageDescriptor",
        "productWebsite",
        "textColor",
        "textBackgroundColor",
        "slides",
    ]
    missing = [key for key in required_root if key not in payload]
    if missing:
        raise ValueError(f"Generated JSON missing root keys: {missing}")

    if not isinstance(payload["slides"], list) or not payload["slides"]:
        raise ValueError("`slides` must be a non-empty array.")
    if payload["slideCount"] != len(payload["slides"]):
        raise ValueError("`slideCount` must equal `len(slides)`.")

    for index, slide in enumerate(payload["slides"], start=1):
        for key in ("slideType", "title", "backgroundImage"):
            if key not in slide:
                raise ValueError(f"Slide #{index} missing key: {key}")
        if slide["slideType"] != "title" and "content" not in slide:
            raise ValueError(f"Slide #{index} of type {slide['slideType']} is missing `content`.")

        bg = slide["backgroundImage"]
        for key in ("descriptor", "style", "mood", "negativePrompt"):
            if key not in bg:
                raise ValueError(f"Slide #{index} backgroundImage missing key: {key}")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "product"


def get_next_index(output_dir: Path, product_slug: str) -> int:
    pattern = re.compile(rf"^{re.escape(product_slug)}_(\d+)\.json$")
    max_index = 0
    for file_path in output_dir.glob(f"{product_slug}_*.json"):
        match = pattern.match(file_path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def generate_script(args: argparse.Namespace) -> Path | None:
    load_env_file(Path(".env"))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env.")

    product_md = args.product_md.read_text(encoding="utf-8")
    schema_text = args.schema.read_text(encoding="utf-8")
    template_text = args.template.read_text(encoding="utf-8")
    prompt = build_prompt(product_md, schema_text, template_text, args.slides)

    if args.dry_run:
        print("Dry run enabled. Prompt preview:")
        print("-" * 80)
        print(prompt[:4000])
        print("-" * 80)
        print("Dry run complete (no API request made).")
        return None

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=args.model,
        input=[
            {"role": "system", "content": "You produce only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
    )
    output_text = response.output_text
    generated = parse_llm_json(output_text)
    validate_script_shape(generated)

    product_slug = slugify(args.product_md.stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_index = get_next_index(args.output_dir, product_slug)
    output_path = args.output_dir / f"{product_slug}_{file_index:02d}.json"
    output_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")

    return output_path


def main() -> None:
    args = parse_args()
    output_path = generate_script(args)
    if output_path is not None:
        print(f"Wrote generated slideshow script to: {output_path}")


if __name__ == "__main__":
    main()
