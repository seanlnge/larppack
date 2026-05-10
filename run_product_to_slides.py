import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full workflow from product markdown to generated slide PNG outputs."
    )
    parser.add_argument(
        "--product-md",
        type=Path,
        required=True,
        help="Path to product markdown file (for example: products/my_product.md).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("input_schema.json"),
        help="Schema guidance JSON path.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("template_input_schema.json"),
        help="Template JSON path.",
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path("scripts"),
        help="Directory where generated slideshow script JSON is written.",
    )
    parser.add_argument(
        "--slides-output-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder for final rendered slide image folders.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("larppack"),
        help="Image pool directory for background selection.",
    )
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        default=Path("larppack_embeddings.json"),
        help="Precomputed image embeddings JSON path.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model used for script generation.",
    )
    parser.add_argument(
        "--clip-model-path",
        type=str,
        default="clip-ViT-B-16",
        help="CLIP model path/id used for image ranking.",
    )
    parser.add_argument(
        "--slides",
        type=int,
        default=7,
        help="Target number of slides for generated script JSON.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature for the script-writing model.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional explicit font path for slide rendering.",
    )
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=0.35,
        help="Negative prompt subtraction weight for CLIP query embedding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only dry-run script generation prompt, skip rendering.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    return completed.stdout + "\n" + completed.stderr


def extract_script_path(output: str) -> Path:
    match = re.search(r"Wrote generated slideshow script to:\s*(.+)", output)
    if not match:
        raise RuntimeError("Could not find generated script path in generator output.")
    return Path(match.group(1).strip())


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parent

    if not args.product_md.exists():
        raise FileNotFoundError(f"Product markdown not found: {args.product_md}")

    print("Step 1/2: Generating slideshow script JSON from product markdown...")
    gen_cmd = [
        sys.executable,
        "generate_tiktok_script.py",
        "--product-md",
        str(args.product_md),
        "--schema",
        str(args.schema),
        "--template",
        str(args.template),
        "--output-dir",
        str(args.scripts_dir),
        "--model",
        args.openai_model,
        "--slides",
        str(args.slides),
        "--temperature",
        str(args.temperature),
    ]
    if args.dry_run:
        gen_cmd.append("--dry-run")

    generation_output = run_command(gen_cmd, workspace)
    if args.dry_run:
        print("Dry run complete. Rendering skipped.")
        return

    generated_script_path = extract_script_path(generation_output)
    if not generated_script_path.is_absolute():
        generated_script_path = (workspace / generated_script_path).resolve()

    print(f"Generated script: {generated_script_path}")
    print("Step 2/2: Rendering final slide images from generated script...")
    render_cmd = [
        sys.executable,
        "generate_slides.py",
        "--input-json",
        str(generated_script_path),
        "--images-dir",
        str(args.images_dir),
        "--embeddings-json",
        str(args.embeddings_json),
        "--model-path",
        args.clip_model_path,
        "--output-root",
        str(args.slides_output_root),
        "--negative-weight",
        str(args.negative_weight),
    ]
    if args.font_path is not None:
        render_cmd.extend(["--font-path", str(args.font_path)])

    run_command(render_cmd, workspace)
    print("Workflow complete.")


if __name__ == "__main__":
    main()
