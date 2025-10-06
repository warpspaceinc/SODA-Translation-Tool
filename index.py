#!/usr/bin/env python3
import os
import json
import click
from dotenv import load_dotenv
from src.dataset_loader import (
    load_soda_dataset,
    generate_filtered_samples,
    create_dataset_from_samples,
    save_to_huggingface
)
from src.processor import run_translation
from src.utils import (
    validate_dataset,
    print_validation_result,
    CheckpointManager,
    estimate_translation_cost,
    print_cost_estimation
)

# .env 파일 로드
load_dotenv(override=True)


@click.group()
def cli():
    """SODA-KR Translation CLI"""
    pass


@cli.command()
@click.option("--samples", default=5, help="Number of samples to preview")
def preview(samples):
    """데이터셋 샘플 확인"""
    click.echo(f"Loading SODA dataset...")
    dataset = load_soda_dataset()

    click.echo(f"\nDataset size: {len(dataset)}")
    click.echo(f"Preview {samples} samples:\n")

    for i, sample in enumerate(generate_filtered_samples(dataset, max_samples=samples)):
        click.echo(f"{'='*60}")
        click.echo(f"Sample {i+1} (Index: {sample['index']})")
        click.echo(f"{'='*60}")
        click.echo(json.dumps(sample, ensure_ascii=False, indent=2))
        click.echo()


@cli.command()
@click.option("--model", required=True, help="Translation model")
@click.option("--chunk-size", default=10, help="Chunk size for parallel processing")
@click.option("--max-samples", default=None, type=int, help="Max number of samples to translate")
@click.option("--target-lang", default="Korean", help="Target language (e.g., Korean, English, Japanese)")
@click.option("--checkpoint", default="./temp/checkpoints/checkpoint.json", help="Checkpoint file path")
@click.option("--output", default="./temp/output/translated.json", help="Output JSON file path")
def translate(model, chunk_size, max_samples, target_lang, checkpoint, output):
    """번역 실행"""
    click.echo(f"Starting translation...")
    click.echo(f"Model: {model}")
    click.echo(f"Target language: {target_lang}")
    click.echo(f"Chunk size: {chunk_size}")
    click.echo(f"Max samples: {max_samples or 'All'}")

    # 데이터셋 로드
    dataset = load_soda_dataset()
    samples = list(generate_filtered_samples(dataset, max_samples=max_samples))

    click.echo(f"\nFiltered samples: {len(samples)}")

    # 번역 실행
    translated = run_translation(
        samples=samples,
        model=model,
        chunk_size=chunk_size,
        checkpoint_path=checkpoint,
        target_lang=target_lang
    )

    # 결과 저장
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    click.echo(f"\nTranslation saved to: {output}")

    # 통계 출력
    checkpoint_mgr = CheckpointManager(checkpoint, model)
    stats = checkpoint_mgr.get_stats()
    click.echo(f"\nStatistics:")
    click.echo(f"  Total processed: {stats['total_processed']}")
    click.echo(f"  Success: {stats['success']}")
    click.echo(f"  Errors: {stats['errors']}")


@cli.command()
@click.option("--model", required=True, help="Translation model to test")
@click.option("--samples", default=10, help="Number of samples to test (default: 10)")
@click.option("--total-size", default=None, type=int, help="Total dataset size (auto-detect if not specified)")
def estimate(model, samples, total_size):
    """비용 및 시간 추정"""
    click.echo(f"Estimating cost and duration...")
    click.echo(f"Model: {model}")
    click.echo(f"Sample size: {samples}\n")

    # 전체 데이터셋 크기 자동 감지
    if total_size is None:
        dataset = load_soda_dataset()
        all_samples = list(generate_filtered_samples(dataset))
        total_size = len(all_samples)
        click.echo(f"Auto-detected total dataset size: {total_size}\n")

    # 비용 추정 실행
    result = estimate_translation_cost(
        model=model,
        sample_count=samples,
        total_dataset_size=total_size
    )

    # 결과 출력
    print_cost_estimation(result)


@cli.command()
@click.option("--input", "input_path", required=True, help="Input JSON file to validate")
def validate(input_path):
    """번역 결과 검증"""
    click.echo(f"Validating {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    result = validate_dataset(items)
    print_validation_result(result)


@cli.command()
@click.option("--input", "input_path", required=True, help="Input JSON file to upload")
@click.option("--repo-name", required=True, help="HuggingFace repository name (e.g., username/dataset-name)")
@click.option("--private", is_flag=True, help="Make repository private")
def upload(input_path, repo_name, private):
    """HuggingFace에 업로드"""
    click.echo(f"Uploading to HuggingFace...")
    click.echo(f"Repository: {repo_name}")
    click.echo(f"Private: {private}")

    # JSON 파일 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # 검증
    click.echo("\nValidating dataset before upload...")
    result = validate_dataset(items)

    if not result['valid']:
        click.echo("Validation failed. Please fix errors before uploading.")
        print_validation_result(result)
        return

    click.echo(f"Validation passed ({result['total']} items)")

    # Dataset 생성
    dataset = create_dataset_from_samples(items)

    # 업로드
    save_to_huggingface(
        dataset=dataset,
        repo_name=repo_name,
        private=private
    )


@cli.command()
@click.option("--checkpoint", required=True, help="Checkpoint file to resume from")
@click.option("--model", required=True, help="Translation model")
@click.option("--chunk-size", default=10, help="Chunk size")
@click.option("--target-lang", default="Korean", help="Target language")
@click.option("--output", default="./temp/output/translated.json", help="Output file")
def resume(checkpoint, model, chunk_size, target_lang, output):
    """중단된 작업 재개"""
    click.echo(f"Resuming from checkpoint: {checkpoint}")

    # 체크포인트 로드
    checkpoint_mgr = CheckpointManager(checkpoint, model)
    stats = checkpoint_mgr.get_stats()

    click.echo(f"Checkpoint stats:")
    click.echo(f"  Model: {stats['model']}")
    click.echo(f"  Processed: {stats['total_processed']}")
    click.echo(f"  Errors: {stats['errors']}")
    click.echo(f"  Last updated: {stats['last_updated']}")

    # 데이터셋 로드
    dataset = load_soda_dataset()
    all_samples = list(generate_filtered_samples(dataset))

    click.echo(f"\nTotal samples in dataset: {len(all_samples)}")

    # 번역 재개
    translated = run_translation(
        samples=all_samples,
        model=model,
        chunk_size=chunk_size,
        checkpoint_path=checkpoint,
        target_lang=target_lang
    )

    # 결과 저장
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    click.echo(f"\nTranslation saved to: {output}")


@cli.command()
@click.option("--model", default="moonshotai/kimi-k2-0905", help="Translation model")
@click.option("--samples", default=10, help="Number of samples to translate")
@click.option("--chunk-size", default=10, help="Chunk size")
@click.option("--target-lang", default="Korean", help="Target language")
@click.option("--repo-name", default="CaveduckAI/simplified_soda_kr", help="HuggingFace repo")
def quick_run(model, samples, chunk_size, target_lang, repo_name):
    """빠른 실행: 번역 + 검증 + 업로드를 한번에"""
    click.echo("Quick Run: Translate -> Validate -> Upload")
    click.echo(f"Model: {model}")
    click.echo(f"Samples: {samples}")
    click.echo(f"Target language: {target_lang}")
    click.echo(f"Repository: {repo_name}\n")

    # 1. 번역
    click.echo("=" * 60)
    click.echo("Step 1: Translation")
    click.echo("=" * 60)

    dataset = load_soda_dataset()
    sample_list = list(generate_filtered_samples(dataset, max_samples=samples))

    checkpoint_path = "./temp/checkpoints/quick_run.json"
    output_path = "./temp/output/quick_run.json"

    translated = run_translation(
        samples=sample_list,
        model=model,
        chunk_size=chunk_size,
        checkpoint_path=checkpoint_path,
        target_lang=target_lang
    )

    # 결과 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    # 2. 검증
    click.echo("\n" + "=" * 60)
    click.echo("Step 2: Validation")
    click.echo("=" * 60)

    result = validate_dataset(translated)
    print_validation_result(result)

    if not result['valid']:
        click.echo("❌ Validation failed. Stopping before upload.")
        return

    # 3. 업로드
    click.echo("\n" + "=" * 60)
    click.echo("Step 3: Upload to HuggingFace")
    click.echo("=" * 60)

    dataset_obj = create_dataset_from_samples(translated)
    save_to_huggingface(
        dataset=dataset_obj,
        repo_name=repo_name,
        private=False
    )

    click.echo("\n" + "=" * 60)
    click.echo("Quick run completed!")
    click.echo("=" * 60)
    click.echo(f"Dataset URL: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    cli()
