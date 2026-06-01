#!/usr/bin/env python3
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


async def submit_one(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    audio_path: Path,
    model: str,
    language: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data: dict[str, str] = {"mode": "sync", "model": model}
    if language:
        data["language"] = language
    started = time.perf_counter()
    with audio_path.open("rb") as handle:
        files = {"file": (audio_path.name, handle, "audio/wav")}
        response = await client.post(
            f"{base_url.rstrip('/')}/v1/stt/transcribe",
            headers=headers,
            data=data,
            files=files,
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started
    payload: dict[str, Any]
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text}
    return {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "elapsed_ms": elapsed * 1000.0,
        "payload": payload,
    }


async def run_case(
    base_url: str,
    token: str,
    audio_path: Path,
    model: str,
    language: str | None,
    requests: int,
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=max(concurrency * 2, 10), max_keepalive_connections=max(concurrency, 5))
    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def run_guarded() -> dict[str, Any]:
            async with semaphore:
                return await submit_one(client, base_url, token, audio_path, model, language, timeout)

        started = time.perf_counter()
        results = await asyncio.gather(*(run_guarded() for _ in range(requests)))
        elapsed = time.perf_counter() - started

    latencies = [r["elapsed_ms"] for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    return {
        "model": model,
        "language": language,
        "audio": str(audio_path),
        "requests": requests,
        "concurrency": concurrency,
        "ok": len(latencies),
        "failed": len(failures),
        "elapsed_s": elapsed,
        "rps": len(results) / elapsed if elapsed > 0 else 0.0,
        "latency_ms": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "failures": failures[:5],
    }


async def main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for model in args.models:
        for language in args.languages:
            lang_value = None if language in {"auto", "none", ""} else language
            for audio_path in args.audio:
                for concurrency in args.concurrency:
                    case = await run_case(
                        args.base_url,
                        args.token,
                        audio_path,
                        model,
                        lang_value,
                        args.requests,
                        concurrency,
                        args.timeout,
                    )
                    cases.append(case)
                    print(
                        f"{model} lang={language} audio={audio_path.name} c={concurrency} "
                        f"ok={case['ok']} failed={case['failed']} rps={case['rps']:.2f} "
                        f"p95={case['latency_ms']['p95']:.1f}ms"
                    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"stt-qwen3-5090-{timestamp}.json"
    md_path = output_dir / f"stt-qwen3-5090-{timestamp}.md"
    json_path.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [
        "# Qwen3-ASR RTX 5090 Benchmark",
        "",
        "| model | language | audio | concurrency | requests | ok | failed | rps | avg ms | p95 ms | p99 ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        rows.append(
            "| {model} | {language} | {audio} | {concurrency} | {requests} | {ok} | {failed} | {rps:.2f} | {avg:.1f} | {p95:.1f} | {p99:.1f} |".format(
                model=case["model"],
                language=case["language"] or "auto",
                audio=Path(case["audio"]).name,
                concurrency=case["concurrency"],
                requests=case["requests"],
                ok=case["ok"],
                failed=case["failed"],
                rps=case["rps"],
                avg=case["latency_ms"]["avg"],
                p95=case["latency_ms"]["p95"],
                p99=case["latency_ms"]["p99"],
            )
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR API batching on RTX 5090.")
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument("--token", default="")
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["0.6b", "1.7b"])
    parser.add_argument("--languages", nargs="+", default=["vi", "en", "auto"])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 96, 128])
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", default="benchmarks/stt-qwen3-5090")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
