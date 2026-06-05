#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import math
import re
import shutil
import statistics
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


SAMPLE_RATE = 24000


KOKORO_LANGUAGE_CASES = {
    "a": ("af_heart", "Hello from Kokoro. This is an American English benchmark sentence."),
    "b": ("bf_emma", "Hello from Kokoro. This is a British English benchmark sentence."),
    "e": ("ef_dora", "Hola desde Kokoro. Esta es una frase breve para la prueba."),
    "f": ("ff_siwis", "Bonjour depuis Kokoro. Ceci est une courte phrase de test."),
    "h": ("hf_alpha", "नमस्ते, यह कोकोरो की एक छोटी परीक्षण पंक्ति है।"),
    "i": ("if_sara", "Ciao da Kokoro. Questa è una breve frase di prova."),
    "j": ("jf_alpha", "こんにちは。これはココロの短いテスト文です。"),
    "p": ("pf_dora", "Ola do Kokoro. Esta e uma frase curta para teste."),
    "z": ("zf_xiaobei", "你好，这是 Kokoro 的一个简短测试句子。"),
}


@dataclass
class KokoroResult:
    index: int
    lang_code: str
    voice: str
    ok: bool
    elapsed_ms: float
    audio_duration_ms: float
    rtf: float
    chars: int
    error: str = ""


@dataclass
class OmniVoiceResult:
    index: int
    ok: bool
    status_code: int
    submit_ms: float
    elapsed_ms: float
    audio_duration_ms: float
    rtf: float
    chunks_total: int
    chunks_completed: int
    chunks_failed: int
    size_bytes: int
    request_id: str
    error: str = ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize_kokoro(results: list[KokoroResult], total_elapsed_s: float) -> dict[str, Any]:
    ok = [result for result in results if result.ok]
    latencies = [result.elapsed_ms for result in ok]
    rtfs = [result.rtf for result in ok if result.rtf > 0]
    audio_ms = sum(result.audio_duration_ms for result in ok)
    return {
        "requests": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "total_elapsed_s": round(total_elapsed_s, 3),
        "requests_per_s": round(len(ok) / total_elapsed_s, 3) if total_elapsed_s else 0,
        "audio_duration_s": round(audio_ms / 1000, 3),
        "audio_s_per_s": round((audio_ms / 1000) / total_elapsed_s, 3) if total_elapsed_s else 0,
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p90_ms": round(percentile(latencies, 90), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0,
        "avg_rtf": round(statistics.mean(rtfs), 4) if rtfs else 0,
        "p50_rtf": round(percentile(rtfs, 50), 4),
        "p95_rtf": round(percentile(rtfs, 95), 4),
        "p99_rtf": round(percentile(rtfs, 99), 4),
        "max_rtf": round(max(rtfs), 4) if rtfs else 0,
    }


def summarize_omnivoice(results: list[OmniVoiceResult], total_elapsed_s: float) -> dict[str, Any]:
    ok = [result for result in results if result.ok]
    latencies = [result.elapsed_ms for result in ok]
    submit_latencies = [result.submit_ms for result in results if result.submit_ms > 0]
    rtfs = [result.rtf for result in ok if result.rtf > 0]
    audio_ms = sum(result.audio_duration_ms for result in ok)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[str(result.status_code)] = status_counts.get(str(result.status_code), 0) + 1
    return {
        "requests": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "status_counts": status_counts,
        "total_elapsed_s": round(total_elapsed_s, 3),
        "requests_per_s": round(len(ok) / total_elapsed_s, 3) if total_elapsed_s else 0,
        "audio_duration_s": round(audio_ms / 1000, 3),
        "audio_s_per_s": round((audio_ms / 1000) / total_elapsed_s, 3) if total_elapsed_s else 0,
        "avg_submit_ms": round(statistics.mean(submit_latencies), 2) if submit_latencies else 0,
        "p95_submit_ms": round(percentile(submit_latencies, 95), 2),
        "chunks_total": sum(result.chunks_total for result in results),
        "chunks_completed": sum(result.chunks_completed for result in results),
        "chunks_failed": sum(result.chunks_failed for result in results),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p90_ms": round(percentile(latencies, 90), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0,
        "avg_rtf": round(statistics.mean(rtfs), 4) if rtfs else 0,
        "p50_rtf": round(percentile(rtfs, 50), 4),
        "p95_rtf": round(percentile(rtfs, 95), 4),
        "p99_rtf": round(percentile(rtfs, 99), 4),
        "max_rtf": round(max(rtfs), 4) if rtfs else 0,
    }


def probe_audio_duration_ms(body: bytes, suffix: str) -> tuple[float, str]:
    if not body:
        return 0.0, "empty audio body"
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        tmp_path = Path(f"/tmp/mixed-tts-benchmark-{time.time_ns()}.{suffix}")
        try:
            tmp_path.write_bytes(body)
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            duration_s = float(completed.stdout.strip())
            return max(0.0, duration_s * 1000), ""
        except Exception as exc:
            return 0.0, f"ffprobe failed: {exc!r}"
        finally:
            tmp_path.unlink(missing_ok=True)
    try:
        import io
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(body)) as audio:
            return (audio.frames / audio.samplerate) * 1000, ""
    except Exception as exc:
        return 0.0, f"soundfile failed: {exc!r}"


def split_text_for_tts(text: str, target_chars: int = 200) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= target_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def parse_omnivoice_audio_duration_ms(body: bytes, suffix: str, content_type: str) -> tuple[float, str]:
    if "octet-stream" not in content_type:
        return probe_audio_duration_ms(body, suffix)
    try:
        if len(body) < 4:
            return 0.0, "short length-prefixed body"
        chunk_count = struct.unpack(">I", body[0:4])[0]
        offset = 4
        total = 0.0
        for _ in range(chunk_count):
            if len(body) - offset < 4:
                return total, "truncated chunk length"
            size = struct.unpack(">I", body[offset:offset + 4])[0]
            offset += 4
            if len(body) - offset < size:
                return total, "truncated chunk data"
            dur_ms, _ = probe_audio_duration_ms(body[offset:offset + size], suffix)
            total += dur_ms
            offset += size
        return total, ""
    except Exception as exc:
        return 0.0, f"length-prefixed parse failed: {exc!r}"


def gpu_query() -> dict[str, float]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        util, mem, power = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return {"util_pct": float(util), "memory_mib": float(mem), "power_w": float(power)}
    except Exception:
        return {"util_pct": 0.0, "memory_mib": 0.0, "power_w": 0.0}


async def sample_gpu(stop: asyncio.Event, samples: list[dict[str, float]], interval_s: float) -> None:
    while not stop.is_set():
        samples.append(gpu_query())
        await asyncio.sleep(interval_s)


def summarize_gpu(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    return {
        "samples": len(samples),
        "avg_util_pct": round(statistics.mean(s["util_pct"] for s in samples), 2),
        "max_util_pct": round(max(s["util_pct"] for s in samples), 2),
        "avg_memory_mib": round(statistics.mean(s["memory_mib"] for s in samples), 2),
        "max_memory_mib": round(max(s["memory_mib"] for s in samples), 2),
        "avg_power_w": round(statistics.mean(s["power_w"] for s in samples), 2),
        "max_power_w": round(max(s["power_w"] for s in samples), 2),
    }


def load_kokoro_pipeline(lang_code: str, voice: str, device: str):
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=device)
    list(pipeline("Warmup sentence for Kokoro.", voice=voice, speed=1.0, split_pattern=None))
    return pipeline


def run_kokoro_one(pipeline: Any, text: str, voice: str, lang_code: str, index: int) -> KokoroResult:
    started = time.perf_counter()
    try:
        import torch

        with torch.inference_mode():
            outputs = list(pipeline(text, voice=voice, speed=1.0, split_pattern=None))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        samples = 0
        for output in outputs:
            audio = output.audio
            samples += int(audio.shape[-1] if hasattr(audio, "shape") else len(audio))
        audio_duration_ms = (samples / SAMPLE_RATE) * 1000
        return KokoroResult(
            index=index,
            lang_code=lang_code,
            voice=voice,
            ok=True,
            elapsed_ms=elapsed_ms,
            audio_duration_ms=audio_duration_ms,
            rtf=elapsed_ms / audio_duration_ms if audio_duration_ms else 0,
            chars=len(text),
        )
    except Exception as exc:
        return KokoroResult(
            index=index,
            lang_code=lang_code,
            voice=voice,
            ok=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            audio_duration_ms=0,
            rtf=0,
            chars=len(text),
            error=repr(exc),
        )


async def run_kokoro_load(args: argparse.Namespace) -> tuple[dict[str, Any], list[KokoroResult]]:
    voice, _ = KOKORO_LANGUAGE_CASES[args.lang_code]
    if args.voice:
        voice = args.voice
    text = args.text
    pipeline = load_kokoro_pipeline(args.lang_code, voice, args.device)
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            loop.run_in_executor(executor, run_kokoro_one, pipeline, text, voice, args.lang_code, index)
            for index in range(args.requests)
        ]
        results = await asyncio.gather(*futures)
    total_elapsed = time.perf_counter() - started
    return summarize_kokoro(list(results), total_elapsed), list(results)


async def run_kokoro_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from kokoro import KModel
    import torch

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(args.device).eval()
    results: list[KokoroResult] = []
    started = time.perf_counter()
    for index, (lang_code, (voice, text)) in enumerate(KOKORO_LANGUAGE_CASES.items()):
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", model=model)
        results.append(run_kokoro_one(pipeline, text, voice, lang_code, index))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - started
    return {
        "summary": summarize_kokoro(results, total_elapsed),
        "results": [asdict(result) for result in results],
    }


async def run_omnivoice_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    index: int,
    args: argparse.Namespace,
) -> OmniVoiceResult:
    headers = {"Authorization": f"Bearer {args.token}", "Content-Type": "application/json"}
    payload = {
        "chunks": split_text_for_tts(args.omnivoice_text, args.omnivoice_chunk_size_chars),
        "ref_audio_url": args.ref_audio_url,
        "ref_text": args.ref_text,
        "language": args.omnivoice_language,
        "num_step": args.num_step,
        "format": args.format,
    }
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(f"{args.base_url.rstrip('/')}/v1/tts", headers=headers, json=payload)
            submit_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 202:
                return OmniVoiceResult(
                    index=index,
                    ok=False,
                    status_code=response.status_code,
                    submit_ms=submit_ms,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    audio_duration_ms=0,
                    rtf=0,
                    chunks_total=0,
                    chunks_completed=0,
                    chunks_failed=0,
                    size_bytes=len(response.content),
                    request_id=response.headers.get("x-request-id", ""),
                    error=response.text[:500],
                )
            job = response.json()
            request_id = str(job["request_id"])
            chunks_total = int(job.get("chunks_total") or 0)
            chunks_completed = int(job.get("chunks_completed") or 0)
            chunks_failed = int(job.get("chunks_failed") or 0)
            status_url = f"{args.base_url.rstrip('/')}{job['status_url']}"
            audio_url = ""
            while True:
                if time.perf_counter() - started > args.timeout:
                    return OmniVoiceResult(
                        index=index,
                        ok=False,
                        status_code=0,
                        submit_ms=submit_ms,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        audio_duration_ms=0,
                        rtf=0,
                        chunks_total=chunks_total,
                        chunks_completed=chunks_completed,
                        chunks_failed=chunks_failed,
                        size_bytes=0,
                        request_id=request_id,
                        error="timeout",
                    )
                await asyncio.sleep(args.poll_interval)
                status_response = await client.get(status_url, headers=headers)
                status_response.raise_for_status()
                status = status_response.json()
                chunks_total = int(status.get("chunks_total") or chunks_total)
                chunks_completed = int(status.get("chunks_completed") or chunks_completed)
                chunks_failed = int(status.get("chunks_failed") or chunks_failed)
                if status["status"] == "succeeded":
                    audio_url = f"{args.base_url.rstrip('/')}{status['audio_url']}"
                    break
                if status["status"] == "failed":
                    return OmniVoiceResult(
                        index=index,
                        ok=False,
                        status_code=500,
                        submit_ms=submit_ms,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        audio_duration_ms=0,
                        rtf=0,
                        chunks_total=chunks_total,
                        chunks_completed=chunks_completed,
                        chunks_failed=chunks_failed,
                        size_bytes=0,
                        request_id=request_id,
                        error=str(status.get("detail", "failed")),
                    )
            audio_response = await client.get(audio_url, headers=headers)
            elapsed_ms = (time.perf_counter() - started) * 1000
            audio_duration_ms, duration_error = parse_omnivoice_audio_duration_ms(
                audio_response.content,
                args.format,
                audio_response.headers.get("Content-Type", ""),
            )
            ok = audio_response.status_code == 200
            return OmniVoiceResult(
                index=index,
                ok=ok,
                status_code=audio_response.status_code,
                submit_ms=submit_ms,
                elapsed_ms=elapsed_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=elapsed_ms / audio_duration_ms if audio_duration_ms else 0,
                chunks_total=chunks_total,
                chunks_completed=chunks_completed,
                chunks_failed=chunks_failed,
                size_bytes=len(audio_response.content),
                request_id=request_id,
                error="" if ok else audio_response.text[:500] or duration_error,
            )
        except Exception as exc:
            return OmniVoiceResult(
                index=index,
                ok=False,
                status_code=0,
                submit_ms=0,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                audio_duration_ms=0,
                rtf=0,
                chunks_total=0,
                chunks_completed=0,
                chunks_failed=0,
                size_bytes=0,
                request_id="",
                error=repr(exc),
            )


async def run_omnivoice_load(args: argparse.Namespace) -> tuple[dict[str, Any], list[OmniVoiceResult]]:
    semaphore = asyncio.Semaphore(args.omnivoice_concurrency)
    timeout = httpx.Timeout(args.timeout)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        results = await asyncio.gather(
            *[
                run_omnivoice_one(client, semaphore, index, args)
                for index in range(args.omnivoice_requests)
            ]
        )
    total_elapsed = time.perf_counter() - started
    return summarize_omnivoice(list(results), total_elapsed), list(results)


async def run_mixed(args: argparse.Namespace) -> dict[str, Any]:
    kokoro_args = argparse.Namespace(
        lang_code=args.lang_code,
        voice=args.voice,
        text=args.kokoro_text,
        device=args.device,
        concurrency=args.kokoro_concurrency,
        requests=args.kokoro_requests,
    )
    stop_gpu = asyncio.Event()
    gpu_samples: list[dict[str, float]] = []
    sampler = asyncio.create_task(sample_gpu(stop_gpu, gpu_samples, args.gpu_sample_interval))
    started = time.perf_counter()
    kokoro_task = asyncio.create_task(run_kokoro_load(kokoro_args))
    omnivoice_task = asyncio.create_task(run_omnivoice_load(args))
    kokoro_summary, kokoro_results = await kokoro_task
    omnivoice_summary, omnivoice_results = await omnivoice_task
    total_elapsed = time.perf_counter() - started
    stop_gpu.set()
    await sampler
    return {
        "metadata": {
            "mode": "mixed",
            "kokoro_concurrency": args.kokoro_concurrency,
            "kokoro_requests": args.kokoro_requests,
            "omnivoice_concurrency": args.omnivoice_concurrency,
            "omnivoice_requests": args.omnivoice_requests,
            "omnivoice_chunks_per_request": len(
                split_text_for_tts(args.omnivoice_text, args.omnivoice_chunk_size_chars)
            ),
            "omnivoice_chunk_size_chars": args.omnivoice_chunk_size_chars,
            "num_step": args.num_step,
            "format": args.format,
        },
        "total_elapsed_s": round(total_elapsed, 3),
        "kokoro": {
            "summary": kokoro_summary,
            "results": [asdict(result) for result in kokoro_results],
        },
        "omnivoice": {
            "summary": omnivoice_summary,
            "results": [asdict(result) for result in omnivoice_results],
        },
        "gpu": summarize_gpu(gpu_samples),
    }


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.results_csv:
        rows = []
        if "results" in payload:
            rows = payload["results"]
        elif "kokoro" in payload:
            rows = [
                {"kind": "kokoro", **row}
                for row in payload["kokoro"]["results"]
            ] + [
                {"kind": "omnivoice", **row}
                for row in payload["omnivoice"]["results"]
            ]
        if rows:
            Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with Path(args.results_csv).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def add_common_kokoro_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lang-code", default="a", choices=sorted(KOKORO_LANGUAGE_CASES))
    parser.add_argument("--voice", default="")
    parser.add_argument(
        "--text",
        default="Kokoro benchmark sentence. The text is short enough to represent interactive TTS traffic.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=80)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-json", default="")
    parser.add_argument("--results-csv", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Kokoro and mixed Kokoro plus OmniVoice workloads.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--device", default="cuda")
    add_output_args(smoke)

    kokoro = subparsers.add_parser("kokoro")
    add_common_kokoro_args(kokoro)
    add_output_args(kokoro)

    mixed = subparsers.add_parser("mixed")
    mixed.add_argument("--base-url", required=True)
    mixed.add_argument("--token", required=True)
    mixed.add_argument("--ref-audio-url", required=True)
    mixed.add_argument("--ref-text", required=True)
    mixed.add_argument("--omnivoice-text", default="Xin chao, day la benchmark OmniVoice tren RTX 5090.")
    mixed.add_argument("--omnivoice-text-file", default="")
    mixed.add_argument("--omnivoice-chunk-size-chars", type=int, default=200)
    mixed.add_argument("--omnivoice-language", default="")
    mixed.add_argument("--omnivoice-concurrency", type=int, default=8)
    mixed.add_argument("--omnivoice-requests", type=int, default=32)
    mixed.add_argument("--kokoro-text", default="Kokoro benchmark sentence for concurrent GPU synthesis.")
    mixed.add_argument("--kokoro-concurrency", type=int, default=8)
    mixed.add_argument("--kokoro-requests", type=int, default=96)
    mixed.add_argument("--lang-code", default="a", choices=sorted(KOKORO_LANGUAGE_CASES))
    mixed.add_argument("--voice", default="")
    mixed.add_argument("--device", default="cuda")
    mixed.add_argument("--num-step", type=int, default=32)
    mixed.add_argument("--format", choices=["wav", "mp3"], default="wav")
    mixed.add_argument("--poll-interval", type=float, default=0.1)
    mixed.add_argument("--timeout", type=float, default=300)
    mixed.add_argument("--gpu-sample-interval", type=float, default=0.5)
    add_output_args(mixed)
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "smoke":
        payload = await run_kokoro_smoke(args)
        write_outputs(payload, args)
        return 0 if payload["summary"]["failed"] == 0 else 1
    if args.command == "kokoro":
        summary, results = await run_kokoro_load(args)
        payload = {"summary": summary, "results": [asdict(result) for result in results]}
        write_outputs(payload, args)
        return 0 if summary["failed"] == 0 else 1
    if args.command == "mixed":
        payload = await run_mixed(args)
        write_outputs(payload, args)
        return 0 if payload["kokoro"]["summary"]["failed"] == 0 and payload["omnivoice"]["summary"]["failed"] == 0 else 1
    raise AssertionError(args.command)


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "omnivoice_text_file", ""):
        args.omnivoice_text = Path(args.omnivoice_text_file).read_text(encoding="utf-8")
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
