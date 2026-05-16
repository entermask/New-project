#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import math
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx


@dataclass
class Result:
    index: int
    text_index: int
    ref_audio_index: int
    ref_audio_url: str
    status_code: int
    ok: bool
    submit_ms: float
    end_to_end_ms: float
    elapsed_ms: float
    audio_duration_ms: float
    rtf: float
    size_bytes: int
    cache_hit: str
    request_id: str
    transcript: str
    chunks_total: int
    chunks_completed: int
    chunks_failed: int
    error: str


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize(results: list[Result], total_elapsed: float) -> dict[str, Any]:
    ok_results = [result for result in results if result.ok]
    latencies = [result.end_to_end_ms for result in ok_results]
    submit_latencies = [result.submit_ms for result in results if result.submit_ms > 0]
    audio_durations = [result.audio_duration_ms for result in ok_results if result.audio_duration_ms > 0]
    rtfs = [result.rtf for result in ok_results if result.rtf > 0]
    total_bytes = sum(result.size_bytes for result in ok_results)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[str(result.status_code)] = status_counts.get(str(result.status_code), 0) + 1
    return {
        "requests": len(results),
        "ok": len(ok_results),
        "failed": len(results) - len(ok_results),
        "accepted": sum(1 for result in results if result.request_id),
        "rejected_429": sum(1 for result in results if result.status_code == 429),
        "non_429_failed": sum(1 for result in results if not result.ok and result.status_code != 429),
        "status_counts": status_counts,
        "total_elapsed_s": round(total_elapsed, 3),
        "requests_per_s": round(len(ok_results) / total_elapsed, 3) if total_elapsed else 0,
        "bytes_downloaded": total_bytes,
        "audio_duration_ms": round(sum(audio_durations), 2) if audio_durations else 0,
        "audio_duration_s": round(sum(audio_durations) / 1000, 3) if audio_durations else 0,
        "text_count": len({result.text_index for result in results}),
        "ref_audio_count": len({result.ref_audio_url for result in results}),
        "avg_submit_ms": round(statistics.mean(submit_latencies), 2) if submit_latencies else 0,
        "p95_submit_ms": round(percentile(submit_latencies, 95), 2),
        "chunks_total": sum(result.chunks_total for result in results),
        "chunks_completed": sum(result.chunks_completed for result in results),
        "chunks_failed": sum(result.chunks_failed for result in results),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "min_ms": round(min(latencies), 2) if latencies else 0,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p90_ms": round(percentile(latencies, 90), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0,
        "avg_rtf": round(statistics.mean(rtfs), 3) if rtfs else 0,
        "min_rtf": round(min(rtfs), 3) if rtfs else 0,
        "p50_rtf": round(percentile(rtfs, 50), 3),
        "p90_rtf": round(percentile(rtfs, 90), 3),
        "p95_rtf": round(percentile(rtfs, 95), 3),
        "p99_rtf": round(percentile(rtfs, 99), 3),
        "max_rtf": round(max(rtfs), 3) if rtfs else 0,
        "cache_hits": sum(1 for result in ok_results if result.cache_hit == "true"),
        "cache_misses": sum(1 for result in ok_results if result.cache_hit == "false"),
    }


def summarize_subset(results: list[Result]) -> dict[str, Any]:
    ok_results = [result for result in results if result.ok]
    latencies = [result.end_to_end_ms for result in ok_results]
    audio_durations = [result.audio_duration_ms for result in ok_results if result.audio_duration_ms > 0]
    rtfs = [result.rtf for result in ok_results if result.rtf > 0]
    return {
        "requests": len(results),
        "ok": len(ok_results),
        "failed": len(results) - len(ok_results),
        "accepted": sum(1 for result in results if result.request_id),
        "rejected_429": sum(1 for result in results if result.status_code == 429),
        "chunks_total": sum(result.chunks_total for result in results),
        "chunks_completed": sum(result.chunks_completed for result in results),
        "chunks_failed": sum(result.chunks_failed for result in results),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "min_ms": round(min(latencies), 2) if latencies else 0,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p90_ms": round(percentile(latencies, 90), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0,
        "audio_duration_ms": round(sum(audio_durations), 2) if audio_durations else 0,
        "audio_duration_s": round(sum(audio_durations) / 1000, 3) if audio_durations else 0,
        "avg_rtf": round(statistics.mean(rtfs), 3) if rtfs else 0,
        "p50_rtf": round(percentile(rtfs, 50), 3),
        "p95_rtf": round(percentile(rtfs, 95), 3),
        "cache_hits": sum(1 for result in ok_results if result.cache_hit == "true"),
        "cache_misses": sum(1 for result in ok_results if result.cache_hit == "false"),
    }


def summarize_by_ref_audio(results: list[Result]) -> list[dict[str, Any]]:
    groups: dict[int, list[Result]] = {}
    for result in results:
        groups.setdefault(result.ref_audio_index, []).append(result)
    summaries = []
    for ref_audio_index in sorted(groups):
        group = groups[ref_audio_index]
        summaries.append(
            {
                "ref_audio_index": ref_audio_index,
                "ref_audio_url": group[0].ref_audio_url,
                "summary": summarize_subset(group),
            }
        )
    return summaries


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        lines = [
            line.strip()
            for line in Path(args.text_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise SystemExit("--text-file has no non-empty lines")
        texts = lines
    else:
        texts = [args.text]
    if args.text_repeat < 1:
        raise SystemExit("--text-repeat must be >= 1")
    if args.text_repeat > 1:
        texts = [" ".join([text] * args.text_repeat) for text in texts]
    return texts


def set_query_param(url: str, name: str, value: int) -> str:
    parts = urlsplit(url)
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True) if key != name]
    query.append((name, str(value)))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def load_ref_audio_urls(args: argparse.Namespace) -> list[str]:
    base_urls = list(args.ref_audio_url or [])
    if args.ref_audio_url_file:
        base_urls.extend(
            line.strip()
            for line in Path(args.ref_audio_url_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not base_urls:
        raise SystemExit("At least one --ref-audio-url or --ref-audio-url-file entry is required")
    if args.ref_audio_variants < 1:
        raise SystemExit("--ref-audio-variants must be >= 1")

    urls = []
    for base_url in base_urls:
        if args.ref_audio_variants == 1:
            urls.append(base_url)
            continue
        for offset in range(args.ref_audio_variants):
            value = args.ref_audio_variant_start + offset
            urls.append(set_query_param(base_url, args.ref_audio_variant_param, value))
    return urls


def select_ref_audio_index(index: int, total_requests: int, ref_audio_count: int, strategy: str) -> int:
    if strategy == "round-robin":
        return index % ref_audio_count
    if strategy == "grouped":
        group_size = max(1, math.ceil(total_requests / ref_audio_count))
        return min(ref_audio_count - 1, index // group_size)
    raise ValueError(f"Unsupported ref audio selection strategy: {strategy}")


def estimate_chunks(text: str, chunk_size_chars: int) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / max(1, chunk_size_chars)))


def probe_audio_duration_ms(body: bytes, suffix: str) -> tuple[float, str]:
    if not body:
        return 0.0, "empty audio body"

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        tmp_path = Path(f"/tmp/omnivoice-benchmark-{time.time_ns()}.{suffix}")
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
            if duration_s > 0:
                return duration_s * 1000, ""
            return 0.0, f"ffprobe returned non-positive duration: {duration_s}"
        except Exception as exc:
            return 0.0, f"ffprobe duration probe failed: {exc!r}"
        finally:
            tmp_path.unlink(missing_ok=True)

    try:
        import io

        import soundfile as sf

        with sf.SoundFile(io.BytesIO(body)) as audio:
            if audio.samplerate > 0:
                return (audio.frames / audio.samplerate) * 1000, ""
        return 0.0, "soundfile returned non-positive samplerate"
    except Exception as exc:
        return 0.0, f"audio duration probe unavailable or failed: {exc!r}"


async def run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    index: int,
    text_index: int,
    ref_audio_index: int,
    ref_audio_url: str,
    url: str,
    base_url: str,
    token: str,
    payload: dict[str, Any],
    output_dir: Path | None,
    mode: str,
    poll_interval: float,
    timeout_seconds: float,
) -> Result:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(url, headers=headers, json=payload)
            submit_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 202:
                elapsed_ms = (time.perf_counter() - started) * 1000
                body = response.content
                return Result(
                    index=index,
                    text_index=text_index,
                    ref_audio_index=ref_audio_index,
                    ref_audio_url=ref_audio_url,
                    status_code=response.status_code,
                    ok=False,
                    submit_ms=submit_ms,
                    end_to_end_ms=elapsed_ms,
                    elapsed_ms=elapsed_ms,
                    audio_duration_ms=0,
                    rtf=0,
                    size_bytes=len(body),
                    cache_hit="",
                    request_id=response.headers.get("x-request-id", ""),
                    transcript="",
                    chunks_total=0,
                    chunks_completed=0,
                    chunks_failed=0,
                    error=body[:500].decode("utf-8", errors="replace"),
                )

            job = response.json()
            request_id = str(job["request_id"])
            status_url = base_url + str(job["status_url"])
            audio_url = ""
            cache_hit = ""
            transcript = ""
            chunks_total = int(job.get("chunks_total") or 0)
            chunks_completed = int(job.get("chunks_completed") or 0)
            chunks_failed = int(job.get("chunks_failed") or 0)
            while True:
                if time.perf_counter() - started > timeout_seconds:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    return Result(
                        index=index,
                        text_index=text_index,
                        ref_audio_index=ref_audio_index,
                        ref_audio_url=ref_audio_url,
                        status_code=0,
                        ok=False,
                        submit_ms=submit_ms,
                        end_to_end_ms=elapsed_ms,
                        elapsed_ms=elapsed_ms,
                        audio_duration_ms=0,
                        rtf=0,
                        size_bytes=0,
                        cache_hit=cache_hit,
                        request_id=request_id,
                        transcript=transcript,
                        chunks_total=chunks_total,
                        chunks_completed=chunks_completed,
                        chunks_failed=chunks_failed,
                        error="Timed out waiting for TTS job to finish",
                    )
                await asyncio.sleep(poll_interval)
                status_response = await client.get(status_url, headers=headers)
                status_response.raise_for_status()
                job = status_response.json()
                chunks_total = int(job.get("chunks_total") or chunks_total)
                chunks_completed = int(job.get("chunks_completed") or chunks_completed)
                chunks_failed = int(job.get("chunks_failed") or chunks_failed)
                if "cache_hit" in job:
                    cache_hit = str(job["cache_hit"]).lower()
                if job["status"] == "succeeded":
                    audio_url = base_url + str(job["audio_url"])
                    transcript = str(job.get("transcript", ""))
                    break
                if job["status"] == "failed":
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    return Result(
                        index=index,
                        text_index=text_index,
                        ref_audio_index=ref_audio_index,
                        ref_audio_url=ref_audio_url,
                        status_code=500,
                        ok=False,
                        submit_ms=submit_ms,
                        end_to_end_ms=elapsed_ms,
                        elapsed_ms=elapsed_ms,
                        audio_duration_ms=0,
                        rtf=0,
                        size_bytes=0,
                        cache_hit=cache_hit,
                        request_id=request_id,
                        transcript=transcript,
                        chunks_total=chunks_total,
                        chunks_completed=chunks_completed,
                        chunks_failed=chunks_failed,
                        error=str(job.get("detail", "TTS job failed")),
                    )

            response = await client.get(audio_url, headers=headers)
            elapsed_ms = (time.perf_counter() - started) * 1000
            body = response.content
            ok = response.status_code == 200
            cache_hit = response.headers.get("x-cache-hit", cache_hit)
            transcript = response.headers.get("x-transcript", transcript)

            audio_duration_ms = 0.0
            rtf = 0.0
            duration_error = ""
            if ok:
                audio_duration_ms, duration_error = probe_audio_duration_ms(body, payload["format"])
                if audio_duration_ms > 0:
                    rtf = elapsed_ms / audio_duration_ms

            if ok and output_dir:
                suffix = payload["format"]
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"{index:05d}.{suffix}").write_bytes(body)
            return Result(
                index=index,
                text_index=text_index,
                ref_audio_index=ref_audio_index,
                ref_audio_url=ref_audio_url,
                status_code=response.status_code,
                ok=ok,
                submit_ms=submit_ms,
                end_to_end_ms=elapsed_ms,
                elapsed_ms=elapsed_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=rtf,
                size_bytes=len(body),
                cache_hit=cache_hit,
                request_id=request_id,
                transcript=transcript,
                chunks_total=chunks_total,
                chunks_completed=chunks_completed,
                chunks_failed=chunks_failed,
                error=duration_error if ok and duration_error else ("" if ok else body[:500].decode("utf-8", errors="replace")),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return Result(
                index=index,
                text_index=text_index,
                ref_audio_index=ref_audio_index,
                ref_audio_url=ref_audio_url,
                status_code=0,
                ok=False,
                submit_ms=0,
                end_to_end_ms=elapsed_ms,
                elapsed_ms=elapsed_ms,
                audio_duration_ms=0,
                rtf=0,
                size_bytes=0,
                cache_hit="",
                request_id="",
                transcript="",
                chunks_total=0,
                chunks_completed=0,
                chunks_failed=0,
                error=repr(exc),
            )


async def run(args: argparse.Namespace) -> int:
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")
    texts = load_texts(args)
    ref_audio_urls = load_ref_audio_urls(args)
    base_url = args.base_url.rstrip("/")
    url = base_url + "/v1/tts"
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    output_dir = Path(args.output_dir) if args.output_dir else None
    requests = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for index in range(args.requests):
            text_index = index % len(texts)
            ref_audio_index = select_ref_audio_index(
                index,
                args.requests,
                len(ref_audio_urls),
                args.ref_audio_selection,
            )
            text = texts[text_index]
            ref_audio_url = ref_audio_urls[ref_audio_index]
            payload = {
                "text": text,
                "ref_audio_url": ref_audio_url,
                "num_step": args.num_step,
                "format": args.format,
            }
            if args.speed is not None:
                payload["speed"] = args.speed
            if args.language:
                payload["language"] = args.language
            if args.ref_text:
                payload["ref_text"] = args.ref_text
            requests.append(
                run_one(
                    client=client,
                    semaphore=semaphore,
                    index=index,
                    text_index=text_index,
                    ref_audio_index=ref_audio_index,
                    ref_audio_url=ref_audio_url,
                    url=url,
                    base_url=base_url,
                    token=args.token,
                    payload=payload,
                    output_dir=output_dir,
                    mode="poll",
                    poll_interval=args.poll_interval,
                    timeout_seconds=args.timeout,
                )
            )

        started = time.perf_counter()
        results = await asyncio.gather(*requests)
        total_elapsed = time.perf_counter() - started

    summary = summarize(results, total_elapsed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.results_json:
        Path(args.results_json).write_text(
            json.dumps(
                {
                    "summary": summary,
                    "metadata": {
                        "base_url": base_url,
                        "mode": "poll",
                        "format": args.format,
                        "num_step": args.num_step,
                        "speed": args.speed,
                        "language": args.language,
                        "text_count": len(texts),
                        "input_chars": [len(text.strip()) for text in texts],
                        "estimated_chunks": [
                            estimate_chunks(text, args.chunk_size_chars) for text in texts
                        ],
                        "chunk_size_chars": args.chunk_size_chars,
                        "text_repeat": args.text_repeat,
                        "concurrency": args.concurrency,
                        "requests": args.requests,
                        "ref_audio_selection": args.ref_audio_selection,
                        "ref_audio_urls": ref_audio_urls,
                    },
                    "ref_audio_summaries": summarize_by_ref_audio(results),
                    "results": [asdict(result) for result in results],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if args.results_csv:
        with Path(args.results_csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    blocking_failures = summary["failed"] - summary["rejected_429"]
    if args.fail_on_error and blocking_failures:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the OmniVoice /v1/tts API.")
    parser.add_argument("--base-url", required=True, help="Base API URL, e.g. https://id-8001.thundercompute.net")
    parser.add_argument("--token", required=True, help="Bearer API token")
    parser.add_argument(
        "--ref-audio-url",
        action="append",
        default=[],
        help="Reference audio URL. May be repeated.",
    )
    parser.add_argument("--ref-audio-url-file", default="", help="Optional file with one reference audio URL per line")
    parser.add_argument(
        "--ref-audio-variants",
        type=int,
        default=1,
        help="Create N cache-distinct variants per reference URL by setting a query parameter",
    )
    parser.add_argument(
        "--ref-audio-variant-param",
        default="a",
        help="Query parameter name used by --ref-audio-variants, e.g. a creates .mp3?a=1",
    )
    parser.add_argument(
        "--ref-audio-variant-start",
        type=int,
        default=1,
        help="First query parameter value for --ref-audio-variants",
    )
    parser.add_argument(
        "--ref-audio-selection",
        choices=["round-robin", "grouped"],
        default="round-robin",
        help="How requests are assigned to multiple reference audio URLs",
    )
    parser.add_argument("--ref-text", default="", help="Optional reference transcript")
    parser.add_argument("--text", default="Xin chào, đây là benchmark OmniVoice.", help="Text for all requests")
    parser.add_argument("--text-file", default="", help="Optional file with one text per line")
    parser.add_argument("--text-repeat", type=int, default=1, help="Repeat each text N times to benchmark long TTS jobs")
    parser.add_argument("--chunk-size-chars", type=int, default=200, help="Estimated server chunk size for benchmark metadata")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent HTTP requests")
    parser.add_argument("--language", default="", help="Optional language code, e.g. vi or de")
    parser.add_argument("--num-step", type=int, default=32, help="OmniVoice num_step")
    parser.add_argument("--speed", type=float, default=None, help="Optional speech speed factor, e.g. 1.1")
    parser.add_argument("--format", choices=["wav", "mp3"], default="mp3", help="Output format")
    parser.add_argument("--mode", choices=["poll"], default="poll", help="Use async polling jobs")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval seconds for --mode poll")
    parser.add_argument("--timeout", type=float, default=300, help="Per-request timeout seconds")
    parser.add_argument("--output-dir", default="", help="Optional directory to save returned audio")
    parser.add_argument("--results-json", default="", help="Optional JSON results output path")
    parser.add_argument("--results-csv", default="", help="Optional CSV results output path")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any non-429 request fails",
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
