#!/usr/bin/env python3
"""Generate side-by-side OmniVoice TTS pronunciation samples.

This is intended to compare the production/base OmniVoice bridge against a
candidate OmniVoice-compatible model using the same request contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


TOKEN = os.getenv("OMNIVOICE_COMPARE_TOKEN", "change-me")

OLD_ENDPOINTS = [
    "https://u8825-92ef-d72b27a8.singapore-b.gpuhub.com:8443",
    "https://u8825-9e60-7f86c67b.singapore-b.gpuhub.com:8443",
    "https://u8825-9e60-9bee4eec.singapore-b.gpuhub.com:8443",
    "https://u8825-bc35-cb4beb2e.singapore-b.gpuhub.com:8443",
    "https://u8825-ae64-5262d58e.singapore-b.gpuhub.com:8443",
    "https://u8825-ae64-179c1164.singapore-b.gpuhub.com:8443",
]

DEFAULT_NEW_ENDPOINT = "http://127.0.0.1:6006"

REF_AUDIO_URL = "https://persist.cdn.ai33.pro/samples/edge_sample_vi-VN-HoaiMyNeural.mp3"
REF_TEXT = (
    "Xin chào, tôi là trợ lý AI của bạn! Hãy cho tôi biết tôi có thể giúp gì "
    "để hiện thực hóa ý tưởng của bạn."
)

CASES = [
    {
        "id": "vi_diacritics",
        "text": (
            "Tôi muốn kiểm tra phát âm tiếng Việt với các từ khó: Nguyễn, Nghĩa, "
            "khuỷu tay, xoắn xuýt, loanh quanh, khuya khoắt, rượu, được, trường, "
            "lượng, chuyển, chuyện và quyển sách."
        ),
    },
    {
        "id": "vi_initials_tones",
        "text": (
            "Câu này dùng để nghe rõ ch, tr, s, x, r, d, gi, cùng dấu hỏi và dấu ngã: "
            "rõ ràng, dữ dội, giữ gìn, sửa soạn, xa xôi, tranh chấp, chỉn chu."
        ),
    },
    {
        "id": "vi_mixed_terms",
        "text": (
            "AI ba mươi ba Pro đang thử Khanh TTS OmniVoice trên Triton. Mục tiêu là "
            "giọng đọc tiếng Việt tự nhiên hơn, ít nuốt âm cuối và không lệch dấu."
        ),
    },
]


@dataclass
class Endpoint:
    label: str
    base_url: str
    verify_tls: bool


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def resolve_url(base_url: str, maybe_relative: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", maybe_relative)


def parse_audio_chunks(payload: bytes) -> list[bytes]:
    if len(payload) < 4:
        raise ValueError("audio payload is too small")
    offset = 0
    chunk_count = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    chunks: list[bytes] = []
    for index in range(chunk_count):
        if offset + 4 > len(payload):
            raise ValueError(f"unexpected end while reading chunk {index} size")
        size = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        if offset + size > len(payload):
            raise ValueError(f"unexpected end while reading chunk {index} body")
        chunks.append(payload[offset : offset + size])
        offset += size
    return chunks


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    expected_status: set[int] | None = None,
) -> dict[str, Any]:
    response = client.request(method, url, headers=headers(), json=json_body)
    expected_status = expected_status or {200}
    if response.status_code not in expected_status:
        raise RuntimeError(
            f"{method} {url} returned {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def health(endpoint: Endpoint) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=8, verify=endpoint.verify_tls) as client:
            response = client.get(resolve_url(endpoint.base_url, "/health"))
            if response.status_code != 200:
                return None
            payload = response.json()
            if payload.get("status") != "ok" or not payload.get("model_loaded"):
                return None
            return payload
    except Exception:
        return None


def sorted_healthy_old_endpoints(endpoints: list[Endpoint]) -> list[tuple[Endpoint, dict[str, Any]]]:
    candidates: list[tuple[int, int, Endpoint, dict[str, Any]]] = []
    for endpoint in endpoints:
        payload = health(endpoint)
        if not payload:
            continue
        queued = int(payload.get("queued_chunks") or 0)
        active = int(payload.get("active_generations") or 0)
        candidates.append((queued, active, endpoint, payload))
    if not candidates:
        raise RuntimeError("no healthy old OmniVoice endpoints found")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(endpoint, payload) for _, _, endpoint, payload in candidates]


def pick_old_endpoint(endpoints: list[Endpoint]) -> tuple[Endpoint, dict[str, Any]]:
    return sorted_healthy_old_endpoints(endpoints)[0]


def request_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return None
    return language.strip()


def create_tts_job(
    endpoint: Endpoint,
    text: str,
    *,
    timeout: float,
    language: str | None,
) -> tuple[dict[str, Any], bytes]:
    payload = {
        "chunks": [text],
        "ref_audio_url": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "num_step": 16,
        "format": "mp3",
        "guidance_scale": 2,
    }
    language_value = request_language(language)
    if language_value:
        payload["language"] = language_value
    with httpx.Client(timeout=timeout, verify=endpoint.verify_tls) as client:
        created = request_json(
            client,
            "POST",
            resolve_url(endpoint.base_url, "/v1/tts"),
            json_body=payload,
            expected_status={202},
        )
        status_url = resolve_url(endpoint.base_url, str(created["status_url"]))
        deadline = time.time() + timeout
        last_status: dict[str, Any] = created
        while time.time() < deadline:
            last_status = request_json(client, "GET", status_url)
            status = last_status.get("status")
            if status == "succeeded":
                audio_url = resolve_url(endpoint.base_url, str(last_status["audio_url"]))
                audio_response = client.get(audio_url, headers=headers())
                if audio_response.status_code != 200:
                    raise RuntimeError(
                        f"GET {audio_url} returned {audio_response.status_code}: "
                        f"{audio_response.text[:500]}"
                    )
                chunks = parse_audio_chunks(audio_response.content)
                if len(chunks) != 1:
                    raise RuntimeError(f"expected 1 audio chunk, got {len(chunks)}")
                return last_status, chunks[0]
            if status == "failed":
                raise RuntimeError(f"TTS job failed: {json.dumps(last_status, ensure_ascii=False)}")
            time.sleep(2)
    raise TimeoutError(f"TTS job timed out: {json.dumps(last_status, ensure_ascii=False)}")


def create_old_tts_job(
    endpoints: list[Endpoint],
    text: str,
    *,
    timeout: float,
    language: str | None,
) -> tuple[Endpoint, dict[str, Any], dict[str, Any], bytes]:
    errors: list[str] = []
    for endpoint, endpoint_health in sorted_healthy_old_endpoints(endpoints):
        try:
            status, audio = create_tts_job(endpoint, text, timeout=timeout, language=language)
            return endpoint, endpoint_health, status, audio
        except Exception as exc:
            errors.append(f"{endpoint.base_url}: {exc}")
            continue
    raise RuntimeError("all old OmniVoice endpoints failed:\n" + "\n".join(errors))


def ffprobe_duration(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(completed.stdout.strip())
    except Exception:
        return None


def normalize_transcript(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if lch == rch else 1),
                )
            )
        previous = current
    return previous[-1]


def error_rate(expected: str, actual: str) -> float:
    expected_norm = normalize_transcript(expected)
    actual_norm = normalize_transcript(actual)
    if not expected_norm:
        return 0.0 if not actual_norm else 1.0
    return edit_distance(expected_norm, actual_norm) / max(1, len(expected_norm))


def transcribe(paths: list[Path], model_name: str) -> dict[str, str]:
    import torch
    import librosa
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=dtype,
        device=device,
    )

    results: dict[str, str] = {}
    for path in paths:
        audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
        prediction = pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
        results[str(path)] = str(prediction.get("text", "")).strip()
    return results


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    asr_model: str | None,
) -> None:
    lines = [
        "# OmniVoice Pronunciation Comparison",
        "",
        f"- created_at: `{manifest['created_at']}`",
        f"- old_endpoint: `{manifest['old']['base_url']}`",
        f"- old_model: `{manifest['old']['health'].get('model')}`",
        f"- old_acceleration: `{manifest['old']['health'].get('acceleration')}`",
        f"- new_endpoint: `{manifest['new']['base_url']}`",
        f"- new_model: `{manifest['new']['health'].get('model')}`",
        f"- new_acceleration: `{manifest['new']['health'].get('acceleration')}`",
        f"- request_language: `{manifest.get('request_language', 'unknown')}`",
        f"- ref_audio_url: `{REF_AUDIO_URL}`",
        f"- asr_proxy_model: `{asr_model or 'not-run'}`",
        "",
        "ASR CER is only a proxy for pronunciation. Listen to the paired MP3 files before making a product decision.",
        "",
        "| Case | Text | Old audio | New audio | Old CER | New CER |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]

    for case in manifest["cases"]:
        old = case["outputs"]["old"]
        new = case["outputs"]["new"]
        old_cer = old.get("cer")
        new_cer = new.get("cer")
        lines.append(
            "| {case_id} | {text} | [{old_name}]({old_name}) | [{new_name}]({new_name}) | {old_cer} | {new_cer} |".format(
                case_id=case["id"],
                text=case["text"].replace("|", "\\|"),
                old_name=Path(old["path"]).name,
                new_name=Path(new["path"]).name,
                old_cer=f"{old_cer:.3f}" if isinstance(old_cer, float) else "",
                new_cer=f"{new_cer:.3f}" if isinstance(new_cer, float) else "",
            )
        )

    if asr_model:
        lines.extend(["", "## ASR Transcripts", ""])
        for case in manifest["cases"]:
            lines.append(f"### {case['id']}")
            lines.append("")
            lines.append(f"- target: {case['text']}")
            lines.append(f"- old: {case['outputs']['old'].get('asr_text', '')}")
            lines.append(f"- new: {case['outputs']['new'].get('asr_text', '')}")
            lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-endpoint", default=DEFAULT_NEW_ENDPOINT)
    parser.add_argument("--old-endpoint", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"pronunciation_comparison/{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_candidates = [
        Endpoint("old", value.rstrip("/"), verify_tls=False)
        for value in (args.old_endpoint or OLD_ENDPOINTS)
    ]
    old_endpoint, old_health = pick_old_endpoint(old_candidates)
    new_endpoint = Endpoint("new", args.new_endpoint.rstrip("/"), verify_tls=True)
    new_health = health(new_endpoint)
    if not new_health:
        raise RuntimeError(f"new endpoint is not healthy: {new_endpoint.base_url}")

    manifest: dict[str, Any] = {
        "created_at": timestamp,
        "request_language": request_language(args.language) or "auto",
        "ref_audio_url": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "old": {"base_url": old_endpoint.base_url, "health": old_health},
        "new": {"base_url": new_endpoint.base_url, "health": new_health},
        "cases": [],
    }

    audio_paths: list[Path] = []
    for case in CASES:
        print(f"Generating {case['id']} on old bridge", flush=True)
        used_old_endpoint, used_old_health, old_status, old_audio = create_old_tts_job(
            old_candidates,
            case["text"],
            timeout=args.timeout,
            language=args.language,
        )
        print(f"Generating {case['id']} on new: {new_endpoint.base_url}", flush=True)
        new_status, new_audio = create_tts_job(
            new_endpoint,
            case["text"],
            timeout=args.timeout,
            language=args.language,
        )

        old_path = output_dir / f"{case['id']}.old.mp3"
        new_path = output_dir / f"{case['id']}.new.mp3"
        old_path.write_bytes(old_audio)
        new_path.write_bytes(new_audio)
        audio_paths.extend([old_path, new_path])

        manifest["cases"].append(
            {
                **case,
                "outputs": {
                    "old": {
                        "path": str(old_path),
                        "endpoint": used_old_endpoint.base_url,
                        "health_before_request": used_old_health,
                        "bytes": len(old_audio),
                        "duration_seconds": ffprobe_duration(old_path),
                        "status": old_status,
                    },
                    "new": {
                        "path": str(new_path),
                        "bytes": len(new_audio),
                        "duration_seconds": ffprobe_duration(new_path),
                        "status": new_status,
                    },
                },
            }
        )

    asr_model = args.asr_model if args.transcribe else None
    if asr_model:
        print(f"Transcribing generated audio with {asr_model}", flush=True)
        transcripts = transcribe(audio_paths, asr_model)
        for case in manifest["cases"]:
            for label in ["old", "new"]:
                output = case["outputs"][label]
                asr_text = transcripts.get(output["path"], "")
                output["asr_text"] = asr_text
                output["cer"] = error_rate(case["text"], asr_text)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, manifest, asr_model=asr_model)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
