# Migration Guide: v1/tts Chunks API

> **Breaking change** — `POST /v1/tts` no longer accepts `text`. Client must send pre-split `chunks[]`.

## What Changed

| Before | After |
|--------|-------|
| Client gửi `text` (string) | Client gửi `chunks` (array of strings) |
| Server tự split text thành chunks | **Client tự split** |
| Audio response: 1 merged file | Audio response: **length-prefixed binary** chứa từng chunk riêng |
| `Content-Type: audio/wav` or `audio/mpeg` | `Content-Type: application/octet-stream` |

---

## 1. POST /v1/tts — Request

### Before

```json
{
  "text": "Đoạn văn dài 500 ký tự cần đọc...",
  "ref_audio_url": "https://example.com/ref.wav",
  "format": "mp3",
  "num_step": 16
}
```

### After

```json
{
  "chunks": [
    "Đoạn văn thứ nhất, khoảng 150-200 ký tự.",
    "Đoạn văn thứ hai, tách tại ranh giới câu.",
    "Đoạn văn thứ ba."
  ],
  "ref_audio_url": "https://example.com/ref.wav",
  "format": "mp3",
  "num_step": 16
}
```

### Chunk splitting guidelines

- **Target: 150–250 ký tự/chunk** — sweet spot cho OmniVoice
- Split tại **ranh giới câu** (`.` `!` `?` `;`) — giữ ngữ nghĩa tự nhiên
- **Không split giữa từ**
- **Min ~80 ký tự** — quá ngắn model không ổn định

### Unchanged fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `ref_audio_url` | string | required | HTTP(S) URL to reference audio |
| `ref_text` | string? | null | Optional reference transcript |
| `language` | string? | null | e.g. `"vi"`, `"en"` |
| `num_step` | int | 32 | 4–64, lower = faster |
| `speed` | float? | null | 0.5–2.0 |
| `format` | string | `"wav"` | `"wav"` or `"mp3"` |

---

## 2. Job Status — Response

Unchanged. Poll `GET /v1/tts/jobs/{request_id}`:

```json
{
  "request_id": "abc-123",
  "status": "succeeded",
  "chunks_total": 3,
  "chunks_completed": 3,
  "chunks_failed": 0,
  "input_chars": 450,
  "format": "mp3",
  "audio_url": "/v1/tts/jobs/abc-123/audio",
  "transcript": "...",
  "cache_hit": true
}
```

---

## 3. GET /v1/tts/jobs/{request_id}/audio — Response

### Before

Binary file (wav/mp3). Parse trực tiếp.

### After

**Length-prefixed binary** — `Content-Type: application/octet-stream`

```
┌─────────────────────────────────────────────────┐
│ 4 bytes: chunk_count (uint32 big-endian)        │
├─────────────────────────────────────────────────┤
│ 4 bytes: chunk_0_size (uint32 big-endian)       │
│ chunk_0_size bytes: chunk_0 audio binary        │
├─────────────────────────────────────────────────┤
│ 4 bytes: chunk_1_size (uint32 big-endian)       │
│ chunk_1_size bytes: chunk_1 audio binary        │
├─────────────────────────────────────────────────┤
│ ...                                             │
└─────────────────────────────────────────────────┘
```

Mỗi chunk là một **file audio hoàn chỉnh** (wav hoặc mp3 tuỳ `format`).

### Response Headers

```http
Content-Type: application/octet-stream
X-Request-Id: abc-123
X-Chunks-Total: 3
X-Audio-Format: audio/mpeg
X-Cache-Hit: true
X-Transcript: url-encoded-transcript
X-Transcript-Encoding: urlencoded-utf8
```

---

## 4. Node.js Client Implementation

### Parse audio response

```javascript
/**
 * Parse length-prefixed binary response from /v1/tts/jobs/{id}/audio
 * @param {Response} response - fetch Response object
 * @returns {Buffer[]} Array of audio chunk Buffers
 */
async function parseChunkedAudio(response) {
  const buffer = Buffer.from(await response.arrayBuffer());
  let offset = 0;

  // First 4 bytes: number of chunks
  const chunkCount = buffer.readUInt32BE(offset);
  offset += 4;

  const chunks = [];
  for (let i = 0; i < chunkCount; i++) {
    // 4 bytes: size of this chunk
    const size = buffer.readUInt32BE(offset);
    offset += 4;

    // Read chunk audio data
    chunks.push(buffer.subarray(offset, offset + size));
    offset += size;
  }

  return chunks;
}
```

### Full flow example

```javascript
const BASE_URL = "http://your-server:8080";
const TOKEN = "your-api-token";
const headers = {
  "Authorization": `Bearer ${TOKEN}`,
  "Content-Type": "application/json",
};

// 1. Submit job with pre-split chunks
const submitRes = await fetch(`${BASE_URL}/v1/tts`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    chunks: [
      "Xin chào, đây là đoạn văn thứ nhất.",
      "Đây là đoạn thứ hai để kiểm tra.",
    ],
    ref_audio_url: "https://example.com/ref.mp3",
    format: "mp3",
    num_step: 16,
  }),
});
const job = await submitRes.json();
const requestId = job.request_id;

// 2. Poll until succeeded
let status;
do {
  await new Promise((r) => setTimeout(r, 500));
  const pollRes = await fetch(`${BASE_URL}/v1/tts/jobs/${requestId}`, { headers });
  status = await pollRes.json();
} while (status.status === "queued" || status.status === "running");

if (status.status !== "succeeded") {
  throw new Error(`TTS failed: ${status.detail}`);
}

// 3. Download and parse chunked audio
const audioRes = await fetch(`${BASE_URL}${status.audio_url}`, { headers });
const audioChunks = await parseChunkedAudio(audioRes);

// audioChunks[0] = Buffer of first chunk (complete mp3 file)
// audioChunks[1] = Buffer of second chunk (complete mp3 file)

// Save each chunk
for (let i = 0; i < audioChunks.length; i++) {
  fs.writeFileSync(`output_chunk_${i}.mp3`, audioChunks[i]);
}

// Or concatenate for playback
const combined = Buffer.concat(audioChunks);
fs.writeFileSync("output_combined.mp3", combined);
```

### Text splitting helper (recommended)

```javascript
/**
 * Split text into chunks suitable for OmniVoice TTS.
 * Target: 150-250 chars per chunk, split at sentence boundaries.
 */
function splitTextForTTS(text, targetChars = 200) {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  // Split at sentence boundaries
  const sentences = normalized.split(/(?<=[.!?。！？])\s+/);
  const chunks = [];
  let current = "";

  for (const sentence of sentences) {
    if (!current) {
      current = sentence;
    } else if (current.length + 1 + sentence.length <= targetChars) {
      current = `${current} ${sentence}`;
    } else {
      chunks.push(current);
      current = sentence;
    }
  }
  if (current) chunks.push(current);

  return chunks;
}

// Usage:
const text = "Đoạn văn dài 500 ký tự. Cần split trước khi gửi. Mỗi chunk 150-250 chars.";
const chunks = splitTextForTTS(text);
// → ["Đoạn văn dài 500 ký tự. Cần split trước khi gửi.", "Mỗi chunk 150-250 chars."]
```

---

## 5. Error Handling

| Status | Meaning |
|--------|---------|
| `400` | `chunks` trống hoặc thiếu |
| `401` | Token sai |
| `429` | Server busy — retry sau `Retry-After` header |
| `409` | Job chưa hoàn tất (poll audio trước khi succeeded) |
| `410` | Audio đã expired (quá `JOB_TTL_SECONDS`) |
