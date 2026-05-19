# 🎙️ Tài liệu Tích hợp API Speech-to-Text (Whisper Large V3)

Tài liệu này hướng dẫn cách kết nối và sử dụng dịch vụ **Speech-to-Text (STT)** dựa trên mô hình **Whisper Large V3 (CTranslate2 FP16)** kết hợp **Silero VAD** chạy trên hạ tầng GPU L40S.

---

## 🔒 Xác thực (Authentication)
Tất cả các API đều yêu cầu xác thực bằng Bearer Token thông qua Header:
* **Header**: `Authorization: Bearer <API_TOKEN>`

---

## 🚀 1. Gửi Yêu Cầu Nhận Dạng (`POST /v1/stt/transcribe`)

Gửi tệp tin âm thanh lên server để thực hiện dịch từ giọng nói thành văn bản. Hỗ trợ 2 chế độ: **Sync (Đồng bộ)** và **Async (Bất đồng bộ)**.

* **URL**: `/v1/stt/transcribe`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

### 📋 Tham số Form (Form-Data Parameters):

| Tham số | Kiểu | Bắt buộc | Mặc định | Mô tả |
| :--- | :--- | :---: | :---: | :--- |
| `file` | File | **Có** | - | Tệp tin âm thanh cần dịch (hỗ trợ `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, v.v.). |
| `mode` | String | Không | `"sync"` | Chế độ xử lý: <br>• `"sync"`: Trả kết quả ngay (phù hợp file ngắn < 30s).<br>• `"async"`: Trả về `job_id` lập tức để Polling (phù hợp file dài/nặng). |
| `language` | String | Không | - | Mã ngôn ngữ 2 chữ cái ISO (ví dụ: `"vi"`, `"en"`, `"de"`). Nếu bỏ trống, mô hình sẽ **tự động nhận diện ngôn ngữ** của audio. |
| `beam_size` | Integer | Không | `5` | Kích thước beam search giải mã. Số càng cao dịch càng chuẩn nhưng chậm hơn một chút (khuyên dùng 5). |
| `vad_filter` | Boolean | Không | `true` | Bộ lọc khoảng lặng Silero VAD. Loại bỏ tạp âm và chống tình trạng mô hình bị lặp từ (hallucinations). |
| `word_timestamps` | Boolean | Không | `false` | Bật tính năng trích xuất vị trí chi tiết của từng từ (Word-level timestamps). Khi bật sẽ trả về thêm trường `"words"` trong từng phân đoạn. |

---

### 🟢 1.1 Phản hồi Chế độ Sync (`mode=sync`)
Trả về kết quả trực tiếp sau khi xử lý xong.

* **Status Code**: `200 OK`
* **JSON Response Body** (Khi bật `word_timestamps=true`):
```json
{
  "status": "completed",
  "language": "de",
  "language_probability": 1.0,
  "duration_seconds": 10.785,
  "text": "Guten Tag, ich freue mich, Sie bei unseren Sprachdiensten unterstützen zu dürfen.",
  "segments": [
    {
      "start": 0.0,
      "end": 4.64,
      "text": " Guten Tag, ich freue mich, Sie tại unseren Sprachdiensten unterstützen zu dürfen.",
      "words": [
        {
          "start": 0.0,
          "end": 0.5,
          "word": "Guten",
          "probability": 0.99
        },
        {
          "start": 0.5,
          "end": 1.0,
          "word": "Tag,",
          "probability": 0.99
        }
      ]
    }
  ],
  "srt": "1\n00:00:00,000 --> 00:00:04,640\nGuten Tag, ich freue mich, Sie bei unseren Sprachdiensten unterstützen zu dürfen.\n"
}
```

---

### 🟡 1.2 Phản hồi Chế độ Async (`mode=async`)
Trả về `job_id` lập tức. Việc dịch sẽ diễn ra dưới nền.

* **Status Code**: `202 Accepted`
* **JSON Response Body**:
```json
{
  "job_id": "stt_job_ebc4433679bd",
  "status": "queued",
  "created_at": 1779106481.23
}
```

---

## 🔍 2. Kiểm tra Trạng thái Job Async (`GET /v1/stt/status/{job_id}`)

Sử dụng để Polling (thăm dò) trạng thái của các Job được tạo bằng chế độ Async.

* **URL**: `/v1/stt/status/{job_id}`
* **Method**: `GET`

### 🟢 Phản hồi (Responses):

#### A. Khi Job đang xử lý (`queued` hoặc `running`):
* **Status Code**: `200 OK`
```json
{
  "job_id": "stt_job_ebc4433679bd",
  "status": "running",
  "created_at": 1779106481.23,
  "updated_at": 1779106485.10
}
```

#### B. Khi Job đã hoàn thành (`completed`):
* **Status Code**: `200 OK`
```json
{
  "job_id": "stt_job_ebc4433679bd",
  "status": "completed",
  "created_at": 1779106481.23,
  "updated_at": 1779106492.35,
  "result": {
    "language": "de",
    "language_probability": 1.0,
    "duration_seconds": 10.785,
    "text": "Guten Tag, ich freue mich, Sie bei unseren Sprachdiensten unterstützen zu dürfen.",
    "segments": [
      {
        "start": 0.0,
        "end": 4.64,
        "text": " Guten Tag, ich freue mich, Sie bei unseren Sprachdiensten unterstützen zu dürfen."
      }
    ],
    "srt": "1\n00:00:00,000 --> 00:00:04,640\nGuten Tag, ich freue mich, Sie bei unseren Sprachdiensten unterstützen zu dürfen.\n"
  }
}
```

#### C. Khi Job bị lỗi (`failed`):
* **Status Code**: `200 OK`
```json
{
  "job_id": "stt_job_ebc4433679bd",
  "status": "failed",
  "created_at": 1779106481.23,
  "updated_at": 1779106482.11,
  "error": "Error details here..."
}
```

---

## 🧹 3. Dọn dẹp toàn bộ bộ nhớ Cache (`POST /v1/cache/clear`)

Yêu cầu server xóa sạch toàn bộ các file âm thanh tham chiếu (`ref-audio`), các tệp tin transcripts đã lưu, và giải phóng bộ nhớ cache trong RAM (`voice_prompt_cache`). Phù hợp khi bạn muốn giải phóng đĩa cứng hoặc làm sạch toàn bộ dữ liệu mẫu cũ.

* **URL**: `/v1/cache/clear`
* **Method**: `POST`

### 🟢 Phản hồi thành công:
* **Status Code**: `200 OK`
```json
{
  "status": "ok",
  "message": "All cached reference audios, transcripts, and memory prompts cleared successfully."
}
```

---

## 💻 Mã nguồn Mẫu Client (Client Code Examples)

### 1. Dùng cURL (Command Line)
**Chạy chế độ Sync:**
```bash
curl -X POST "http://<IP_OR_DOMAIN>:8001/v1/stt/transcribe" \
     -H "Authorization: Bearer <API_TOKEN>" \
     -F "file=@/path/to/your/audio.mp3" \
     -F "mode=sync" \
     -F "language=vi"
```

---

### 2. Dùng Node.js / Bun (TypeScript + Axios)
Mẫu code tự động xử lý Polling nếu chọn chế độ Async:

```typescript
import axios from 'axios';
import * as fs from 'fs';
import * as FormData from 'form-data';

const API_URL = 'http://localhost:8001';
const API_TOKEN = 'your-secret-token';

async function transcribeAudio(filePath: string, language?: string) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('mode', 'async'); // Dùng async cho file dài
  if (language) form.append('language', language);

  const headers = {
    ...form.getHeaders(),
    'Authorization': `Bearer ${API_TOKEN}`,
  };

  try {
    // 1. Gửi file lên
    console.log('🚀 Đang gửi file lên GPU Server...');
    const response = await axios.post(`${API_URL}/v1/stt/transcribe`, form, { headers });
    const { job_id, status } = response.data;
    console.log(`✅ Đã khởi tạo thành công Job: ${job_id} (Trạng thái: ${status})`);

    // 2. Polling kết quả
    return await pollJobStatus(job_id);
  } catch (error) {
    console.error('❌ Lỗi gửi STT:', error);
  }
}

async function pollJobStatus(jobId: string): Promise<any> {
  const headers = { 'Authorization': `Bearer ${API_TOKEN}` };
  
  while (true) {
    console.log(`🔍 Đang kiểm tra trạng thái Job: ${jobId}...`);
    const response = await axios.get(`${API_URL}/v1/stt/status/${jobId}`, { headers });
    const { status, result, error } = response.data;

    if (status === 'completed') {
      console.log('🎉 Nhận dạng thành công!');
      return result; // Trả về kết quả chứa text, segments và srt
    } else if (status === 'failed') {
      throw new Error(`Tác vụ thất bại: ${error}`);
    }

    // Chờ 3 giây trước khi kiểm tra lại
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

// Chạy thử
transcribeAudio('./demo.wav', 'vi').then((res) => {
  if (res) {
    console.log('📝 Văn bản gốc:', res.text);
    console.log('🎞️ Phụ đề SRT:', res.srt);
  }
});
```

---

### 3. Dùng Python
```python
import requests
import time

API_URL = "http://localhost:8001"
API_TOKEN = "your-secret-token"

def transcribe(file_path, language="vi"):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    files = {"file": open(file_path, "rb")}
    data = {
        "mode": "async",
        "language": language
    }
    
    # 1. Gửi file
    res = requests.post(f"{API_URL}/v1/stt/transcribe", headers=headers, files=files, data=data)
    res_data = res.json()
    job_id = res_data["job_id"]
    
    # 2. Polling
    while True:
        status_res = requests.get(f"{API_URL}/v1/stt/status/{job_id}", headers=headers)
        status_data = status_res.json()
        status = status_data["status"]
        
        if status == "completed":
            return status_data["result"]
        elif status == "failed":
            raise Exception(f"Job failed: {status_data.get('error')}")
            
        time.sleep(2)

result = transcribe("demo.mp3", "vi")
print("SRT Subtitles:\n", result["srt"])
```
