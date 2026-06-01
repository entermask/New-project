# OmniVoice Pronunciation Comparison

- created_at: `20260530T145847Z`
- old_endpoint: `https://u8825-92ef-d72b27a8.singapore-b.gpuhub.com:8443`
- old_model: `k2-fsa/OmniVoice`
- old_acceleration: `triton`
- new_endpoint: `http://127.0.0.1:6006`
- new_model: `kjanh/KhanhTTS-OmniVoice`
- new_acceleration: `triton`
- request_language: `auto`
- ref_audio_url: `https://persist.cdn.ai33.pro/samples/edge_sample_vi-VN-HoaiMyNeural.mp3`
- asr_proxy_model: `openai/whisper-large-v3-turbo`

ASR CER is only a proxy for pronunciation. Listen to the paired MP3 files before making a product decision.

| Case | Text | Old audio | New audio | Old CER | New CER |
| --- | --- | --- | --- | ---: | ---: |
| vi_diacritics | Tôi muốn kiểm tra phát âm tiếng Việt với các từ khó: Nguyễn, Nghĩa, khuỷu tay, xoắn xuýt, loanh quanh, khuya khoắt, rượu, được, trường, lượng, chuyển, chuyện và quyển sách. | [vi_diacritics.old.mp3](vi_diacritics.old.mp3) | [vi_diacritics.new.mp3](vi_diacritics.new.mp3) | 0.031 | 0.050 |
| vi_initials_tones | Câu này dùng để nghe rõ ch, tr, s, x, r, d, gi, cùng dấu hỏi và dấu ngã: rõ ràng, dữ dội, giữ gìn, sửa soạn, xa xôi, tranh chấp, chỉn chu. | [vi_initials_tones.old.mp3](vi_initials_tones.old.mp3) | [vi_initials_tones.new.mp3](vi_initials_tones.new.mp3) | 0.114 | 0.220 |
| vi_mixed_terms | AI ba mươi ba Pro đang thử Khanh TTS OmniVoice trên Triton. Mục tiêu là giọng đọc tiếng Việt tự nhiên hơn, ít nuốt âm cuối và không lệch dấu. | [vi_mixed_terms.old.mp3](vi_mixed_terms.old.mp3) | [vi_mixed_terms.new.mp3](vi_mixed_terms.new.mp3) | 0.080 | 0.080 |

## ASR Transcripts

### vi_diacritics

- target: Tôi muốn kiểm tra phát âm tiếng Việt với các từ khó: Nguyễn, Nghĩa, khuỷu tay, xoắn xuýt, loanh quanh, khuya khoắt, rượu, được, trường, lượng, chuyển, chuyện và quyển sách.
- old: Tôi muốn kiểm tra phát âm tiếng Việt với các từ khó, nguyễn, nghĩa, khủyểu tay, xoắn suýt, loanh quanh, khuya khoắc, rượu, được trường, lượng, chuyển, chuyện và quyển sách.
- new: Tôi muốn kiểm tra phát âm tiếng Việt với các từ khó, Nguyễn, Nghĩa, Khỉu Tây, Xoắn Xích, Loanh Quanh, Khuya Khuất, Rượu, Được, Trường, Lượng, Chuyển, Chuyện và Quyển Sách.

### vi_initials_tones

- target: Câu này dùng để nghe rõ ch, tr, s, x, r, d, gi, cùng dấu hỏi và dấu ngã: rõ ràng, dữ dội, giữ gìn, sửa soạn, xa xôi, tranh chấp, chỉn chu.
- old: Câu này dùng để nghe rõ chật, trơ S cùng dấu hỏi và dấu ngã, rõ ràng dữ dội, giữ gìn sửa soạn xa xôi tranh chấp dĩnh chu.
- new: G, S để nghe rõ, D, G, S cùng dấu hỏi và dấu ngã, rõ ràng, giữ dội, giữ gìn, sửa soạn, xa xôi, tranh chấp, chỉnh chu.

### vi_mixed_terms

- target: AI ba mươi ba Pro đang thử Khanh TTS OmniVoice trên Triton. Mục tiêu là giọng đọc tiếng Việt tự nhiên hơn, ít nuốt âm cuối và không lệch dấu.
- old: AI33 Pro đang thử khanh TTS OmniVoice trên Triton. Mục tiêu là giọng đọc tiếng Việt tự nhiên hơn, ít nuốt âm cuối và không lệch dấu.
- new: AI 33 Pro đang thử khanh TTS Omni Voice trên Triton. Mục tiêu là giọng đọc tiếng Việt tự nhiên hơn, ít nuốt âm cuối và không lệch dấu.
