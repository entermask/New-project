# OmniVoice Adapter Comparison

- created_at: `20260530T153403Z`
- mode: `auto`
- old_model: `k2-fsa/OmniVoice`
- new_model: `k2-fsa/OmniVoice`
- new_acceleration: `base`
- new_adapters: `ar, fr, zh`
- asr_proxy_model: `openai/whisper-large-v3-turbo`

| Adapter | Old audio | New audio | New job adapter | Old CER | New CER |
| --- | --- | --- | --- | ---: | ---: |
| zh | [zh_lora.old.mp3](zh_lora.old.mp3) | [zh_lora.new.mp3](zh_lora.new.mp3) | zh | 0.128 | 0.043 |
| fr | [fr_lora.old.mp3](fr_lora.old.mp3) | [fr_lora.new.mp3](fr_lora.new.mp3) | fr | 0.000 | 0.000 |
| ar | [ar_lora.old.mp3](ar_lora.old.mp3) | [ar_lora.new.mp3](ar_lora.new.mp3) | ar | 0.026 | 0.026 |

## ASR Transcripts

### zh_lora

- target: 今天我们测试中文发音，重点听清楚智能语音、上海、知识、支持、持续学习，以及普通话的声调是否自然。
- old: 今天我们测试中文发音重点听清楚智能语音上海知识支持持续学习以及普通话的声调是否自然
- new: 今天我们测试中文发音重点听清楚智能语音、上海、知识、支持、持续学习以及普通话的声调是否自然

### fr_lora

- target: Bonjour, nous testons la prononciation française avec les mots aujourd'hui, très sérieux, génération vocale, réussite, réseau, utilisateur et précision.
- old: Bonjour, nous testons la prononciation française avec les mots aujourd'hui, très sérieux, génération vocale, réussite, réseau, utilisateur et précision.
- new: Bonjour, nous testons la prononciation française avec les mots aujourd'hui, très sérieux, génération vocale, réussite, réseau, utilisateur et précision.

### ar_lora

- target: نختبر اليوم النطق العربي في الجمل الطويلة، مع كلمات مثل الذكاء الاصطناعي، التجربة، المستخدم، الجودة، واللغة العربية الفصحى.
- old: نختبر اليوم النطق العربي في الجمل الطويلة مع كلمات مثل الذكاء الاصطناعي، التجربة المستخدم، الجودة، واللغة العربي كالفصحة.
- new: نختبر اليوم النطق العربي في الشتجمل الطويلة مع كلمات مثل الذكاء الاصطناعي، التجربة، المستخدم، الجودة، واللغة العربية الفصحة
