# OmniVoice Adapter Comparison

- created_at: `20260531T122554Z`
- suite: `extended`
- mode: `auto`
- old_model: `k2-fsa/OmniVoice`
- new_model: `k2-fsa/OmniVoice`
- new_acceleration: `base`
- new_adapters: `ar, fr, zh`
- asr_proxy_model: `openai/whisper-large-v3-turbo`

## Adapter Summary

| Adapter | Cases | Old Avg CER | New Avg CER | New Better | Same | New Worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 5 | 0.028 | 0.016 | 3 | 1 | 1 |
| fr | 5 | 0.124 | 0.120 | 1 | 3 | 1 |
| zh | 5 | 0.292 | 0.287 | 2 | 2 | 1 |

## Case Details

| Case | Adapter | Old audio | New audio | New job adapter | Old CER | New CER |
| --- | --- | --- | --- | --- | ---: | ---: |
| zh_basic | zh | [zh_basic.old.mp3](zh_basic.old.mp3) | [zh_basic.new.mp3](zh_basic.new.mp3) | zh | 0.043 | 0.043 |
| fr_basic | fr | [fr_basic.old.mp3](fr_basic.old.mp3) | [fr_basic.new.mp3](fr_basic.new.mp3) | fr | 0.000 | 0.000 |
| ar_basic | ar | [ar_basic.old.mp3](ar_basic.old.mp3) | [ar_basic.new.mp3](ar_basic.new.mp3) | ar | 0.043 | 0.009 |
| zh_sibilants_places | zh | [zh_sibilants_places.old.mp3](zh_sibilants_places.old.mp3) | [zh_sibilants_places.new.mp3](zh_sibilants_places.new.mp3) | zh | 0.705 | 0.705 |
| zh_numbers_dates | zh | [zh_numbers_dates.old.mp3](zh_numbers_dates.old.mp3) | [zh_numbers_dates.new.mp3](zh_numbers_dates.new.mp3) | zh | 0.571 | 0.619 |
| zh_technical | zh | [zh_technical.old.mp3](zh_technical.old.mp3) | [zh_technical.new.mp3](zh_technical.new.mp3) | zh | 0.116 | 0.070 |
| zh_questions | zh | [zh_questions.old.mp3](zh_questions.old.mp3) | [zh_questions.new.mp3](zh_questions.new.mp3) | zh | 0.024 | 0.000 |
| fr_nasal_liaison | fr | [fr_nasal_liaison.old.mp3](fr_nasal_liaison.old.mp3) | [fr_nasal_liaison.new.mp3](fr_nasal_liaison.new.mp3) | fr | 0.072 | 0.072 |
| fr_numbers_dates | fr | [fr_numbers_dates.old.mp3](fr_numbers_dates.old.mp3) | [fr_numbers_dates.new.mp3](fr_numbers_dates.new.mp3) | fr | 0.548 | 0.511 |
| fr_technical | fr | [fr_technical.old.mp3](fr_technical.old.mp3) | [fr_technical.new.mp3](fr_technical.new.mp3) | fr | 0.000 | 0.000 |
| fr_questions | fr | [fr_questions.old.mp3](fr_questions.old.mp3) | [fr_questions.new.mp3](fr_questions.new.mp3) | fr | 0.000 | 0.015 |
| ar_emphatic | ar | [ar_emphatic.old.mp3](ar_emphatic.old.mp3) | [ar_emphatic.new.mp3](ar_emphatic.new.mp3) | ar | 0.000 | 0.023 |
| ar_numbers_dates | ar | [ar_numbers_dates.old.mp3](ar_numbers_dates.old.mp3) | [ar_numbers_dates.new.mp3](ar_numbers_dates.new.mp3) | ar | 0.038 | 0.009 |
| ar_technical | ar | [ar_technical.old.mp3](ar_technical.old.mp3) | [ar_technical.new.mp3](ar_technical.new.mp3) | ar | 0.039 | 0.019 |
| ar_questions | ar | [ar_questions.old.mp3](ar_questions.old.mp3) | [ar_questions.new.mp3](ar_questions.new.mp3) | ar | 0.020 | 0.020 |

## ASR Transcripts

### zh_basic

- target: 今天我们测试中文发音，重点听清楚智能语音、上海、知识、支持、持续学习，以及普通话的声调是否自然。
- old: 今天我们测试中文发音重点听清楚智能语音、上海、知识、支持、持续学习以及普通话的声调是否自然
- new: 今天我们测试中文发音重点听清楚智能语音、上海、知识、支持、持续学习以及普通话的声调是否自然

### fr_basic

- target: Bonjour, nous testons la prononciation française avec les mots aujourd'hui, très sérieux, génération vocale, réussite, réseau, utilisateur et précision.
- old: Bonjour, nous testons la prononciation française avec les mots aujourd'hui, très sérieux, génération vocale, réussite, réseau, utilisateur et précision.
- new: Bonjour, nous testons la prononciation française avec les mots aujourd'hui, très sérieux, génération vocale, réussite, réseau, utilisateur et précision.

### ar_basic

- target: نختبر اليوم النطق العربي في الجمل الطويلة، مع كلمات مثل الذكاء الاصطناعي، التجربة، المستخدم، الجودة، واللغة العربية الفصحى.
- old: نختبر اليوم النطق العربي في الجمل الطويلة، مع كلمات مثل الذكاء الاصطناعي، التجربة، اسم استخدم، الجفدة، واللغة العربية الفصحة.
- new: نختبر اليوم النطق العربي في الجمل الطويلة مع كلمات مثل الذكاء الاصطناعي، التجربة، المستخدم، الجودة، واللغة العربية الفصحة.

### zh_sibilants_places

- target: 请连续读出这些容易混淆的词：知识、支持、事实、城市、十四、四十、出租车、工程师、真实世界。
- old: 请连续读出这些容易混淆的词
- new: 请连续读出这些容易混淆的词

### zh_numbers_dates

- target: 订单编号是二零二六零五三一，金额为一千二百三十四点五六元，请在下午三点十五分之前确认。
- old: 订单编号是20260531金额为1214我物则4请在下午3点15分之前确认
- new: 订单编号是202605A31行为1234.56元请在下午3点15分析之前确认

### zh_technical

- target: 我们正在评估端到端语音合成系统，关注文本切分、韵律控制、上下文一致性和多语言发音稳定性。
- old: 我们正在评估端到端语音合成系统关注文竊分韵律控制上下文一致性和多语言发音稳定性
- new: 我们正在评估端到端语音合成系统关注文本切分韵律控制上下文一致性和多语言发音稳定性

### zh_questions

- target: 如果今天晚上下雨，我们还要按原计划开会吗？请先确认参会人员，再通知上海和北京两个团队。
- old: 如果今天晚上下雨,我们还要按原计划开会吗?请先确认参会人员再通知上海和北京两个团队。
- new: 如果今天晚上下雨,我们还要按原计划开会吗?请先确认参会人员,再通知上海和北京两个团队。

### fr_nasal_liaison

- target: Nous vérifions les voyelles nasales dans un bon vin blanc, un ancien document, cinquante invitations et une décision importante.
- old: Nous vérifions les voyelles nasales dans un bon vin blanc, un ancien document, 50 invitations et une décision importante.
- new: Nous vérifions les voyelles nasales dans un bon vin blanc, un ancien document, 50 invitations, et une décision importante.

### fr_numbers_dates

- target: Le rendez-vous est fixé au trente et un mai deux mille vingt-six, à quinze heures quarante-cinq, avec un budget de mille deux cents euros.
- old: Le rendez-vous est fixé au 31 mai 2026, à 15h45, avec un budget de 1200 euros.
- new: Le rendez-vous est fixé au 30 et 1 mai 2026, à 15h45, avec un budget de 1 200 euros.

### fr_technical

- target: Cette démonstration évalue la synthèse vocale neuronale, la stabilité du timbre, le rythme des phrases longues et la prononciation des termes techniques.
- old: Cette démonstration évalue la synthèse vocale neuronale, la stabilité du timbre, le rythme des phrases longues et la prononciation des termes techniques.
- new: Cette démonstration évalue la synthèse vocale neuronale, la stabilité du timbre, le rythme des phrases longues et la prononciation des termes techniques.

### fr_questions

- target: Pouvez-vous vérifier si l'utilisateur reçoit bien la notification, puis relancer la génération audio après la mise à jour du serveur ?
- old: Pouvez-vous vérifier si l'utilisateur reçoit bien la notification, puis relancer la génération audio après la mise à jour du serveur ?
- new: Pouvez-vous vérifier si l'utilisateur reçoit bien la notification, puis relancer le Générations Audio, après la mise à jour du serveur ?

### ar_emphatic

- target: نراجع مخارج الحروف في كلمات مثل الصباح، الضمان، الطريق، الظرف، القطار، القرار، والخطاب الرسمي.
- old: نراجع مخارج الحروف في كلمات مثل الصباح، الضمان، الطريق، الظرف، القطار، القرار، والخطاب الرسمي.
- new: نراجع مخارج الحروف في كلمات مثل الصباح، الضمان، التطيق، الظرف، القطار، القرار والخطاب الرسمي.

### ar_numbers_dates

- target: موعد الاجتماع في الحادي والثلاثين من مايو عام ألفين وستة وعشرين، عند الساعة الثالثة وخمس وأربعين دقيقة مساءً.
- old: موعد الاجتماع في الحادي والثلاثين من مايو عام الفين وستة وعشرين عند الساعة الثالثة وخمسة واربعين دقيقك مساء
- new: موعد الاجتماع في الحادي والثلاثين من مايو عام الفين وستة وعشرين عند الساعة الثالثة وخمس وأربعين دقيقة مساء

### ar_technical

- target: هذا الاختبار يقيس جودة تحويل النص إلى كلام، وثبات النبرة، وتناسق الجمل الطويلة، ودقة نطق المصطلحات التقنية.
- old: هذا الاختبار يقيس جودة تحويل النص إلى كلام، وثبات النبرة، وتناسق الجمل جمل الطويلة، ودقة نطق المصطلحات التقنية.
- new: هذا الاختبار يقيس جودة تحويل النص إلى كلام، وثبات النبرة، وتناسق الجمل الطويلة، ودقة النطق المصطلحات التقنية.

### ar_questions

- target: هل يمكنك التأكد من أن المستخدم استلم الرسالة، ثم إعادة تشغيل عملية التوليد بعد تحديث إعدادات الخادم؟
- old: هل يمكنك التأكد من أن المستخدم استلم الرسالة، ثم إعادة تشغيل العملية التوليد بعد تحديث إعدادات الخادم؟
- new: هل يمكنك التأكد من أن المستخدم استلم الرسالة؟ ثم إعادة التشغيل عملية التوليد بعد تحديث إعدادات الخادم؟
