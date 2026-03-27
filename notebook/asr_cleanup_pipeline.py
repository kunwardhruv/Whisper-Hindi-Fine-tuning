import re
import os
import json
import glob
import torch
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Number mapping for Hindi numbers
DIGITS = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5, 'पाँच': 5, 'छह': 6, 'सात': 7, 'आठ': 8, 'नौ': 9,
    'दस': 10, 'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15, 'पन्द्रह': 15, 'सोलह': 16, 'सत्रह': 17,
    'अट्ठारह': 18, 'उन्नीस': 19, 'बीस': 20, 'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24, 'पच्चीस': 25,
    'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28, 'उनतीस': 29, 'तीस': 30, 'चालीस': 40, 'पचास': 50, 'साठ': 60,
    'सत्तर': 70, 'अस्सी': 80, 'नब्बे': 90, 'सौ': 100, 'हज़ार': 1000, 'हज़ारों': 1000, 'हजार': 1000, 'लाख': 100000,
    'करोड़': 10000000
}

MULTIPLIERS = {'सौ': 100, 'हज़ार': 1000, 'हजार': 1000, 'लाख': 100000, 'करोड़': 10000000}
IDIOM_SKIP = {'दो-चार', 'चार-पांच', 'दस-बीस', 'एक-आधा', 'एक-दो', 'सात-आठ'}

ENGLISH_HINTS = {
    'इंटरव्यू', 'जॉब', 'प्रॉब्लम', 'कंप्यूटर', 'लैपटॉप', 'मॉडल', 'गेम', 'ट्रेनिंग', 'स्कूल', 'कॉलेज', 'ब्रेक',
    'बैकग्राउंड', 'प्लान', 'क्लास', 'स्ट्रक्चर', 'स्टूडेंट', 'फार्म', 'थ्री', 'सिक्योरिटी', 'डाटा', 'डॉक्स', 'कम्युनिकेशन',
    'पॉइंट', 'गूगल', 'गैंज', 'मेटाडेटा', 'रेट', 'प्रोसेस', 'डिप्लॉय', 'हाइपर', 'पैकेज', 'स्क्रिप्ट', 'बग', 'डिबग', 'कोड'}

# Embellish: also treat common nonstandard spelling forms
ENGLISH_HINTS = {w for w in ENGLISH_HINTS} | {w.replace('ॉ', 'ो') for w in ENGLISH_HINTS}

number_token_pattern = re.compile(r'^[\u0900-\u097F\-]+$')


def normalize_numbers(text):
    tokens = re.split(r'(\s+)', text)
    out = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.strip() == '' or not number_token_pattern.match(token.strip()):
            out.append(token); i += 1; continue
        word = token.strip()
        if word in IDIOM_SKIP:
            out.append(word); i += 1; continue

        # sequence scanning for numeric words
        if word in DIGITS or word in MULTIPLIERS:
            value = 0
            current = 0
            j = i
            consumed = False
            while j < len(tokens):
                w = tokens[j].strip()
                if w == '' or not number_token_pattern.match(w):
                    break
                if w in IDIOM_SKIP:
                    break
                if w in MULTIPLIERS:
                    mul = MULTIPLIERS[w]
                    if current == 0:
                        current = 1
                    current *= mul
                    value += current
                    current = 0
                    consumed = True
                elif w in DIGITS:
                    current += DIGITS[w]
                    consumed = True
                else:
                    break
                j += 1
                # if next token contains dash with digits, treat as non-numeric phrase
                if j < len(tokens) and '-' in tokens[j]:
                    break
            if consumed and j > i:
                value += current
                if value != 0:
                    out.append(str(value))
                    i = j
                    continue
        out.append(token)
        i += 1

    return ''.join(out)


def tag_english_words(text):
    words = text.split()
    out = []
    for w in words:
        if w in ENGLISH_HINTS or w.lower() in ENGLISH_HINTS:
            out.append(f"[EN]{w}[/EN]")
        else:
            out.append(w)
    return ' '.join(out)


def transcribe_whisper_small(audio_path, processor, model):
    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors='pt').input_features
    with torch.no_grad():
        ids = model.generate(input_features, language='hindi', task='transcribe', max_length=200)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    root = os.path.abspath('..')
    audio_dir = os.path.join(root, 'data', 'audio')
    text_dir = os.path.join(root, 'data', 'transcripts_txt')

    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
    model.eval()

    samples = []

    for audio_path in tqdm(sorted(glob.glob(os.path.join(audio_dir, '*.wav'))), desc='ASR'):        
        file_id = os.path.basename(audio_path).replace('.wav', '')
        ref_path = os.path.join(text_dir, f'{file_id}.txt')
        if not os.path.exists(ref_path):
            continue

        ref = open(ref_path, 'r', encoding='utf-8').read().strip()
        raw = transcribe_whisper_small(audio_path, processor, model)
        norm = normalize_numbers(raw)
        tagged = tag_english_words(norm)

        samples.append({
            'id': file_id,
            'reference': ref,
            'raw': raw,
            'norm': norm,
            'tagged': tagged
        })

    out_path = os.path.join(root, 'outputs', 'asr_cleanup_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print('Done. Saved to', out_path)


if __name__ == '__main__':
    main()
