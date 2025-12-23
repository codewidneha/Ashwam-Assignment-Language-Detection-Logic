
import json
import re
import sys
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import unicodedata

@dataclass
class DetectionResult:
    id: str
    primary_language: str
    script: str
    confidence: float
    evidence: Dict[str, Any]

class LanguageDetector:
    
    HINDI_WORDS = {
        'hai', 'haan', 'han', 'hain', 'mujhe', 'mujhse', 'aaj', 'kal', 
        'nahi', 'na', 'nahin', 'thoda', 'thode', 'theek', 'theekhai',
        'yaar', 'bhai', 'dost', 'accha', 'achha', 'bahut', 'bilkul',
        'kya', 'kaise', 'kyun', 'kahan', 'kab', 'kaun', 'kisko', 'kiska',
        'mera', 'meri', 'mere', 'tera', 'teri', 'tere', 'apna', 'apni', 'apne',
        'hum', 'tum', 'aap', 'woh', 'ye', 'yeh', 'voh', 'sab', 'sabhi',
        'dil', 'pyar', 'ishq', 'mohabbat', 'dosti', 'friendship',
        'lag', 'lagta', 'lagti', 'lagte', 'raha', 'rahi', 'rahe', 'rah',
        'kar', 'karna', 'karne', 'kiya', 'kiye', 'karo', 'karenge',
        'ja', 'jana', 'jane', 'jao', 'jayenge', 'gayi', 'gaye',
        'de', 'dena', 'dene', 'diya', 'diye', 'do', 'denge',
        'le', 'lena', 'lene', 'liya', 'liye', 'lo', 'lenge',
        'khana', 'khane', 'khaya', 'khaye', 'khao', 'khayenge',
        'sona', 'sone', 'soya', 'soye', 'soo', 'soyenge',
        'mein', 'me', 'ko', 'se', 'pe', 'par', 'ke', 'ka', 'ki', 'liye',
        'bhi', 'to', 'hi', 'tak', 'tak', 'tak', 'tak', 'tak', 'tak', 'tak', 'tak',
        'aur', 'ya', 'lekin', 'par', 'magar', 'kyunki', 'isliye', 'warna',
        'chalo', 'chal', 'chalte', 'aate', 'aate', 'aayenge', 'aaya', 'aye',
        'dekh', 'dekhte', 'dekhna', 'dekho', 'dekhenge', 'dekha', 'dekhe',
        'bol', 'bolte', 'bolna', 'bolo', 'bolenge', 'bola', 'bole',
        'sun', 'sunte', 'sunna', 'suno', 'sunenge', 'suna', 'sune',
        'samajh', 'samajhte', 'samajhna', 'samjho', 'samjhenge', 'samjha', 'samjhe'
    }
    
    ENGLISH_WORDS = {
        'the', 'is', 'was', 'at', 'and', 'to', 'a', 'an', 'in', 'on', 'of', 
        'for', 'with', 'as', 'by', 'from', 'or', 'but', 'not', 'be', 'are',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'what', 'where', 'when', 'why', 'how',
        'who', 'whom', 'whose', 'which', 'whoever', 'whatever', 'whenever',
        'wherever', 'however', 'whichever', 'if', 'then', 'else', 'whether',
        'while', 'until', 'because', 'since', 'though', 'although', 'unless',
        'today', 'tomorrow', 'yesterday', 'now', 'here', 'there', 'hello', 'hi', 'hey',
        'am', 'go', 'going', 'come', 'coming', 'ok', 'okay', 'sorry', 'please', 'thanks', 'thank', 'yes', 'no'
    }
    
    def __init__(self):
        self.devanagari_chars = re.compile(r'[\u0900-\u097F]')
        self.latin_chars = re.compile(r'[a-zA-Z]')
        self.emojis = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        self.numeric_only = re.compile(r'^[\d\s\-\+\.\,\%]+$')
        self.alphanumeric = re.compile(r'^[a-zA-Z0-9]+$')
        
    def _count_scripts(self, text: str) -> Dict[str, int]:
        latin = len(self.latin_chars.findall(text))
        devanagari = len(self.devanagari_chars.findall(text))
        total = len([c for c in text if not c.isspace()])
        
        return {
            'latin_count': latin,
            'devanagari_count': devanagari,
            'total_chars': total
        }
    
    def _compute_ratios(self, char_counts: Dict[str, int]) -> Dict[str, float]:
        total = char_counts['total_chars']
        if total == 0:
            return {'latin_ratio': 0.0, 'devanagari_ratio': 0.0}
        
        return {
            'latin_ratio': char_counts['latin_count'] / total,
            'devanagari_ratio': char_counts['devanagari_count'] / total
        }
    
    def _analyze_tokens(self, text: str) -> Dict[str, Any]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        clean_tokens = [t for t in tokens if self.alphanumeric.match(t)]
        
        hindi_hits = sum(1 for t in clean_tokens if t in self.HINDI_WORDS)
        english_hits = sum(1 for t in clean_tokens if t in self.ENGLISH_WORDS)
        
        return {
            'total_tokens': len(tokens),
            'alnum_tokens': len(clean_tokens),
            'hi_lexicon_hits': hindi_hits,
            'en_stopword_hits': english_hits,
            'tokens': tokens
        }
    
    def _check_noise(self, text: str) -> Dict[str, bool]:
        stripped = text.strip()
        
        return {
            'emoji_only': bool(self.emojis.fullmatch(stripped)),
            'numeric_only': bool(self.numeric_only.fullmatch(stripped)),
            'very_short': len(stripped) < 3
        }
    
    def _decide_script(self, char_counts: Dict[str, int], ratios: Dict[str, float]) -> str:
        total = char_counts['total_chars']
        if total == 0:
            return 'other'
        
        latin_ratio = ratios['latin_ratio']
        dev_ratio = ratios['devanagari_ratio']
        
        meaningful = 0.1
        
        has_latin_mix = latin_ratio > meaningful
        has_dev_mix = dev_ratio > meaningful
        
        has_latin_chars = char_counts['latin_count'] > 0
        has_dev_chars = char_counts['devanagari_count'] > 0
        
        if has_latin_mix and has_dev_mix:
            return 'mixed'
        elif has_latin_mix or has_latin_chars:
            if has_dev_chars:
                return 'mixed'
            else:
                return 'latin'
        elif has_dev_mix or has_dev_chars:
            if has_latin_chars:
                return 'mixed'
            else:
                return 'devanagari'
        else:
            return 'other'
    
    def _decide_language(self, char_counts: Dict[str, int], ratios: Dict[str, float], 
                           token_stats: Dict[str, Any], script: str) -> str:
        
        total_chars = char_counts['total_chars']
        total_tokens = token_stats['total_tokens']
        clean_tokens = token_stats['alnum_tokens']
        
        if total_chars < 3 or total_tokens == 0:
            return 'unknown'
        
        hindi_hits = token_stats['hi_lexicon_hits']
        english_hits = token_stats['en_stopword_hits']
        
        hindi_ratio = hindi_hits / max(clean_tokens, 1)
        english_ratio = english_hits / max(clean_tokens, 1)
        
        if script == 'devanagari':
            if total_chars >= 5:
                return 'hi'
            else:
                return 'unknown'
        
        elif script == 'latin':
            non_hindi = clean_tokens - hindi_hits
            non_hindi_score = non_hindi / max(clean_tokens, 1)
            
            strong_hindi_presence = (hindi_hits > 3 * english_hits) and (hindi_ratio > 0.3)
            
            if strong_hindi_presence:
                return 'hinglish'
            
            if english_ratio > 3 * hindi_ratio and english_ratio > 0.3:
                return 'en'
            
            if (hindi_hits >= 1 and english_hits >= 1 and 
                english_ratio >= 0.15 and hindi_ratio >= 0.15):
                return 'mixed'
                
            if (0.2 <= hindi_ratio <= 0.7 and non_hindi_score >= 0.2 and 
                clean_tokens >= 4):
                 if english_ratio > 0.1:
                     return 'mixed'
            
            if hindi_ratio > 0.2:
                if english_ratio > 0.1:
                    return 'mixed'
                else:
                    return 'hinglish'
            elif english_ratio > 0.15:
                if hindi_ratio > 0.05:
                    return 'mixed'
                else:
                    return 'en'
            elif hindi_ratio > 0.05:
                if english_ratio > 0.05:
                    return 'mixed'
                else:
                    return 'hinglish'
            else:
                if total_chars >= 10:
                    return 'en'
                else:
                    return 'unknown'
        
        elif script == 'mixed':
             if hindi_ratio > 0.1 and english_ratio > 0.1:
                return 'mixed'
             else:
                return 'mixed'
        
        else:
            return 'unknown'
    
    def _compute_confidence(self, char_counts: Dict[str, int], ratios: Dict[str, float],
                             token_stats: Dict[str, Any], script: str, 
                             language: str, noise: Dict[str, bool]) -> float:
        
        total_chars = char_counts['total_chars']
        total_tokens = token_stats['total_tokens']
        clean_tokens = token_stats['alnum_tokens']
        
        if language == 'unknown':
            if noise['emoji_only'] or noise['numeric_only']:
                return 0.0
            elif noise['very_short']:
                return 0.1
            else:
                return 0.2
        
        score = 0.3
        
        if total_chars >= 20:
            score += 0.2
        elif total_chars >= 10:
            score += 0.1
        
        dominance = max(ratios['latin_ratio'], ratios['devanagari_ratio'])
        if dominance > 0.8:
            score += 0.1
        
        if clean_tokens > 0:
            hindi_ratio = token_stats['hi_lexicon_hits'] / clean_tokens
            english_ratio = token_stats['en_stopword_hits'] / clean_tokens
            
            if language == 'hinglish':
                 if hindi_ratio > 0.4: score += 0.3
                 elif hindi_ratio > 0.2: score += 0.15
            elif language == 'en':
                 if english_ratio > 0.3: score += 0.2
                 elif english_ratio > 0.15: score += 0.1
            elif language == 'hi':
                 if hindi_ratio > 0.1: score += 0.15
            elif language == 'mixed':
                 if hindi_ratio > 0.3 and english_ratio > 0.3:
                     score += 0.05
        
        if noise['very_short']:
            score -= 0.3
        if noise['emoji_only'] or noise['numeric_only']:
            score -= 0.4
            
        if language == 'hinglish' and total_tokens < 6:
            score = min(0.85, score)
        
        if script in ['latin', 'devanagari'] and language in ['en', 'hi', 'hinglish']:
            score += 0.1
        
        if total_tokens < 4:
            score -= 0.15
        elif total_tokens < 6:
            score -= 0.05
            
        if language == 'mixed':
            score -= 0.1
            score = min(0.85, score)
        else:
            score = min(0.95, score)
            
        return max(0.0, score)
    
    def _gather_evidence(self, char_counts: Dict[str, int], ratios: Dict[str, float],
                         token_stats: Dict[str, Any]) -> Dict[str, Any]:
        
        clean_tokens = token_stats['alnum_tokens']
        
        data = {
            'latin_ratio': ratios['latin_ratio'],
            'devanagari_ratio': ratios['devanagari_ratio'],
            'hi_lexicon_hits': token_stats['hi_lexicon_hits'],
            'en_stopword_hits': token_stats['en_stopword_hits'],
            'n_tokens': token_stats['total_tokens'],
            'n_alnum_tokens': clean_tokens
        }
        
        if clean_tokens > 0:
            data['hi_lexicon_ratio'] = token_stats['hi_lexicon_hits'] / clean_tokens
            data['en_stopword_ratio'] = token_stats['en_stopword_hits'] / clean_tokens
        else:
            data['hi_lexicon_ratio'] = 0.0
            data['en_stopword_ratio'] = 0.0
        
        return data
    
    def detect(self, text_id: str, text: str) -> DetectionResult:
        chars = self._count_scripts(text)
        ratios = self._compute_ratios(chars)
        tokens = self._analyze_tokens(text)
        noise = self._check_noise(text)
        
        script = self._decide_script(chars, ratios)
        lang = self._decide_language(chars, ratios, tokens, script)
        confidence = self._compute_confidence(chars, ratios, tokens, script, lang, noise)
        evidence = self._gather_evidence(chars, ratios, tokens)
        
        return DetectionResult(
            id=text_id,
            primary_language=lang,
            script=script,
            confidence=confidence,
            evidence=evidence
        )

def process_file(input_path: str, output_path: str) -> None:
    detector = LanguageDetector()
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    tid = data.get('id', f'line_{i}')
                    text = data.get('text', '')
                    
                    res = detector.detect(tid, text)
                    
                    output = {
                        'id': res.id,
                        'primary_language': res.primary_language,
                        'script': res.script,
                        'confidence': res.confidence,
                        'evidence': res.evidence
                    }
                    
                    outfile.write(json.dumps(output, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != '--in' or sys.argv[3] != '--out':
        print("Usage: python detector.py --in <input_file> --out <output_file>", file=sys.stderr)
        sys.exit(1)
    
    process_file(sys.argv[2], sys.argv[4])
    print(f"Processed {sys.argv[2]} -> {sys.argv[4]}")
