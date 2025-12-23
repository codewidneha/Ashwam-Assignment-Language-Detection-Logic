#!/usr/bin/env python3
"""
Comprehensive tests for the Ashwam Journaling Language Detection System.

Tests cover:
- Devanagari vs Latin detection
- Hinglish detection  
- Mixed script detection
- Unknown handling for very short / noisy strings
- Confidence scoring
- Edge cases
"""

import unittest
import json
import tempfile
import os
from pathlib import Path

from detector import LanguageDetector, DetectionResult, process_file
from lang_detect import validate_jsonl_format, main


class TestLanguageDetector(unittest.TestCase):
    """
    Test the core LanguageDetector class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_devanagari_detection(self):
        """Test detection of pure Devanagari text."""
        test_cases = [
            ("नमस्ते दुनिया", "hi", "devanagari"),
            ("मैं आज बहुत खुश हूं", "hi", "devanagari"),
            ("यह एक अच्छा दिन है", "hi", "devanagari"),
            ("हिंदी में लिखा गया है", "hi", "devanagari")
        ]
        
        for text, expected_lang, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
                self.assertEqual(result.script, expected_script)
                self.assertGreater(result.confidence, 0.5)
    
    def test_latin_english_detection(self):
        """Test detection of pure English text."""
        test_cases = [
            ("Hello world", "en", "latin"),
            ("The weather is nice today", "en", "latin"),
            ("This is a test sentence", "en", "latin"),
            ("I am going to the market", "en", "latin")
        ]
        
        for text, expected_lang, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
                self.assertEqual(result.script, expected_script)
                self.assertGreater(result.confidence, 0.5)
    
    def test_hinglish_detection(self):
        """Test detection of Hinglish (Romanized Hindi)."""
        test_cases = [
            ("mujhe aaj bahut khushi ho rahi hai", "hinglish", "latin"),
            ("yaar aaj kya plan hai", "hinglish", "latin"),
            ("main kal ja raha hun", "hinglish", "latin"),  # "main" matches Hindi lexicon context
            ("kya baat hai bhai", "hinglish", "latin"),
            ("theek hai main aa raha hun", "hinglish", "latin"),
            ("aaj bahut accha din tha", "hinglish", "latin")
        ]
        
        for text, expected_lang, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
                self.assertEqual(result.script, expected_script)
                self.assertGreater(result.confidence, 0.4)
    
    def test_mixed_language_detection(self):
        """Test detection of mixed language text."""
        test_cases = [
            ("Hello mujhe aaj khushi ho rahi hai", "hinglish", "latin"),  # Dominant Hindi -> Hinglish
            ("The weather is bahut accha today", "mixed", "latin"),
            ("I am going to market aur kuch leke aaunga", "mixed", "latin"),
            ("Hello नमस्ते how are you", "mixed", "mixed"),
            ("English और Hindi together", "mixed", "mixed")
        ]
        
        for text, expected_lang, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
                self.assertEqual(result.script, expected_script)
    
    def test_unknown_detection(self):
        """Test detection of unknown/very short/noisy text."""
        test_cases = [
            ("", "unknown"),
            ("a", "unknown"),
            ("123", "unknown"),
            ("😀", "unknown"),
            ("42", "unknown"),
            ("@#$%", "unknown"),
            ("abc", "unknown")  # Very short, might be unknown
        ]
        
        for text, expected_lang in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
    
    def test_script_detection_edge_cases(self):
        """Test script detection for edge cases."""
        test_cases = [
            ("한국어 텍스트", "other"),  # Korean
            ("日本語テキスト", "other"),  # Japanese
            ("العربية النص", "other"),  # Arabic
            ("123 abc 456", "latin"),  # Mixed numbers and Latin
            ("!@# $%^", "other"),  # Only symbols
        ]
        
        for text, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.script, expected_script)
    
    def test_confidence_scoring(self):
        """Test confidence scoring logic."""
        # High confidence cases (capped at 0.95)
        high_confidence_cases = [
            "This is a proper English sentence with good structure",
            "मैं आज बहुत खुश हूं और मेरा दिन अच्छा रहा",
            "mujhe aaj bahut khushi ho rahi hai kyunki din accha tha"
        ]
        
        for text in high_confidence_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertGreater(result.confidence, 0.5)
                self.assertLessEqual(result.confidence, 0.95)  # Confidence capped at 0.95
        
        # Low confidence cases
        low_confidence_cases = [
            "hi",
            "😀",
            "123",
            "a"
        ]
        
        for text in low_confidence_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertLess(result.confidence, 0.5)
    
    def test_evidence_object(self):
        """Test that evidence object contains required fields."""
        result = self.detector.detect("test", "Hello mujhe aaj khushi ho rahi hai")
        
        required_fields = [
            'latin_ratio', 'devanagari_ratio', 'hi_lexicon_hits',
            'en_stopword_hits', 'n_tokens', 'n_alnum_tokens',
            'hi_lexicon_ratio', 'en_stopword_ratio'
        ]
        
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, result.evidence)
                self.assertIsInstance(result.evidence[field], (int, float))
    
    def test_hindi_lexicon_coverage(self):
        """Test that important Hindi words are in lexicon."""
        important_hindi_words = [
            'hai', 'nahi', 'mujhe', 'yaar', 'kya', 'aaj', 'kal', 
            'bhai', 'accha', 'bahut', 'dil', 'pyar'
        ]
        
        for word in important_hindi_words:
            with self.subTest(word=word):
                self.assertIn(word, self.detector.HINDI_WORDS)
    
    def test_english_stopwords_coverage(self):
        """Test that important English stopwords are covered."""
        important_english_words = [
            'the', 'is', 'was', 'and', 'to', 'a', 'in', 'on',
            'for', 'with', 'i', 'you', 'he', 'she', 'it'
        ]
        
        for word in important_english_words:
            with self.subTest(word=word):
                self.assertIn(word, self.detector.ENGLISH_WORDS)


class TestJSONLProcessing(unittest.TestCase):
    """
    Test JSONL file processing functionality.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_process_jsonl_basic(self):
        """Test basic JSONL processing."""
        test_data = [
            {"id": "1", "text": "Hello world"},
            {"id": "2", "text": "mujhe aaj khushi ho rahi hai"},
            {"id": "3", "text": "नमस्ते दुनिया"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            for item in test_data:
                input_file.write(json.dumps(item) + '\n')
            input_file_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_file_path = output_file.name
        
        try:
            process_file(input_file_path, output_file_path)
            
            # Read and verify output
            with open(output_file_path, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            self.assertEqual(len(results), 3)
            
            # Check first result (English)
            self.assertEqual(results[0]['primary_language'], 'en')
            self.assertEqual(results[0]['script'], 'latin')
            
            # Check second result (Hinglish)
            self.assertEqual(results[1]['primary_language'], 'hinglish')
            self.assertEqual(results[1]['script'], 'latin')
            
            # Check third result (Hindi)
            self.assertEqual(results[2]['primary_language'], 'hi')
            self.assertEqual(results[2]['script'], 'devanagari')
            
            # Verify all results have required fields
            for result in results:
                self.assertIn('id', result)
                self.assertIn('primary_language', result)
                self.assertIn('script', result)
                self.assertIn('confidence', result)
                self.assertIn('evidence', result)
        
        finally:
            os.unlink(input_file_path)
            os.unlink(output_file_path)
    
    def test_malformed_jsonl_handling(self):
        """Test handling of malformed JSONL input."""
        test_data = [
            {"id": "1", "text": "Hello world"},
            "invalid json line",
            {"id": "2", "text": "mujhe khushi hai"},
            {"text": "missing id"},
            ""
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            for item in test_data:
                if isinstance(item, dict):
                    input_file.write(json.dumps(item) + '\n')
                else:
                    input_file.write(str(item) + '\n')
            input_file_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_file_path = output_file.name
        
        try:
            # Should not raise exception, should handle gracefully
            process_file(input_file_path, output_file_path)
            
            # Read and verify output
            with open(output_file_path, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            # Should have processed valid lines
            self.assertGreaterEqual(len(results), 2)
        
        finally:
            os.unlink(input_file_path)
            os.unlink(output_file_path)


class TestValidation(unittest.TestCase):
    """
    Test input validation functions.
    """
    
    def test_validate_jsonl_format_valid(self):
        """Test validation of valid JSONL files."""
        valid_data = [
            {"id": "1", "text": "Hello"},
            {"id": "2", "text": "World"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in valid_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            # Should not raise exception
            validate_jsonl_format(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_jsonl_format_invalid(self):
        """Test validation of invalid JSONL files."""
        invalid_cases = [
            'not json at all',
            '{"invalid": "json without text field"}',
            '["array", "instead", "of", "object"]'
        ]
        
        for invalid_content in invalid_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write(invalid_content + '\n')
                temp_path = f.name
            
            try:
                with self.assertRaises(ValueError):
                    validate_jsonl_format(temp_path)
            finally:
                os.unlink(temp_path)


class TestEdgeCases(unittest.TestCase):
    """
    Test various edge cases and boundary conditions.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_unicode_handling(self):
        """Test proper Unicode handling."""
        test_cases = [
            ("Hello 🌍 world", "en", "latin"),  # Emoji with English
            ("नमस्ते 🙏", "hi", "devanagari"),  # Emoji with Hindi
            ("café naïve", "unknown", "latin"),  # Accented characters (no strong signals)
            ("München ist schön", "en", "latin"),  # German (classified as English due to Latin script)
        ]
        
        for text, expected_lang, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
                self.assertEqual(result.script, expected_script)
    
    def test_code_switching_patterns(self):
        """Test various code-switching patterns."""
        test_cases = [
            ("I am going to market aur kuch leke aaunga", "mixed", "latin"),
            ("Let's go yaar, bahut der ho gayi", "mixed", "latin"),  # English + Hindi
            ("Today main busy hun kal baat karte hain", "mixed", "latin"),  # English + Hindi
            ("Sorry yaar, aaj nahi aa sakta", "mixed", "latin"),  # English + Hindi
        ]
        
        for text, expected_lang, expected_script in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
                self.assertEqual(result.script, expected_script)
    
    def test_numeric_and_special_characters(self):
        """Test handling of numbers and special characters."""
        test_cases = [
            ("123 456 789", "unknown"),
            ("Rs. 100 only", "en"),
            ("2pm par mil", "hinglish"),
            ("@#$%^&*()", "unknown"),
            ("test@example.com", "en"),
        ]
        
        for text, expected_lang in test_cases:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertEqual(result.primary_language, expected_lang)
    
    def test_confidence_bounds(self):
        """Test that confidence is always within bounds."""
        test_texts = [
            "", "a", "hello", "mujhe khushi hai", "नमस्ते दुनिया",
            "Hello world! How are you today? I hope you're doing well.",
            "yaar aaj kya plan hai bahut din baad mil rahe hain",
            "यह एक बहुत लंबा और विस्तृत हिंदी वाक्य है जिसमें बहुत सारे शब्द हैं"
        ]
        
        for text in test_texts:
            with self.subTest(text=text):
                result = self.detector.detect("test", text)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)


def run_integration_tests():
    """
    Run integration tests with sample data.
    """
    print("Running integration tests...")
    
    # Create sample test data
    sample_data = [
        {"id": "test_en_1", "text": "The weather is beautiful today"},
        {"id": "test_hi_1", "text": "आज मौसम बहुत अच्छा है"},
        {"id": "test_hinglish_1", "text": "aaj mausam bahut accha hai"},
        {"id": "test_mixed_1", "text": "Today mausam bahut accha hai"},
        {"id": "test_unknown_1", "text": "😀"},
        {"id": "test_short_1", "text": "hi"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
        for item in sample_data:
            input_file.write(json.dumps(item) + '\n')
        input_file_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
        output_file_path = output_file.name
    
    try:
        # Process the file
        process_file(input_file_path, output_file_path)
        
        # Read and display results
        with open(output_file_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        print(f"\nProcessed {len(results)} items:")
        for result in results:
            print(f"  {result['id']}: {result['primary_language']} ({result['script']}) - {result['confidence']:.3f}")
        
        print("✅ Integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        raise
    finally:
        os.unlink(input_file_path)
        os.unlink(output_file_path)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    run_integration_tests()
