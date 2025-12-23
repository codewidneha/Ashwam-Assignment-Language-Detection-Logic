#!/usr/bin/env python3
"""
Ashwam Journaling - Language Detection CLI

Usage: lang_detect --in texts.jsonl --out lang.jsonl
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

from detector import LanguageDetector, DetectionResult, process_file


def validate_input_file(file_path: str) -> str:
    """
    Validate that the input file exists and is readable.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Input file '{file_path}' does not exist")
    
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"'{file_path}' is not a file")
    
    if not os.access(file_path, os.R_OK):
        raise argparse.ArgumentTypeError(f"Cannot read input file '{file_path}'")
    
    return file_path


def validate_output_file(file_path: str) -> str:
    """
    Validate that the output file can be created/written.
    """
    path = Path(file_path)
    
    # Check if directory exists and is writable
    if path.exists():
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"'{file_path}' exists but is not a file")
        if not os.access(file_path, os.W_OK):
            raise argparse.ArgumentTypeError(f"Cannot write to output file '{file_path}'")
    else:
        # Check if parent directory exists and is writable
        parent = path.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise argparse.ArgumentTypeError(f"Cannot create output directory: {e}")
        if not os.access(parent, os.W_OK):
            raise argparse.ArgumentTypeError(f"Cannot write to output directory '{parent}'")
    
    return file_path


def validate_jsonl_format(file_path: str, sample_size: int = 5) -> None:
    """
    Validate that the input file is in proper JSONL format.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            raise ValueError(f"Line {i+1}: JSON object must be a dictionary")
                        if 'text' not in data:
                            raise ValueError(f"Line {i+1}: JSON object must have 'text' field")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Line {i+1}: Invalid JSON - {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error: {e}")


def print_statistics(results: List[DetectionResult]) -> None:
    """
    Print processing statistics to stderr.
    """
    if not results:
        return
    
    language_counts = {}
    script_counts = {}
    total_confidence = 0
    
    for result in results:
        language_counts[result.primary_language] = language_counts.get(result.primary_language, 0) + 1
        script_counts[result.script] = script_counts.get(result.script, 0) + 1
        total_confidence += result.confidence
    
    avg_confidence = total_confidence / len(results)
    
    print(f"\n=== Processing Statistics ===", file=sys.stderr)
    print(f"Total records processed: {len(results)}", file=sys.stderr)
    print(f"Average confidence: {avg_confidence:.3f}", file=sys.stderr)
    print(f"\nLanguage distribution:", file=sys.stderr)
    for lang, count in sorted(language_counts.items()):
        print(f"  {lang}: {count} ({count/len(results)*100:.1f}%)", file=sys.stderr)
    print(f"\nScript distribution:", file=sys.stderr)
    for script, count in sorted(script_counts.items()):
        print(f"  {script}: {count} ({count/len(results)*100:.1f}%)", file=sys.stderr)
    print("=" * 30, file=sys.stderr)


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description='Detect language and script for journaling text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lang_detect --in texts.jsonl --out lang.jsonl
  lang_detect --in input/data.jsonl --out output/results.jsonl

Output format:
  Each line contains a JSON object with:
  - id: text identifier
  - primary_language: en|hi|hinglish|mixed|unknown
  - script: latin|devanagari|mixed|other
  - confidence: float between 0 and 1
  - evidence: object with interpretable signals
        """
    )
    
    parser.add_argument(
        '--in',
        dest='input_file',
        required=True,
        type=validate_input_file,
        help='Input JSONL file with texts to analyze'
    )
    
    parser.add_argument(
        '--out',
        dest='output_file',
        required=True,
        type=validate_output_file,
        help='Output JSONL file for detection results'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate input format, do not process'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print processing statistics to stderr'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input format
        if not args.quiet:
            print(f"Validating input file: {args.input_file}", file=sys.stderr)
        
        validate_jsonl_format(args.input_file)
        
        if args.validate_only:
            print("Input file validation passed", file=sys.stderr)
            return
        
        # Process the file
        if not args.quiet:
            print(f"Processing: {args.input_file} -> {args.output_file}", file=sys.stderr)
        
        # For statistics, we need to collect results
        results = []
        if args.stats:
            detector = LanguageDetector()
            
            with open(args.input_file, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        text_id = data.get('id', f'line_{line_num}')
                        text = data.get('text', '')
                        
                        result = detector.detect(text_id, text)
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                        continue
        
        # Process the file (writes output)
        process_file(args.input_file, args.output_file)
        
        if not args.quiet:
            print(f"Completed processing", file=sys.stderr)
        
        if args.stats:
            print_statistics(results)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
