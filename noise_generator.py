#!/usr/bin/env python3
"""
Prompt Noise Generator for Robustness Testing
Creates various types of noise to test model robustness against adversarial inputs
"""

import random
import re
import string
from typing import List, Dict, Callable
import numpy as np

class PromptNoiseGenerator:
    """Generate various types of noise for prompt injection robustness testing"""
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)

        self.leet_substitutions = {
            'a': ['4', '@', 'ä', 'á', 'à'],
            'e': ['3', 'ë', 'é', 'è'],
            'i': ['1', '!', 'í', 'ì'],
            'o': ['0', 'ó', 'ò', 'ö'],
            'u': ['ü', 'ú', 'ù'],
            's': ['5', '$', 'ś'],
            't': ['7', '+'],
            'l': ['1', '|'],
            'g': ['9'],
            'b': ['6'],
            'z': ['2']
        }

        self.common_typos = {
            'the': ['teh', 'hte', 'th3'],
            'you': ['u', 'yo', 'yuo'],
            'and': ['nd', 'an', '&'],
            'for': ['4', 'fr', 'fo'],
            'all': ['al', 'a11', 'all'],
            'are': ['r', 'ar', 'a4e'],
            'your': ['ur', 'you', 'y0ur'],
            'from': ['frm', 'fro', 'fr0m'],
            'with': ['wit', 'wth', 'w1th'],
            'have': ['hav', 'hve', 'h4ve'],
            'this': ['tis', 'th1s', 'thsi'],
            'that': ['tat', 'th4t', 'tht'],
            'will': ['wil', 'w1ll', 'wll'],
            'what': ['wat', 'wht', 'wh4t'],
            'ignore': ['ignor', '1gnore', 'ignre', 'ign0re'],
            'instructions': ['instructi0ns', '1nstructions', 'instruct10ns', 'instructons'],
            'previous': ['prev10us', 'previ0us', 'previus', 'prev1ous'],
            'system': ['syst3m', 'sys7em', 'systm', 'syst€m'],
            'forget': ['f0rget', 'forg3t', 'forgt', 'f0rg3t']
        }

    def add_character_substitution(self, text: str, intensity: float = 0.3) -> str:
        """
        Replace characters with leet speak or similar substitutions

        Args:
            text: Input text to modify
            intensity: Proportion of eligible characters to substitute (0.0 to 1.0)

        Returns:
            Modified text with character substitutions
        """
        if not text:
            return text

        result = list(text.lower())

        for i, char in enumerate(result):
            if char in self.leet_substitutions and random.random() < intensity:
                substitutes = self.leet_substitutions[char]
                result[i] = random.choice(substitutes)

        return ''.join(result)

    def add_spacing_noise(self, text: str, intensity: float = 0.2) -> str:
        """
        Add random spaces, remove spaces, or add extra whitespace

        Args:
            text: Input text to modify
            intensity: Proportion of positions to modify (0.0 to 1.0)

        Returns:
            Modified text with spacing noise
        """
        if not text:
            return text

        result = []
        words = text.split()

        for word in words:
            if random.random() < intensity:
                noise_type = random.choice(['insert_space', 'remove_space', 'extra_spaces'])

                if noise_type == 'insert_space' and len(word) > 3:
                    pos = random.randint(1, len(word) - 2)
                    word = word[:pos] + ' ' + word[pos:]
                elif noise_type == 'remove_space':
                    word = word.replace(' ', '')
                elif noise_type == 'extra_spaces':
                    word = word + ' ' * random.randint(1, 4)

            result.append(word)

        if random.random() < intensity:
            if len(result) > 1:
                idx = random.randint(0, len(result) - 2)
                result[idx] = result[idx] + result[idx + 1]
                result.pop(idx + 1)

        return ' '.join(result)

    def add_case_variation(self, text: str, intensity: float = 0.4) -> str:
        """
        Add random capitalization changes

        Args:
            text: Input text to modify
            intensity: Proportion of characters to potentially modify (0.0 to 1.0)

        Returns:
            Modified text with case variations
        """
        if not text:
            return text

        result = []

        for char in text:
            if char.isalpha() and random.random() < intensity:
                if char.islower():
                    result.append(char.upper())
                else:
                    result.append(char.lower())
            else:
                result.append(char)

        return ''.join(result)

    def add_insertion_noise(self, text: str, intensity: float = 0.1) -> str:
        """
        Insert random characters or punctuation

        Args:
            text: Input text to modify
            intensity: Frequency of insertions (0.0 to 1.0)

        Returns:
            Modified text with character insertions
        """
        if not text:
            return text

        noise_chars = ['*', '-', '_', '.', ',', ';', '!', '?', '#', '%', '&']
        result = []

        for i, char in enumerate(text):
            result.append(char)
            if random.random() < intensity:
                result.append(random.choice(noise_chars))

        return ''.join(result)

    def add_deletion_noise(self, text: str, intensity: float = 0.1) -> str:
        """
        Randomly delete characters

        Args:
            text: Input text to modify
            intensity: Proportion of characters to potentially delete (0.0 to 1.0)

        Returns:
            Modified text with character deletions
        """
        if not text:
            return text

        result = []

        for char in text:
            if random.random() > intensity:
                result.append(char)

        if len(result) < len(text) * 0.5:
            return text

        return ''.join(result)

    def add_common_misspellings(self, text: str, intensity: float = 0.3) -> str:
        """
        Replace words with common misspellings and typos

        Args:
            text: Input text to modify
            intensity: Proportion of eligible words to replace (0.0 to 1.0)

        Returns:
            Modified text with misspellings
        """
        if not text:
            return text

        words = text.split()
        result = []

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())

            if clean_word in self.common_typos and random.random() < intensity:
                prefix_punct = re.search(r'^[^\w]*', word)
                suffix_punct = re.search(r'[^\w]*$', word)
                was_capitalized = word[0].isupper() if word else False

                typo = random.choice(self.common_typos[clean_word])
                if was_capitalized:
                    typo = typo.capitalize()

                if prefix_punct:
                    typo = prefix_punct.group() + typo
                if suffix_punct:
                    typo = typo + suffix_punct.group()

                result.append(typo)
            else:
                result.append(word)

        return ' '.join(result)

    def add_unicode_confusion(self, text: str, intensity: float = 0.2) -> str:
        """
        Replace characters with visually similar unicode characters

        Args:
            text: Input text to modify
            intensity: Proportion of characters to potentially replace (0.0 to 1.0)

        Returns:
            Modified text with unicode confusion characters
        """
        unicode_confusion = {
            'a': ['а', 'α', 'а'],  # Cyrillic a, Greek alpha
            'e': ['е', 'ε', 'е'],  # Cyrillic e, Greek epsilon
            'o': ['о', 'ο', 'о'],  # Cyrillic o, Greek omicron
            'p': ['р', 'ρ', 'р'],  # Cyrillic p, Greek rho
            'c': ['с', 'ϲ', 'с'],  # Cyrillic c, Greek lunate sigma
            'x': ['х', 'χ', 'х'],  # Cyrillic x, Greek chi
            'y': ['у', 'υ', 'у'],  # Cyrillic y, Greek upsilon
            'i': ['і', 'ι', 'і'],  # Cyrillic i, Greek iota
            's': ['ѕ', 'σ', 'ѕ'],  # Cyrillic s, Greek sigma
        }

        result = []
        for char in text:
            if char.lower() in unicode_confusion and random.random() < intensity:
                confusion_chars = unicode_confusion[char.lower()]
                replacement = random.choice(confusion_chars)
                if char.isupper():
                    replacement = replacement.upper()
                result.append(replacement)
            else:
                result.append(char)

        return ''.join(result)

    def add_format_noise(self, text: str, intensity: float = 0.2) -> str:
        """
        Add formatting characters like newlines, tabs, HTML tags

        Args:
            text: Input text to modify
            intensity: Frequency of format insertions (0.0 to 1.0)

        Returns:
            Modified text with formatting noise
        """
        if not text:
            return text

        format_chars = ['\n', '\t', '\\n', '\\t', '<br>', '</br>', '&nbsp;', '\r']
        words = text.split()
        result = []

        for word in words:
            result.append(word)
            if random.random() < intensity:
                result.append(random.choice(format_chars))

        return ' '.join(result)

    def generate_noise_variants(self, text: str, num_variants: int = 5) -> List[Dict[str, str]]:
        """
        Generate multiple noisy variants of the input text using different techniques

        Args:
            text: Input text to modify
            num_variants: Number of variants to generate

        Returns:
            List of dictionaries with 'technique' and 'text' keys
        """
        variants = []

        noise_functions = [
            ('character_substitution', lambda t: self.add_character_substitution(t, 0.3)),
            ('spacing_noise', lambda t: self.add_spacing_noise(t, 0.2)),
            ('case_variation', lambda t: self.add_case_variation(t, 0.4)),
            ('insertion_noise', lambda t: self.add_insertion_noise(t, 0.1)),
            ('deletion_noise', lambda t: self.add_deletion_noise(t, 0.1)),
            ('misspellings', lambda t: self.add_common_misspellings(t, 0.3)),
            ('unicode_confusion', lambda t: self.add_unicode_confusion(t, 0.2)),
            ('format_noise', lambda t: self.add_format_noise(t, 0.2)),
        ]

        for technique_name, noise_func in noise_functions[:min(num_variants, len(noise_functions))]:
            noisy_text = noise_func(text)
            variants.append({
                'technique': technique_name,
                'text': noisy_text
            })

        remaining_variants = num_variants - len(variants)
        for _ in range(remaining_variants):
            num_techniques = random.randint(2, 3)
            selected_techniques = random.sample(noise_functions, num_techniques)

            noisy_text = text
            technique_names = []

            for technique_name, noise_func in selected_techniques:
                noisy_text = noise_func(noisy_text)
                technique_names.append(technique_name)

            variants.append({
                'technique': '+'.join(technique_names),
                'text': noisy_text
            })

        return variants

    def test_robustness_suite(self, prompts: List[str], expected_labels: List[str] = None) -> Dict:
        """
        Generate a comprehensive robustness test suite

        Args:
            prompts: List of input prompts to test
            expected_labels: Optional list of expected labels for each prompt

        Returns:
            Dictionary with test cases organized by technique
        """
        test_suite = {
            'original': [],
            'variants': {}
        }

        for i, prompt in enumerate(prompts):
            original_entry = {'text': prompt}
            if expected_labels:
                original_entry['expected_label'] = expected_labels[i]
            test_suite['original'].append(original_entry)

            variants = self.generate_noise_variants(prompt, num_variants=8)

            for variant in variants:
                technique = variant['technique']
                if technique not in test_suite['variants']:
                    test_suite['variants'][technique] = []

                variant_entry = {
                    'original_text': prompt,
                    'noisy_text': variant['text'],
                    'original_index': i
                }

                if expected_labels:
                    variant_entry['expected_label'] = expected_labels[i]

                test_suite['variants'][technique].append(variant_entry)

        return test_suite

import pandas as pd

def add_noise_to_csv(input_csv: str, output_csv: str, text_column: str = "text"):
    df = pd.read_csv(input_csv)
    generator = PromptNoiseGenerator(seed=42)

    noise_functions = {
        'character_substitution': lambda t: generator.add_character_substitution(t, 0.3),
        'spacing_noise': lambda t: generator.add_spacing_noise(t, 0.2),
        'case_variation': lambda t: generator.add_case_variation(t, 0.4),
        'insertion_noise': lambda t: generator.add_insertion_noise(t, 0.1),
        'deletion_noise': lambda t: generator.add_deletion_noise(t, 0.1),
        'misspellings': lambda t: generator.add_common_misspellings(t, 0.3),
        'unicode_confusion': lambda t: generator.add_unicode_confusion(t, 0.2),
        'format_noise': lambda t: generator.add_format_noise(t, 0.2),
    }

    noisy_texts = []
    noise_types = []

    for text in df[text_column].astype(str):
        noise_type = random.choice(list(noise_functions.keys()))
        noisy_text = noise_functions[noise_type](text)

        noisy_texts.append(noisy_text)
        noise_types.append(noise_type)

    df[text_column] = noisy_texts
    df["noise_type"] = noise_types

    df.to_csv(output_csv, index=False)
    print(f"Saved noisy dataset to {output_csv}")

if __name__ == "__main__":
    add_noise_to_csv("unified_dataset.csv", "noisy_dataset.csv", text_column="text")
