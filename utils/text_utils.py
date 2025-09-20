import re
import unicodedata
from typing import List, Dict, Any

class TextUtils:
    """Utility functions for text processing and manipulation"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        return text.strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitting for Vietnamese and English
        sentence_endings = r'[.!?]+\s*'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter out empty sentences
        sentences = [TextUtils.clean_text(sent) for sent in sentences if sent.strip()]
        
        return sentences
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If word boundary is reasonably close
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    @staticmethod
    def extract_code_blocks(text: str) -> List[Dict[str, str]]:
        """Extract code blocks from text"""
        code_blocks = []
        
        # Pattern for code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2)
            
            code_blocks.append({
                'language': language,
                'code': code.strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Pattern for inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.finditer(inline_pattern, text)
        
        for match in inline_matches:
            code = match.group(1)
            
            code_blocks.append({
                'language': 'inline',
                'code': code,
                'start': match.start(),
                'end': match.end()
            })
        
        return code_blocks
    
    @staticmethod
    def highlight_keywords(text: str, keywords: List[str], 
                          highlight_format: str = "**{}**") -> str:
        """Highlight keywords in text"""
        if not keywords:
            return text
        
        # Sort keywords by length (longer first) to avoid partial replacements
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Create case-insensitive pattern
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            
            # Replace with highlighted version
            text = pattern.sub(lambda m: highlight_format.format(m.group()), text)
        
        return text
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect if text is Vietnamese or English"""
        # Vietnamese diacritics pattern
        vietnamese_pattern = r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]'
        
        vietnamese_chars = len(re.findall(vietnamese_pattern, text.lower()))
        total_alpha_chars = len(re.findall(r'[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text))
        
        if total_alpha_chars == 0:
            return 'unknown'
        
        vietnamese_ratio = vietnamese_chars / total_alpha_chars
        
        if vietnamese_ratio > 0.1:  # If more than 10% are Vietnamese diacritics
            return 'vietnamese'
        else:
            return 'english'
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls
    
    @staticmethod
    def remove_urls(text: str, replacement: str = "[URL]") -> str:
        """Remove URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.sub(url_pattern, replacement, text)
    
    @staticmethod
    def extract_email_addresses(text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    @staticmethod
    def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
        """Estimate reading time in minutes"""
        word_count = TextUtils.count_words(text)
        reading_time = max(1, round(word_count / words_per_minute))
        return reading_time
    
    @staticmethod
    def create_excerpt(text: str, max_words: int = 50) -> str:
        """Create an excerpt from text"""
        words = text.split()
        if len(words) <= max_words:
            return text
        
        excerpt = ' '.join(words[:max_words])
        return excerpt + "..."
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove diacritics from Vietnamese text for search purposes"""
        # Normalize to decomposed form
        text = unicodedata.normalize('NFD', text)
        
        # Remove combining characters (diacritics)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Normalize back to composed form
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    @staticmethod
    def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (fuzzy matching)"""
        # Normalize texts
        norm1 = TextUtils.remove_diacritics(text1.lower().strip())
        norm2 = TextUtils.remove_diacritics(text2.lower().strip())
        
        # Simple Jaccard similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        
        return similarity >= threshold
    
    @staticmethod
    def extract_technical_terms(text: str) -> List[str]:
        """Extract technical terms from computer science text"""
        # Common CS terms to look for
        cs_terms = [
            'algorithm', 'thuật toán', 'data structure', 'cấu trúc dữ liệu',
            'database', 'cơ sở dữ liệu', 'network', 'mạng', 'protocol', 'giao thức',
            'function', 'hàm', 'class', 'lớp', 'object', 'đối tượng',
            'variable', 'biến', 'array', 'mảng', 'list', 'danh sách',
            'recursion', 'đệ quy', 'iteration', 'lặp', 'sorting', 'sắp xếp',
            'searching', 'tìm kiếm', 'complexity', 'độ phức tạp',
            'inheritance', 'kế thừa', 'polymorphism', 'đa hình',
            'encapsulation', 'đóng gói', 'abstraction', 'trừu tượng'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in cs_terms:
            if term in text_lower:
                found_terms.append(term)
        
        # Extract camelCase and PascalCase identifiers
        camel_case_pattern = r'\b[a-z]+(?:[A-Z][a-z]*)+\b'
        pascal_case_pattern = r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b'
        
        camel_case_terms = re.findall(camel_case_pattern, text)
        pascal_case_terms = re.findall(pascal_case_pattern, text)
        
        found_terms.extend(camel_case_terms)
        found_terms.extend(pascal_case_terms)
        
        return list(set(found_terms))  # Remove duplicates
