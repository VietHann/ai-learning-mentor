import os
import re
import hashlib
import ast
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

import markdown

@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    target_tokens_min: int = 400
    target_tokens_max: int = 800
    overlap_percentage: float = 0.125  # 12.5%
    respect_boundaries: bool = True


class DocumentProcessor:
    """Process various document types and extract structured content with precise tracking"""
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.config = chunking_config or ChunkingConfig()
        self.chunk_size = self.config.target_tokens_max  # words approximation
        self.overlap_size = int(self.chunk_size * self.config.overlap_percentage)
        
        # Page and char offset mapping: {chunk_id: {page, section, start_char, end_char, quote_spans}}
        self.page_map: Dict[str, Dict[str, Any]] = {}
        
        # Document-level metadata cache
        self.document_cache: Dict[str, Dict[str, Any]] = {}
    
    def process_document(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Process a document and return chunks with metadata"""
        
        if file_type == "application/pdf":
            return self._process_pdf(file_path)
        elif file_type in ["text/markdown", "text/plain"] or file_path.endswith('.md'):
            return self._process_markdown(file_path)
        elif file_path.endswith(('.py', '.java', '.cpp', '.c', '.js', '.html', '.css')):
            return self._process_code(file_path)
        else:
            return self._process_text(file_path)
    
    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents and return combined chunks with metadata"""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # Auto-detect file type based on extension
                if file_path.endswith('.pdf'):
                    file_type = "application/pdf"
                elif file_path.endswith('.md'):
                    file_type = "text/markdown"
                elif file_path.endswith('.txt'):
                    file_type = "text/plain"
                else:
                    file_type = "text/plain"  # Default fallback
                
                # Process single document
                chunks = self.process_document(file_path, file_type)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return all_chunks
    
    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process PDF files"""
        chunks = []
        
        if pdfplumber:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                page_map = {}
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    page_text = self._clean_text(page_text)
                    
                    page_start = len(full_text)
                    full_text += f"\n\n[PAGE {page_num}]\n{page_text}"
                    page_end = len(full_text)
                    
                    page_map[page_num] = (page_start, page_end)
                
                chunks = self._chunk_text_with_headings(
                    full_text, 
                    os.path.basename(file_path),
                    "pdf",
                    page_map
                )
        
        elif PyPDF2:
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                page_map = {}
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    page_text = self._clean_text(page_text)
                    
                    page_start = len(full_text)
                    full_text += f"\n\n[PAGE {page_num}]\n{page_text}"
                    page_end = len(full_text)
                    
                    page_map[page_num] = (page_start, page_end)
                
                chunks = self._chunk_text_with_headings(
                    full_text,
                    os.path.basename(file_path),
                    "pdf", 
                    page_map
                )
        else:
            raise ImportError("No PDF processing library available. Install PyPDF2 or pdfplumber.")
        
        return chunks
    
    def _process_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Markdown files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Convert markdown to HTML to extract structure
        md = markdown.Markdown(extensions=['toc', 'tables', 'fenced_code'])
        html_content = md.convert(content)
        
        chunks = self._chunk_text_with_headings(
            content,
            os.path.basename(file_path),
            "markdown",
            None
        )
        
        return chunks
    
    def _process_code(self, file_path: str) -> List[Dict[str, Any]]:
        """Process code files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        chunks = self._chunk_code(
            content,
            os.path.basename(file_path),
            self._get_language_from_extension(file_path)
        )
        
        return chunks
    
    def _process_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Process plain text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        chunks = self._chunk_text_with_headings(
            content,
            os.path.basename(file_path),
            "text",
            None
        )
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+/\d+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
    
    def _chunk_text_with_headings(self, text: str, filename: str, doc_type: str, page_map: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Chunk text while respecting heading structure with absolute offsets"""
        chunks = []
        
        # Find section boundaries in the original text
        heading_pattern = r'^(#{1,4})\s+(.+)$'
        sections = []
        current_pos = 0
        current_heading = ""
        current_level = 0
        
        # Split text into lines and track positions
        lines = text.split('\n')
        line_positions = []
        cumulative_pos = 0
        
        for line in lines:
            line_positions.append(cumulative_pos)
            cumulative_pos += len(line) + 1  # +1 for \n
        
        # Find heading boundaries
        for line_idx, line in enumerate(lines):
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                heading = match.group(2).strip()
                
                # Save previous section
                if current_heading or line_idx == 0:
                    section_start = line_positions[current_pos] if current_pos < len(line_positions) else 0
                    section_end = line_positions[line_idx] if line_idx < len(line_positions) else len(text)
                    section_text_raw = text[section_start:section_end]
                    section_text = section_text_raw.strip()
                    
                    # Adjust base offset for stripped leading whitespace
                    leading_trim = len(section_text_raw) - len(section_text_raw.lstrip())
                    adjusted_start = section_start + leading_trim
                    
                    if len(section_text) > 100 or not current_heading:  # Process content sections
                        sections.append({
                            'text': section_text,
                            'heading': current_heading,
                            'level': current_level,
                            'start_pos': adjusted_start,  # Use adjusted position
                            'end_pos': adjusted_start + len(section_text)
                        })
                
                current_heading = heading
                current_level = level
                current_pos = line_idx
        
        # Add final section
        if current_pos < len(lines):
            section_start = line_positions[current_pos] if current_pos < len(line_positions) else 0
            section_text_raw = text[section_start:]
            section_text = section_text_raw.strip()
            
            if len(section_text) > 100:
                # Adjust for stripped leading whitespace
                leading_trim = len(section_text_raw) - len(section_text_raw.lstrip())
                adjusted_start = section_start + leading_trim
                
                sections.append({
                    'text': section_text,
                    'heading': current_heading,
                    'level': current_level,
                    'start_pos': adjusted_start,
                    'end_pos': adjusted_start + len(section_text)
                })
        
        # If no sections found, use entire text
        if not sections:
            sections.append({
                'text': text,
                'heading': "",
                'level': 0,
                'start_pos': 0,
                'end_pos': len(text)
            })
        
        # Process each section with absolute offsets
        for section in sections:
            section_chunks = self._create_chunks_from_text_with_offset(
                original_text=text,
                section_text=section['text'],
                base_offset=section['start_pos'],
                filename=filename,
                doc_type=doc_type,
                heading=section['heading'],
                level=section['level'],
                page_map=page_map,
                course=None,
                semester=None
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _create_chunks_from_text(self, text: str, filename: str, doc_type: str, 
                                heading: str, level: int, page_map: Optional[Dict] = None, 
                                course: Optional[str] = None, semester: Optional[str] = None) -> List[Dict[str, Any]]:
        """Legacy method - redirects to new implementation with zero base offset"""
        return self._create_chunks_from_text_with_offset(
            original_text=text,
            section_text=text,
            base_offset=0,
            filename=filename,
            doc_type=doc_type,
            heading=heading,
            level=level,
            page_map=page_map,
            course=course,
            semester=semester
        )
    
    def _create_chunks_from_text_with_offset(self, original_text: str, section_text: str, base_offset: int,
                                           filename: str, doc_type: str, heading: str, level: int, 
                                           page_map: Optional[Dict] = None, course: Optional[str] = None, 
                                           semester: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create overlapping chunks with absolute char offset tracking"""
        chunks = []
        
        # Use regex to find precise word boundaries in section text
        word_boundaries = []
        
        # Find all word boundaries with their positions relative to section
        for match in re.finditer(r'\S+', section_text):
            word_boundaries.append({
                'word': match.group(),
                'start': match.start(),  # relative to section
                'end': match.end()       # relative to section
            })
        
        chunk_size_words = self.chunk_size
        overlap_words = self.overlap_size
        
        for i in range(0, len(word_boundaries), chunk_size_words - overlap_words):
            chunk_boundaries = word_boundaries[i:i + chunk_size_words]
            if not chunk_boundaries:
                continue
                
            # Get positions relative to section
            section_start_pos = chunk_boundaries[0]['start']
            section_end_pos = chunk_boundaries[-1]['end']
            
            # Convert to absolute positions in original document
            absolute_start_pos = base_offset + section_start_pos
            absolute_end_pos = base_offset + section_end_pos
            
            # Extract chunk text from section
            chunk_text = section_text[section_start_pos:section_end_pos]
            
            # Generate unique chunk ID
            chunk_id = str(uuid.uuid4())
            
            # Determine page number using absolute position
            page_info = self._find_page_and_position(chunk_text, absolute_start_pos, page_map) if page_map else None
            page_num = page_info['page'] if page_info else None
            
            # Extract quote spans with absolute base offset
            quote_spans = self._extract_quote_spans(chunk_text, absolute_start_pos)
            
            # Enhanced metadata schema with absolute positions
            chunk_metadata = {
                'chunk_id': chunk_id,
                'content': chunk_text,
                'source_id': filename,
                'title': os.path.splitext(filename)[0],  # Remove extension
                'section': heading,
                'section_level': level,
                'page': page_num,
                'course': course or 'Unknown',
                'semester': semester or 'Unknown',
                'language': self._detect_language(chunk_text),
                'doc_type': doc_type,
                'keywords': self._extract_keywords(chunk_text),
                'hash': hashlib.md5(chunk_text.encode()).hexdigest(),
                'created_at': datetime.now().isoformat(),
                'start_char': absolute_start_pos,  # Absolute position
                'end_char': absolute_end_pos,      # Absolute position
                'chunk_index': len(chunks)
            }
            
            # Store in page map for precise tracking with absolute offsets
            self.page_map[chunk_id] = {
                'page': page_num,
                'section': heading,
                'start_char': absolute_start_pos,  # Absolute position
                'end_char': absolute_end_pos,      # Absolute position
                'quote_spans': quote_spans
            }
            
            chunks.append(chunk_metadata)
        
        return chunks
    
    def _chunk_code(self, code: str, filename: str, language: str) -> List[Dict[str, Any]]:
        """Chunk code files by logical blocks"""
        chunks = []
        
        # Simple code chunking by functions/classes
        if language in ['python', 'java', 'cpp', 'c']:
            chunks.extend(self._chunk_by_functions(code, filename, language))
        else:
            # Fallback to text chunking
            chunks.extend(self._create_chunks_from_text(code, filename, "code", "", 0, None, None, None))
        
        return chunks
    
    def _chunk_by_functions(self, code: str, filename: str, language: str) -> List[Dict[str, Any]]:
        """Chunk code by function/class boundaries"""
        chunks = []
        lines = code.split('\n')
        
        current_chunk = []
        current_function = ""
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect function/class definitions
            if language == 'python':
                if line_stripped.startswith('def ') or line_stripped.startswith('class '):
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        chunks.append(self._create_code_chunk(
                            chunk_text, filename, language, current_function, len(chunks), None, None, len('\n'.join(current_chunk[:-len(chunk_text.split('\n'))]))
                        ))
                    current_chunk = [line]
                    current_function = line_stripped.split('(')[0].replace('def ', '').replace('class ', '')
                else:
                    current_chunk.append(line)
            else:
                # For other languages, use simpler heuristics
                current_chunk.append(line)
                if len(current_chunk) > 50:  # Arbitrary chunk size for code
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(self._create_code_chunk(
                        chunk_text, filename, language, current_function, len(chunks), None, None, 0  # Approximate offset
                    ))
                    current_chunk = []
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_code_chunk(
                chunk_text, filename, language, current_function, len(chunks), None, None, 0  # Approximate offset
            ))
        
        return chunks
    
    def _create_code_chunk(self, code: str, filename: str, language: str, 
                          function_name: str, index: int, course: Optional[str] = None, 
                          semester: Optional[str] = None, base_offset: int = 0) -> Dict[str, Any]:
        """Create a code chunk with metadata and call graph analysis"""
        
        chunk_id = str(uuid.uuid4())
        call_graph = self._extract_call_graph(code, language)
        
        # Calculate absolute char positions
        start_char = base_offset
        end_char = base_offset + len(code)
        
        # Extract quote spans for code (function/class definitions) with absolute offset
        quote_spans = self._extract_code_quote_spans_with_offset(code, language, base_offset)
        
        # Store in page map with absolute positions
        self.page_map[chunk_id] = {
            'page': None,  # Code files don't have pages
            'section': function_name,
            'start_char': start_char,
            'end_char': end_char,
            'quote_spans': quote_spans,
            'call_graph': call_graph
        }
        
        return {
            'chunk_id': chunk_id,
            'content': code,
            'source_id': filename,
            'title': os.path.splitext(filename)[0],
            'section': function_name,
            'section_level': 1,
            'page': None,
            'course': course or 'Programming',
            'semester': semester or 'Unknown',
            'doc_type': 'code',
            'language': language,
            'keywords': self._extract_code_keywords(code, language),
            'hash': hashlib.md5(code.encode()).hexdigest(),
            'created_at': datetime.now().isoformat(),
            'start_char': start_char,
            'end_char': end_char,
            'chunk_index': index,
            'call_graph_summary': self._summarize_call_graph(call_graph)
        }
    
    def _extract_code_quote_spans(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract quotable spans from code (function signatures, class definitions)"""
        quote_spans = []
        lines = code.split('\n')
        
        patterns = {
            'python': [r'^\s*(def\s+\w+.*?:)', r'^\s*(class\s+\w+.*?:)'],
            'java': [r'^\s*(public\s+.*?\{)', r'^\s*(private\s+.*?\{)', r'^\s*(class\s+\w+.*?\{)'],
            'c': [r'^\s*(\w+\s+\w+\s*\([^)]*\)\s*\{)'],
            'cpp': [r'^\s*(\w+\s+\w+\s*\([^)]*\)\s*\{)'],
        }
        
        lang_patterns = patterns.get(language, [])
        current_pos = 0
        
        for i, line in enumerate(lines):
            for pattern in lang_patterns:
                match = re.search(pattern, line)
                if match:
                    quote_spans.append({
                        'sentence_id': len(quote_spans),
                        'text': match.group(1).strip(),
                        'start_char': current_pos,
                        'end_char': current_pos + len(match.group(1)),
                        'length': len(match.group(1)),
                        'line_number': i + 1,
                        'type': 'code_definition'
                    })
            current_pos += len(line) + 1  # +1 for newline
        
        return quote_spans
    
    def _summarize_call_graph(self, call_graph: Dict[str, Any]) -> str:
        """Create a summary of call graph for metadata"""
        summary_parts = []
        
        if call_graph.get('functions'):
            func_names = [f['name'] for f in call_graph['functions'][:3]]  # Top 3
            summary_parts.append(f"Functions: {', '.join(func_names)}")
        
        if call_graph.get('classes'):
            class_names = [c['name'] for c in call_graph['classes'][:2]]  # Top 2  
            summary_parts.append(f"Classes: {', '.join(class_names)}")
        
        if call_graph.get('imports'):
            import_count = len(call_graph['imports'])
            summary_parts.append(f"Imports: {import_count}")
        
        complexity = call_graph.get('complexity_score', 0)
        summary_parts.append(f"Complexity: {complexity}")
        
        return " | ".join(summary_parts) if summary_parts else "Simple code block"
    
    def _get_language_from_extension(self, filename: str) -> str:
        """Get programming language from file extension"""
        ext = os.path.splitext(filename)[1].lower()
        
        language_map = {
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown'
        }
        
        return language_map.get(ext, 'text')
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Count Vietnamese characters
        vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower()))
        total_chars = len(re.findall(r'[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower()))
        
        if total_chars > 0 and vietnamese_chars / total_chars > 0.1:
            return 'vietnamese'
        else:
            return 'english'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        
        # Common stop words to filter out
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over', 'this', 'that', 'these', 'those', 'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return top 10 most frequent keywords
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(10)]
    
    def _extract_code_keywords(self, code: str, language: str) -> List[str]:
        """Extract keywords from code"""
        # Language-specific keywords
        python_keywords = {'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'return', 'yield', 'lambda', 'with', 'as'}
        java_keywords = {'public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'import', 'package', 'if', 'else', 'for', 'while', 'try', 'catch', 'return', 'new'}
        
        keywords_map = {
            'python': python_keywords,
            'java': java_keywords
        }
        
        lang_keywords = keywords_map.get(language, set())
        
        # Extract identifiers and keywords
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        found_keywords = [word for word in identifiers if word in lang_keywords]
        
        return found_keywords[:10]
    
    def _find_page_number(self, text: str, page_map: Dict) -> int | None:
        """Find which page the text belongs to"""
        if not page_map:
            return None
        
        # Simple heuristic: find PAGE markers in text
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        if page_match:
            return int(page_match.group(1))
        
        return None
    
    def _find_page_and_position(self, text: str, char_pos: int, page_map: Dict) -> Optional[Dict[str, Any]]:
        """Find precise page and position information for a text chunk"""
        if not page_map:
            return None
        
        # Find PAGE markers in text for basic page detection
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        if page_match:
            page_num = int(page_match.group(1))
            return {
                'page': page_num,
                'char_offset': char_pos,
                'page_section': self._extract_section_from_text(text)
            }
        
        # Fallback: find closest page based on character position
        for page_num, (start_pos, end_pos) in page_map.items():
            if start_pos <= char_pos <= end_pos:
                return {
                    'page': page_num,
                    'char_offset': char_pos - start_pos,
                    'page_section': self._extract_section_from_text(text)
                }
        
        return None
    
    def _extract_quote_spans(self, text: str, base_offset: int) -> List[Dict[str, Any]]:
        """Extract sentence-level spans for precise citation quoting"""
        quote_spans = []
        
        # Use regex to find sentence boundaries with precise positions
        sentence_pattern = r'[.!?]+\s+|[.!?]+$'
        sentences = re.split(sentence_pattern, text)
        
        # Find actual positions of sentences in text
        current_search_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Find exact position of this sentence in the text
            sentence_start_in_text = text.find(sentence, current_search_pos)
            if sentence_start_in_text == -1:
                continue  # Skip if not found
            
            # Calculate absolute positions
            start_char = base_offset + sentence_start_in_text
            end_char = start_char + len(sentence)
            
            quote_spans.append({
                'sentence_id': i,
                'text': sentence,
                'start_char': start_char,
                'end_char': end_char,
                'length': len(sentence),
                'relative_start': sentence_start_in_text,  # Position within chunk
                'relative_end': sentence_start_in_text + len(sentence)
            })
            
            current_search_pos = sentence_start_in_text + len(sentence)
        
        return quote_spans
    
    def _extract_section_from_text(self, text: str) -> str:
        """Extract section name from text content"""
        # Look for common section patterns
        section_patterns = [
            r'\n#+\s*([^\n]+)',  # Markdown headers
            r'\n([A-Z][^\n]{10,50})\n',  # Capitalized titles
            r'([0-9]+\.?[0-9]*\.?\s+[A-Z][^\n]{5,40})',  # Numbered sections
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return 'Unknown Section'
    
    def _extract_call_graph(self, code: str, language: str) -> Dict[str, Any]:
        """Extract simple call graph from code using AST analysis"""
        call_graph = {
            'functions': [],
            'classes': [],
            'imports': [],
            'calls': [],
            'complexity_score': 0
        }
        
        try:
            if language == 'python':
                call_graph = self._analyze_python_ast(code)
            elif language == 'java':
                call_graph = self._analyze_java_code(code)
            else:
                # Fallback: simple regex-based analysis
                call_graph = self._analyze_code_regex(code, language)
        except Exception as e:
            print(f"Warning: Call graph extraction failed: {e}")
        
        return call_graph
    
    def _analyze_python_ast(self, code: str) -> Dict[str, Any]:
        """Analyze Python code using AST"""
        call_graph = {
            'functions': [],
            'classes': [],
            'imports': [],
            'calls': [],
            'complexity_score': 0
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    call_graph['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': len(node.args.args)
                    })
                elif isinstance(node, ast.ClassDef):
                    call_graph['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        call_graph['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        call_graph['imports'].append(node.module)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    call_graph['calls'].append(node.func.id)
            
            # Simple complexity score
            call_graph['complexity_score'] = (
                len(call_graph['functions']) * 2 + 
                len(call_graph['classes']) * 3 +
                len(call_graph['calls'])
            )
            
        except SyntaxError as e:
            print(f"Python AST parse error: {e}")
        
        return call_graph
    
    def _analyze_java_code(self, code: str) -> Dict[str, Any]:
        """Analyze Java code using regex patterns"""
        call_graph = {
            'functions': [],
            'classes': [],
            'imports': [],
            'calls': [],
            'complexity_score': 0
        }
        
        # Extract classes
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?class\s+(\w+)'
        classes = re.findall(class_pattern, code)
        call_graph['classes'] = [{'name': cls, 'line': 0} for cls in classes]
        
        # Extract methods
        method_pattern = r'(?:public\s+|private\s+|protected\s+|static\s+)*\w+\s+(\w+)\s*\([^)]*\)\s*\{'
        methods = re.findall(method_pattern, code)
        call_graph['functions'] = [{'name': method, 'line': 0, 'args': 0} for method in methods]
        
        # Extract imports
        import_pattern = r'import\s+([^;]+);'
        imports = re.findall(import_pattern, code)
        call_graph['imports'] = imports
        
        # Simple complexity score
        call_graph['complexity_score'] = len(classes) * 3 + len(methods) * 2
        
        return call_graph
    
    def _analyze_code_regex(self, code: str, language: str) -> Dict[str, Any]:
        """Fallback regex-based code analysis for unsupported languages"""
        call_graph = {
            'functions': [],
            'classes': [],
            'imports': [],
            'calls': [],
            'complexity_score': 0
        }
        
        # Generic function detection
        func_patterns = {
            'c': r'\w+\s+(\w+)\s*\([^)]*\)\s*\{',
            'cpp': r'\w+\s+(\w+)\s*\([^)]*\)\s*\{',
            'javascript': r'function\s+(\w+)\s*\([^)]*\)',
        }
        
        pattern = func_patterns.get(language)
        if pattern:
            functions = re.findall(pattern, code)
            call_graph['functions'] = [{'name': func, 'line': 0} for func in functions]
        
        call_graph['complexity_score'] = len(call_graph['functions'])
        
        return call_graph
    
    def _extract_code_quote_spans_with_offset(self, code: str, language: str, base_offset: int) -> List[Dict[str, Any]]:
        """Extract quotable spans from code with absolute offset"""
        quote_spans = self._extract_code_quote_spans(code, language)
        
        # Convert to absolute offsets
        absolute_spans = []
        for span in quote_spans:
            absolute_spans.append({
                **span,
                'start_char': span['start_char'] + base_offset,
                'end_char': span['end_char'] + base_offset
            })
        
        return absolute_spans
    
    def get_page_map(self) -> Dict[str, Dict[str, Any]]:
        """Get the current page mapping for all processed chunks"""
        return self.page_map.copy()
    
    def get_chunk_location(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get precise location information for a specific chunk"""
        return self.page_map.get(chunk_id)
    
    def clear_cache(self):
        """Clear all cached page mapping and document data"""
        self.page_map.clear()
        self.document_cache.clear()
    
    def get_chunking_config(self) -> ChunkingConfig:
        """Get current chunking configuration"""
        return self.config
    
    def update_chunking_config(self, config: ChunkingConfig):
        """Update chunking configuration"""
        self.config = config
        self.chunk_size = config.target_tokens_max
        self.overlap_size = int(self.chunk_size * config.overlap_percentage)
