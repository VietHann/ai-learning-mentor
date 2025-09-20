from typing import List, Dict, Any, Optional
import re

class CitationFormatter:
    """Format citations and references for educational responses"""
    
    def __init__(self):
        self.citation_styles = {
            'academic': 'academic',
            'informal': 'informal', 
            'inline': 'inline'
        }
    
    def format_citations(self, citations: List[Dict[str, Any]], 
                        style: str = 'academic') -> List[str]:
        """Format citations according to specified style"""
        
        if not citations:
            return []
        
        formatted_citations = []
        
        for i, citation in enumerate(citations, 1):
            if style == 'academic':
                formatted = self._format_academic_citation(citation, i)
            elif style == 'informal':
                formatted = self._format_informal_citation(citation, i)
            else:  # inline or default
                formatted = self._format_inline_citation(citation, i)
            
            formatted_citations.append(formatted)
        
        return formatted_citations
    
    def _format_academic_citation(self, citation: Dict[str, Any], index: int) -> str:
        """Format citation in academic style"""
        title = citation.get('title', 'Tài liệu không tên')
        section = citation.get('section', '')
        page = citation.get('page', '')
        doc_type = citation.get('doc_type', '')
        similarity_score = citation.get('similarity_score', 0)
        
        # Start with document title
        formatted = f"[{index}] {title}"
        
        # Add document type
        if doc_type:
            formatted += f" ({doc_type.upper()})"
        
        # Add section if available
        if section:
            formatted += f", Mục: {section}"
        
        # Add page if available
        if page:
            formatted += f", Trang: {page}"
        
        # Add relevance score
        formatted += f" (Độ liên quan: {similarity_score:.1%})"
        
        return formatted
    
    def _format_informal_citation(self, citation: Dict[str, Any], index: int) -> str:
        """Format citation in informal style"""
        title = citation.get('title', 'Tài liệu không tên')
        section = citation.get('section', '')
        page = citation.get('page', '')
        
        formatted = f"📚 Nguồn {index}: {title}"
        
        if section:
            formatted += f" - {section}"
        
        if page:
            formatted += f" (trang {page})"
        
        return formatted
    
    def _format_inline_citation(self, citation: Dict[str, Any], index: int) -> str:
        """Format citation for inline use"""
        title = citation.get('title', 'Tài liệu không tên')
        section = citation.get('section', '')
        page = citation.get('page', '')
        
        # Short format for inline use
        formatted = f"[{title}"
        
        if section:
            formatted += f", {section}"
        
        if page:
            formatted += f", tr.{page}"
        
        formatted += "]"
        
        return formatted
    
    def create_quote_with_citation(self, quote_text: str, citation: Dict[str, Any], 
                                  include_char_positions: bool = False) -> str:
        """Create a formatted quote with proper citation"""
        title = citation.get('title', 'Tài liệu không tên')
        section = citation.get('section', '')
        page = citation.get('page', '')
        
        # Format the quote
        formatted_quote = f'"{quote_text}"'
        
        # Add citation
        citation_part = f" [Nguồn: {title}"
        
        if section:
            citation_part += f", Mục: {section}"
        
        if page:
            citation_part += f", Trang: {page}"
        
        # Add character positions if available and requested
        if include_char_positions:
            start_char = citation.get('start_char')
            end_char = citation.get('end_char')
            if start_char is not None and end_char is not None:
                citation_part += f", Vị trí: {start_char}-{end_char}"
        
        citation_part += "]"
        
        return formatted_quote + citation_part
    
    def extract_quotes_from_text(self, text: str, source_content: str, 
                                citation: Dict[str, Any]) -> List[str]:
        """Extract and format quotes from response text that match source content"""
        quotes = []
        
        # Find quoted text in the response
        quote_pattern = r'"([^"]+)"'
        found_quotes = re.findall(quote_pattern, text)
        
        for quote in found_quotes:
            # Check if this quote actually comes from the source
            if self._is_quote_from_source(quote, source_content):
                formatted_quote = self.create_quote_with_citation(quote, citation)
                quotes.append(formatted_quote)
        
        return quotes
    
    def _is_quote_from_source(self, quote: str, source_content: str, 
                             similarity_threshold: float = 0.8) -> bool:
        """Check if a quote actually comes from the source content"""
        if not source_content:
            return False
        
        quote_lower = quote.lower().strip()
        source_lower = source_content.lower()
        
        # Direct match
        if quote_lower in source_lower:
            return True
        
        # Fuzzy match for slight variations
        quote_words = quote_lower.split()
        
        if len(quote_words) < 3:  # Too short to be meaningful
            return False
        
        # Check if most words from quote appear in source
        matching_words = sum(1 for word in quote_words if word in source_lower)
        similarity = matching_words / len(quote_words)
        
        return similarity >= similarity_threshold
    
    def format_reference_list(self, citations: List[Dict[str, Any]]) -> str:
        """Create a formatted reference list"""
        if not citations:
            return "Không có tài liệu tham khảo."
        
        reference_list = "## Tài liệu tham khảo:\n\n"
        
        for i, citation in enumerate(citations, 1):
            title = citation.get('title', 'Tài liệu không tên')
            section = citation.get('section', '')
            page = citation.get('page', '')
            doc_type = citation.get('doc_type', '')
            content_snippet = citation.get('content_snippet', '')
            
            reference = f"{i}. **{title}**"
            
            if doc_type:
                reference += f" ({doc_type})"
            
            if section:
                reference += f"\n   - Mục: {section}"
            
            if page:
                reference += f"\n   - Trang: {page}"
            
            if content_snippet:
                reference += f"\n   - Nội dung: *{content_snippet}*"
            
            reference_list += reference + "\n\n"
        
        return reference_list
    
    def validate_citation_quality(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of citations"""
        if not citations:
            return {
                'total_citations': 0,
                'quality_score': 0,
                'issues': ['Không có tài liệu tham khảo']
            }
        
        issues = []
        quality_scores = []
        
        for citation in citations:
            score = 0
            
            # Check if title exists
            if citation.get('title'):
                score += 25
            else:
                issues.append('Thiếu tiêu đề tài liệu')
            
            # Check if section exists
            if citation.get('section'):
                score += 25
            
            # Check if page exists (for documents that should have pages)
            doc_type = citation.get('doc_type', '')
            if doc_type == 'pdf' and citation.get('page'):
                score += 25
            elif doc_type != 'pdf':
                score += 25  # Not expected to have pages
            
            # Check similarity score
            similarity = citation.get('similarity_score', 0)
            if similarity > 0.5:
                score += 25
            elif similarity > 0.3:
                score += 15
            else:
                issues.append(f'Độ liên quan thấp cho "{citation.get("title", "tài liệu")}"')
            
            quality_scores.append(score)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'total_citations': len(citations),
            'quality_score': avg_quality,
            'issues': issues,
            'individual_scores': quality_scores
        }
    
    def extract_precise_quotes_from_chunk(self, chunk_data: Dict[str, Any], 
                                         quote_text: str) -> Optional[Dict[str, Any]]:
        """Extract precise quote information from chunk using page mapping and quote spans"""
        if not chunk_data:
            return None
        
        quote_spans = chunk_data.get('quote_spans', [])
        chunk_content = chunk_data.get('content', '')
        
        # Find the best matching quote span for the given quote text
        best_match = None
        best_similarity = 0
        
        for span in quote_spans:
            span_text = span.get('text', '')
            similarity = self._calculate_quote_similarity(quote_text, span_text)
            
            if similarity > best_similarity and similarity > 0.7:  # 70% similarity threshold
                best_match = span
                best_similarity = similarity
        
        if best_match:
            return {
                'quote_text': best_match.get('text', quote_text),
                'start_char': best_match.get('start_char'),
                'end_char': best_match.get('end_char'),
                'length': best_match.get('length'),
                'similarity_score': best_similarity,
                'chunk_id': chunk_data.get('chunk_id'),
                'source_file': chunk_data.get('source_id'),
                'page': chunk_data.get('page'),
                'section': chunk_data.get('section')
            }
        
        # Fallback: try to find quote in chunk content directly
        return self._find_quote_in_content(quote_text, chunk_content, chunk_data)
    
    def _calculate_quote_similarity(self, quote1: str, quote2: str) -> float:
        """Calculate similarity between two quotes using word overlap"""
        if not quote1 or not quote2:
            return 0.0
        
        words1 = set(quote1.lower().split())
        words2 = set(quote2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _find_quote_in_content(self, quote_text: str, content: str, 
                              chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find quote text in chunk content and calculate positions"""
        if not content or not quote_text:
            return None
        
        quote_lower = quote_text.lower().strip()
        content_lower = content.lower()
        
        # Try exact match first
        start_pos = content_lower.find(quote_lower)
        if start_pos != -1:
            base_offset = chunk_data.get('start_char', 0)
            
            return {
                'quote_text': content[start_pos:start_pos + len(quote_text)],
                'start_char': base_offset + start_pos,
                'end_char': base_offset + start_pos + len(quote_text),
                'length': len(quote_text),
                'similarity_score': 1.0,
                'chunk_id': chunk_data.get('chunk_id'),
                'source_file': chunk_data.get('source_id'),
                'page': chunk_data.get('page'),
                'section': chunk_data.get('section')
            }
        
        return None
    
    def create_inline_citation_with_highlight(self, quote_info: Dict[str, Any], 
                                            citation: Dict[str, Any]) -> Dict[str, Any]:
        """Create inline citation with highlighting information for UI"""
        quote_text = quote_info.get('quote_text', '')
        start_char = quote_info.get('start_char')
        end_char = quote_info.get('end_char')
        
        # Create base citation
        formatted_citation = self.create_quote_with_citation(
            quote_text, citation, include_char_positions=True
        )
        
        # Add highlighting metadata (for UI consumption)
        highlight_info = {
            'type': 'highlighted_quote',
            'quote': quote_text,
            'citation': formatted_citation,
            'start_char': start_char,
            'end_char': end_char,
            'source_file': quote_info.get('source_file'),
            'chunk_id': quote_info.get('chunk_id'),
            'similarity': quote_info.get('similarity_score', 0)
        }
        
        return highlight_info
    
    def process_response_with_precise_citations(self, response_text: str, 
                                              search_results: List[Dict[str, Any]],
                                              page_mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process response text to add precise citations with highlighting"""
        # Extract all quoted text from response
        quote_pattern = r'"([^"]{10,})"'  # Quotes with at least 10 characters
        found_quotes = re.findall(quote_pattern, response_text)
        
        precise_citations = []
        highlighted_quotes = []
        
        for quote_text in found_quotes:
            best_match = None
            best_source = None
            best_score = 0
            
            # Find best matching chunk for this quote
            for result in search_results:
                chunk_id = result.get('chunk_id')
                if not chunk_id:
                    continue
                
                chunk_mapping = page_mapping.get(chunk_id)
                if not chunk_mapping:
                    continue
                
                # Extract precise quote information
                quote_info = self.extract_precise_quotes_from_chunk(
                    {**result, **chunk_mapping}, quote_text
                )
                
                if quote_info and quote_info.get('similarity_score', 0) > best_score:
                    best_match = quote_info
                    best_source = result  # Track the matching source
                    best_score = quote_info['similarity_score']
            
            if best_match and best_source:
                # Merge source metadata with precise location data
                citation_data = {
                    **best_source,  # Title, doc_type, etc. from search result
                    **best_match    # start_char, end_char, etc. from quote info
                }
                
                # Create inline citation with highlighting
                highlight_info = self.create_inline_citation_with_highlight(
                    best_match, citation_data  # Use merged citation data
                )
                
                highlighted_quotes.append(highlight_info)
                precise_citations.append(best_match)
        
        return {
            'original_response': response_text,
            'highlighted_quotes': highlighted_quotes,
            'precise_citations': precise_citations,
            'total_quotes_found': len(found_quotes),
            'total_matches': len(precise_citations)
        }
    
    def suggest_citation_improvements(self, citations: List[Dict[str, Any]]) -> List[str]:
        """Suggest improvements for citation quality"""
        suggestions = []
        
        validation = self.validate_citation_quality(citations)
        
        if validation['total_citations'] == 0:
            suggestions.append("Thêm tài liệu tham khảo để hỗ trợ câu trả lời")
            return suggestions
        
        if validation['quality_score'] < 70:
            suggestions.append("Cải thiện chất lượng trích dẫn bằng cách thêm thông tin chi tiết hơn")
        
        if validation['total_citations'] < 3:
            suggestions.append("Thêm nhiều nguồn tham khảo để tăng độ tin cậy")
        
        # Check for diversity in sources
        doc_types = [citation.get('doc_type', '') for citation in citations]
        unique_types = set(doc_types)
        
        if len(unique_types) < 2 and len(citations) > 2:
            suggestions.append("Sử dụng đa dạng loại tài liệu (PDF, code, markdown) để có góc nhìn toàn diện")
        
        # Check for recent vs old sources
        sources = [citation.get('title', '') for citation in citations]
        unique_sources = set(sources)
        
        if len(unique_sources) < len(citations):
            suggestions.append("Tránh trích dẫn trùng lặp từ cùng một nguồn")
        
        return suggestions
