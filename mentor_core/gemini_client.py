import os
import json
import logging
from typing import List, Dict, Any, Tuple

import google.generativeai as genai

class GeminiClient:
    """Client for interacting with Gemini API for educational responses"""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def generate_educational_response(self, question: str, question_type: str,
                                    relevant_docs: List[Dict[str, Any]],
                                    academic_mode: bool = True,
                                    integrity_mode: str = "academic",
                                    language: str = "Vietnamese") -> Tuple[str, List[Dict[str, Any]]]:
        """Generate educational response with proper citations"""
        
        # Validate context quality first
        validation_result = self._validate_context_quality(question, relevant_docs)
        if not validation_result['is_sufficient']:
            insufficient_msg = self._generate_insufficient_context_response(
                question, validation_result, language
            )
            return insufficient_msg, []
        
        # Build context from relevant documents
        context = self._build_context(relevant_docs)
        citations = self._extract_citations(relevant_docs)
        
        # Create prompt based on question type and integrity mode
        prompt = self._create_educational_prompt(
            question, question_type, context, academic_mode, integrity_mode, language
        )
        
        try:
            response = self.model.generate_content(prompt)
            
            response_text = response.text or "Xin lỗi, tôi không thể tạo ra câu trả lời phù hợp."
            
            return response_text, citations
            
        except Exception as e:
            error_msg = f"Lỗi khi tạo câu trả lời: {str(e)}"
            return error_msg, []
    
    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Build context string from relevant documents"""
        if not relevant_docs:
            return "Không có tài liệu tham khảo nào được tìm thấy."
        
        context_parts = []
        
        for i, doc in enumerate(relevant_docs[:6], 1):  # Limit to top 6 documents
            title = doc.get('title', 'Tài liệu không tên')
            section = doc.get('section', '')
            page = doc.get('page', '')
            content = doc.get('content', '')
            
            # Format source info
            source_info = f"[Nguồn {i}: {title}"
            if section:
                source_info += f", Mục: {section}"
            if page:
                source_info += f", Trang: {page}"
            source_info += "]"
            
            # Truncate content if too long
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_citations(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from documents"""
        citations = []
        
        for doc in relevant_docs[:6]:
            citation = {
                'title': doc.get('title', 'Tài liệu không tên'),
                'section': doc.get('section', ''),
                'page': doc.get('page', ''),
                'doc_type': doc.get('doc_type', ''),
                'similarity_score': doc.get('similarity_score', 0),
                'content_snippet': doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
            }
            citations.append(citation)
        
        return citations
    
    def _create_educational_prompt(self, question: str, question_type: str,
                                 context: str, academic_mode: bool, 
                                 integrity_mode: str = "academic", language: str = "Vietnamese") -> str:
        """Create appropriate prompt based on question type and mode"""
        
        base_instructions = """
Bạn là một mentor ảo AI chuyên hỗ trợ sinh viên Công nghệ Thông tin. Nhiệm vụ của bạn là:

1. LUÔN trích dẫn nguồn chính xác: Khi sử dụng thông tin từ tài liệu, hãy trích dẫn đúng định dạng [Nguồn X: Tên tài liệu, Mục: Y, Trang: Z] và đặt trong dấu ngoặc kép câu gốc được tham khảo.

2. Nếu không đủ thông tin: Trả lời "Không đủ ngữ cảnh để trả lời chính xác. Bạn có thể cung cấp thêm tài liệu về [chủ đề cụ thể] không?"

3. Phong cách trả lời:
   - Giải thích từng bước một cách rõ ràng
   - Đưa ra ví dụ minh họa khi có thể
   - Gợi ý bài tập luyện tập liên quan
"""
        
        # Enhanced integrity mode instructions
        if integrity_mode == "exam":
            academic_instructions = """
4. CHẾ ĐỘ THI CỬ (Exam Mode - STRICT):
   - TUYỆT ĐỐI KHÔNG viết bất kỳ dòng code nào
   - KHÔNG đưa ra thuật toán cụ thể hoặc pseudocode chi tiết
   - CHỈ gợi ý các khái niệm tổng quát và hướng suy nghĩ
   - KHÔNG giải thích các bước implementation
   - CHỈ khuyến khích tự suy nghĩ và tìm hiểu thêm
   - Nếu được hỏi về code/thuật toán: "Trong chế độ thi, tôi chỉ có thể gợi ý hướng suy nghĩ tổng quát"
   - Khuyến khích học sinh tự nghiên cứu thêm từ tài liệu
"""
        elif integrity_mode == "academic":
            academic_instructions = """
4. CHẾ ĐỘ HỌC THUẬT (Academic Mode):
   - KHÔNG viết code hoàn chỉnh cho bài tập/đồ án
   - Có thể đưa ra pseudocode hoặc outline tổng quát
   - Chỉ đưa ra gợi ý, ý tưởng, cách tiếp cận
   - Giúp kiểm tra logic và review giải pháp
   - Hướng dẫn cách debug và tìm lỗi
   - Gợi ý test cases để kiểm tra
   - Giải thích khái niệm nhưng không làm thay học sinh
"""
        else:  # normal mode
            academic_instructions = """
4. CHẾ ĐỘ HỖ TRỢ CODING ĐẦY ĐỦ (Normal Mode):
   - Có thể cung cấp code mẫu và giải pháp hoàn chỉnh
   - Giải thích chi tiết từng dòng code
   - Đưa ra nhiều cách tiếp cận khác nhau
   - Cung cấp implementation details và best practices
   - Hỗ trợ debugging với code snippets cụ thể
"""
        
        question_specific = ""
        if question_type == "theory_explanation":
            question_specific = "Đây là câu hỏi về lý thuyết. Hãy giải thích khái niệm một cách dễ hiểu với ví dụ thực tế."
        elif question_type == "code_debugging":
            question_specific = "Đây là câu hỏi về debug code. Hãy phân tích lỗi và đưa ra hướng giải quyết."
        elif question_type == "exercise":
            question_specific = "Đây là bài tập. Hãy hướng dẫn cách tiếp cận và các bước giải quyết."
        elif question_type == "definition":
            question_specific = "Đây là câu hỏi định nghĩa. Hãy giải thích rõ ràng và chính xác."
        
        language_instruction = ""
        if language == "English":
            language_instruction = "Trả lời bằng tiếng Anh."
        elif language == "Vietnamese":
            language_instruction = "Trả lời bằng tiếng Việt."
        else:
            language_instruction = "Trả lời bằng ngôn ngữ phù hợp với câu hỏi."
        
        prompt = f"""{base_instructions}
{academic_instructions}
{question_specific}
{language_instruction}

NGỮ CẢNH TÀI LIỆU:
{context}

CÂU HỎI CỦA SINH VIÊN:
{question}

Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. Nhớ luôn trích dẫn nguồn chính xác.
"""
        
        return prompt
    
    def classify_question_with_ai(self, question: str) -> str:
        """Use Gemini to classify question type"""
        classification_prompt = f"""
Phân loại câu hỏi sau vào một trong các loại:
- theory_explanation: Giải thích lý thuyết, khái niệm
- code_debugging: Tìm và sửa lỗi code
- exercise: Bài tập, đề bài cần giải
- definition: Định nghĩa thuật ngữ
- general: Câu hỏi chung

Câu hỏi: {question}

Trả lời chỉ với một từ khóa phân loại:
"""
        
        try:
            response = self.model.generate_content(classification_prompt)
            
            classification = response.text.strip().lower() if response.text else 'general'
            
            valid_types = ['theory_explanation', 'code_debugging', 'exercise', 'definition', 'general']
            if classification in valid_types:
                return classification
            else:
                return 'general'
                
        except Exception as e:
            logging.error(f"Error classifying question: {e}")
            return 'general'
    
    def generate_follow_up_questions(self, question: str, response: str) -> List[str]:
        """Generate follow-up questions to encourage deeper learning"""
        prompt = f"""
Dựa trên câu hỏi và câu trả lời sau, hãy tạo 3 câu hỏi tiếp theo để khuyến khích sinh viên học sâu hơn:

Câu hỏi gốc: {question}
Câu trả lời: {response}

Tạo 3 câu hỏi follow-up giúp sinh viên:
1. Hiểu sâu hơn về chủ đề
2. Áp dụng kiến thức vào thực tế
3. Kết nối với các khái niệm khác

Trả lời dưới dạng danh sách đánh số:
"""
        
        try:
            follow_up_response = self.model.generate_content(prompt)
            
            response_text = follow_up_response.text or ""
            
            # Parse the numbered list
            lines = response_text.strip().split('\n')
            follow_ups = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    clean_line = line.split('.', 1)[-1].strip()
                    clean_line = clean_line.lstrip('- ')
                    if clean_line:
                        follow_ups.append(clean_line)
            
            return follow_ups[:3]  # Return max 3 questions
            
        except Exception as e:
            logging.error(f"Error generating follow-up questions: {e}")
            return []
    
    def generate_practice_exercise(self, topic: str, difficulty: str = "medium") -> str:
        """Generate a practice exercise for the given topic"""
        prompt = f"""
Tạo một bài tập thực hành về chủ đề: {topic}
Độ khó: {difficulty}

Bài tập nên bao gồm:
1. Mô tả bài toán rõ ràng
2. Input/Output mẫu
3. Gợi ý cách tiếp cận (không đưa code hoàn chỉnh)
4. Test cases để kiểm tra

Định dạng:
## Bài tập: [Tên bài tập]

### Mô tả:
[Mô tả chi tiết]

### Input/Output:
[Ví dụ input và output mong đợi]

### Gợi ý:
[Các bước tiếp cận]

### Test cases:
[Các test cases để kiểm tra]
"""
        
        try:
            exercise_response = self.model.generate_content(prompt)
            
            return exercise_response.text or "Không thể tạo bài tập thực hành."
            
        except Exception as e:
            logging.error(f"Error generating practice exercise: {e}")
            return "Lỗi khi tạo bài tập thực hành."
    
    def _validate_context_quality(self, question: str, relevant_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate if context is sufficient for generating a quality educational response
        
        Returns:
            Dict with validation results and quality metrics
        """
        validation_result = {
            'is_sufficient': False,
            'issues': [],
            'metrics': {},
            'recommendations': []
        }
        
        # Metric 1: Document quantity check
        doc_count = len(relevant_docs)
        validation_result['metrics']['document_count'] = doc_count
        
        if doc_count == 0:
            validation_result['issues'].append('no_documents')
            validation_result['recommendations'].append('Tìm kiếm tài liệu về chủ đề được hỏi')
            return validation_result
        
        # Metric 2: Content length and quality
        total_content_length = 0
        valid_docs_count = 0
        min_similarity_scores = []
        
        for doc in relevant_docs:
            content = doc.get('content', '').strip()
            similarity_score = doc.get('similarity_score', 0)
            
            if content and len(content) > 50:  # Minimum meaningful content length
                total_content_length += len(content)
                valid_docs_count += 1
                min_similarity_scores.append(similarity_score)
        
        validation_result['metrics']['total_content_length'] = total_content_length
        validation_result['metrics']['valid_docs_count'] = valid_docs_count
        validation_result['metrics']['avg_similarity'] = sum(min_similarity_scores) / len(min_similarity_scores) if min_similarity_scores else 0
        
        # Metric 3: Content relevance (similarity scores)
        if min_similarity_scores:
            max_similarity = max(min_similarity_scores)
            avg_similarity = sum(min_similarity_scores) / len(min_similarity_scores)
            
            validation_result['metrics']['max_similarity'] = max_similarity
            validation_result['metrics']['avg_similarity'] = avg_similarity
        else:
            validation_result['metrics']['max_similarity'] = 0
            validation_result['metrics']['avg_similarity'] = 0
        
        # Quality thresholds
        MIN_VALID_DOCS = 1
        MIN_CONTENT_LENGTH = 200  # Minimum total content for meaningful response
        MIN_MAX_SIMILARITY = 0.1  # At least one document should be somewhat relevant
        MIN_AVG_SIMILARITY = 0.05  # Average relevance threshold
        
        # Validation checks
        issues = []
        
        if valid_docs_count < MIN_VALID_DOCS:
            issues.append('insufficient_valid_documents')
            validation_result['recommendations'].append(f'Cần ít nhất {MIN_VALID_DOCS} tài liệu có nội dung có nghĩa')
        
        if total_content_length < MIN_CONTENT_LENGTH:
            issues.append('insufficient_content_length')
            validation_result['recommendations'].append(f'Cần ít nhất {MIN_CONTENT_LENGTH} ký tự nội dung để trả lời đầy đủ')
        
        if validation_result['metrics']['max_similarity'] < MIN_MAX_SIMILARITY:
            issues.append('low_relevance')
            validation_result['recommendations'].append('Các tài liệu tìm được có độ liên quan thấp với câu hỏi')
        
        if validation_result['metrics']['avg_similarity'] < MIN_AVG_SIMILARITY:
            issues.append('overall_low_relevance')
            validation_result['recommendations'].append('Độ liên quan trung bình của tài liệu thấp')
        
        # Metric 4: Question-specific validation
        question_keywords = self._extract_question_keywords(question)
        content_coverage = self._check_keyword_coverage(question_keywords, relevant_docs)
        validation_result['metrics']['keyword_coverage'] = content_coverage
        
        if content_coverage < 0.3:  # At least 30% of question keywords should be covered
            issues.append('poor_keyword_coverage')
            validation_result['recommendations'].append('Tài liệu không bao phủ đủ các khái niệm chính trong câu hỏi')
        
        validation_result['issues'] = issues
        validation_result['is_sufficient'] = len(issues) == 0
        
        return validation_result
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract key terms from question for coverage analysis"""
        import re
        
        # Simple keyword extraction (can be enhanced with NLP)
        # Remove common stop words and extract meaningful terms
        stop_words = {'là', 'gì', 'như', 'thế', 'nào', 'tại', 'sao', 'có', 'thể', 'được', 'với', 'của', 'trong', 'và', 'hoặc'}
        
        # Extract words (Vietnamese + English)
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _check_keyword_coverage(self, keywords: List[str], relevant_docs: List[Dict[str, Any]]) -> float:
        """Check how many question keywords are covered in the documents"""
        if not keywords:
            return 1.0
        
        # Combine all document content
        all_content = ""
        for doc in relevant_docs:
            content = doc.get('content', '') + " " + doc.get('title', '') + " " + doc.get('section', '')
            all_content += content.lower() + " "
        
        # Count keyword coverage
        covered_keywords = 0
        for keyword in keywords:
            if keyword.lower() in all_content:
                covered_keywords += 1
        
        return covered_keywords / len(keywords) if keywords else 0.0
    
    def _generate_insufficient_context_response(self, question: str, 
                                              validation_result: Dict[str, Any], 
                                              language: str) -> str:
        """Generate appropriate response when context is insufficient"""
        
        issues = validation_result.get('issues', [])
        recommendations = validation_result.get('recommendations', [])
        metrics = validation_result.get('metrics', {})
        
        if language == "English":
            base_msg = "I don't have sufficient context to provide a comprehensive answer to your question."
        else:
            base_msg = "Không đủ ngữ cảnh để trả lời chính xác câu hỏi của bạn."
        
        # Add specific diagnostic information
        details = []
        
        if 'no_documents' in issues:
            if language == "English":
                details.append("No relevant documents were found.")
            else:
                details.append("Không tìm thấy tài liệu liên quan nào.")
                
        elif 'insufficient_content_length' in issues:
            content_length = metrics.get('total_content_length', 0)
            if language == "English":
                details.append(f"Available content is too brief ({content_length} characters).")
            else:
                details.append(f"Nội dung có sẵn quá ngắn ({content_length} ký tự).")
                
        elif 'low_relevance' in issues:
            max_sim = metrics.get('max_similarity', 0)
            if language == "English":
                details.append(f"Document relevance is too low (max similarity: {max_sim:.2f}).")
            else:
                details.append(f"Độ liên quan tài liệu quá thấp (điểm tương đồng cao nhất: {max_sim:.2f}).")
        
        elif 'poor_keyword_coverage' in issues:
            coverage = metrics.get('keyword_coverage', 0)
            if language == "English":
                details.append(f"Poor coverage of question topics ({coverage*100:.1f}% of key terms found).")
            else:
                details.append(f"Bao phủ kém các chủ đề trong câu hỏi ({coverage*100:.1f}% thuật ngữ chính được tìm thấy).")
        
        # Generate suggestions
        if language == "English":
            suggestion_intro = "\nTo get a better answer, you could:"
            doc_suggestion = "• Provide more specific documents about"
        else:
            suggestion_intro = "\nĐể có câu trả lời tốt hơn, bạn có thể:"
            doc_suggestion = "• Cung cấp thêm tài liệu cụ thể về"
        
        # Extract question topic for suggestion
        question_topic = self._extract_main_topic(question)
        if question_topic:
            if language == "English":
                topic_suggestion = f"{doc_suggestion} {question_topic}"
            else:
                topic_suggestion = f"{doc_suggestion} {question_topic}"
        else:
            if language == "English":
                topic_suggestion = f"{doc_suggestion} the topic you're asking about"
            else:
                topic_suggestion = f"{doc_suggestion} chủ đề bạn đang hỏi"
        
        # Combine response
        full_response = base_msg
        if details:
            full_response += "\n\n" + "\n".join(details)
        
        full_response += suggestion_intro + "\n" + topic_suggestion
        
        # Add search suggestions based on validation
        if recommendations:
            if language == "English":
                full_response += "\n• Try searching with different keywords"
            else:
                full_response += "\n• Thử tìm kiếm với các từ khóa khác nhau"
        
        return full_response
    
    def _extract_main_topic(self, question: str) -> str:
        """Extract main topic from question for suggestion"""
        # Simple topic extraction - can be enhanced with NLP
        import re
        
        # Look for potential technical terms (capitalized words, common CS terms)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        if tech_terms:
            return tech_terms[0]
        
        # Look for quoted terms
        quoted = re.findall(r'"([^"]+)"', question)
        if quoted:
            return quoted[0]
        
        # Fallback: return first meaningful words
        words = question.split()
        meaningful_words = [w for w in words[:5] if len(w) > 3]
        return ' '.join(meaningful_words[:3]) if meaningful_words else "chủ đề này"
