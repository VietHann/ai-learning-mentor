import re
from typing import Dict, List

class QuestionClassifier:
    """Classify questions into different educational categories"""
    
    def __init__(self):
        # Keywords for different question types
        self.classification_patterns = {
            'theory_explanation': {
                'keywords': [
                    'giải thích', 'explain', 'what is', 'là gì', 'khái niệm', 'concept',
                    'tại sao', 'why', 'như thế nào', 'how', 'nguyên lý', 'principle',
                    'cách thức', 'hoạt động', 'works', 'mechanism', 'lý thuyết', 'theory'
                ],
                'patterns': [
                    r'.*là gì.*',
                    r'.*giải thích.*',
                    r'.*what is.*',
                    r'.*how.*work.*',
                    r'.*nguyên lý.*',
                    r'.*khái niệm.*'
                ]
            },
            'code_debugging': {
                'keywords': [
                    'lỗi', 'error', 'bug', 'debug', 'sửa', 'fix', 'không chạy', 'not working',
                    'compile error', 'runtime error', 'exception', 'crash', 'fail',
                    'syntax error', 'logic error', 'tìm lỗi', 'kiểm tra code'
                ],
                'patterns': [
                    r'.*lỗi.*',
                    r'.*error.*',
                    r'.*bug.*',
                    r'.*debug.*',
                    r'.*không chạy.*',
                    r'.*not work.*',
                    r'.*fix.*code.*',
                    r'.*sửa.*code.*'
                ]
            },
            'exercise': {
                'keywords': [
                    'bài tập', 'exercise', 'homework', 'assignment', 'problem', 'bài toán',
                    'làm thế nào', 'how to', 'viết code', 'write code', 'implement',
                    'giải quyết', 'solve', 'tạo', 'create', 'xây dựng', 'build',
                    'thuật toán', 'algorithm', 'code', 'program', 'chương trình'
                ],
                'patterns': [
                    r'.*bài tập.*',
                    r'.*exercise.*',
                    r'.*homework.*',
                    r'.*làm thế nào.*',
                    r'.*how to.*',
                    r'.*viết.*code.*',
                    r'.*write.*code.*',
                    r'.*implement.*',
                    r'.*tạo.*chương trình.*'
                ]
            },
            'definition': {
                'keywords': [
                    'định nghĩa', 'definition', 'nghĩa', 'meaning', 'thuật ngữ', 'term',
                    'từ khóa', 'keyword', 'glossary', 'dictionary', 'có nghĩa là',
                    'means', 'refers to', 'stand for'
                ],
                'patterns': [
                    r'.*định nghĩa.*',
                    r'.*definition.*',
                    r'.*nghĩa.*là.*',
                    r'.*means.*',
                    r'.*thuật ngữ.*',
                    r'.*có nghĩa.*'
                ]
            },
            'code_review': {
                'keywords': [
                    'review', 'đánh giá', 'kiểm tra', 'check', 'code review',
                    'tối ưu', 'optimize', 'cải thiện', 'improve', 'refactor',
                    'best practice', 'thực hành tốt', 'clean code'
                ],
                'patterns': [
                    r'.*review.*code.*',
                    r'.*đánh giá.*code.*',
                    r'.*kiểm tra.*code.*',
                    r'.*tối ưu.*',
                    r'.*optimize.*',
                    r'.*cải thiện.*'
                ]
            }
        }
        
        # Vietnamese technical terms for better classification
        self.technical_terms = {
            'programming': [
                'lập trình', 'programming', 'code', 'coding', 'algorithm', 'thuật toán',
                'biến', 'variable', 'hàm', 'function', 'class', 'lớp', 'object', 'đối tượng'
            ],
            'networking': [
                'mạng', 'network', 'protocol', 'giao thức', 'TCP', 'UDP', 'HTTP', 'IP',
                'socket', 'client', 'server', 'router', 'switch'
            ],
            'database': [
                'cơ sở dữ liệu', 'database', 'SQL', 'query', 'truy vấn', 'table', 'bảng',
                'index', 'primary key', 'foreign key', 'relationship', 'mối quan hệ'
            ],
            'mobile': [
                'mobile', 'di động', 'Android', 'iOS', 'app', 'ứng dụng',
                'activity', 'fragment', 'view', 'layout', 'intent'
            ]
        }
    
    def classify(self, question: str) -> str:
        """Classify a question into one of the predefined categories"""
        question_lower = question.lower()
        
        # Score each category
        category_scores = {}
        
        for category, patterns_data in self.classification_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns_data['keywords']:
                if keyword.lower() in question_lower:
                    score += 1
            
            # Check regex patterns
            for pattern in patterns_data['patterns']:
                if re.search(pattern, question_lower):
                    score += 2  # Patterns have higher weight
            
            category_scores[category] = score
        
        # Find the category with highest score
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            if category_scores[best_category] > 0:
                return best_category
        
        # Additional heuristics for edge cases
        return self._additional_classification(question_lower)
    
    def _additional_classification(self, question_lower: str) -> str:
        """Additional classification logic for edge cases"""
        
        # Check for code-related questions
        code_indicators = ['def ', 'class ', 'import ', 'for ', 'if ', 'while ', 
                          'function', 'method', 'array', 'list', 'dict', 'string']
        
        if any(indicator in question_lower for indicator in code_indicators):
            if any(word in question_lower for word in ['lỗi', 'error', 'bug', 'sửa', 'fix']):
                return 'code_debugging'
            else:
                return 'exercise'
        
        # Check for theoretical questions
        theory_indicators = ['nguyên lý', 'cách thức', 'hoạt động', 'tại sao', 'như thế nào']
        if any(indicator in question_lower for indicator in theory_indicators):
            return 'theory_explanation'
        
        # Check for definition questions
        definition_indicators = ['là gì', 'what is', 'định nghĩa', 'nghĩa']
        if any(indicator in question_lower for indicator in definition_indicators):
            return 'definition'
        
        # Default to general if no clear category
        return 'general'
    
    def get_question_complexity(self, question: str) -> str:
        """Estimate the complexity level of a question"""
        question_lower = question.lower()
        
        # Basic indicators
        basic_indicators = ['là gì', 'what is', 'định nghĩa', 'khái niệm cơ bản']
        
        # Intermediate indicators  
        intermediate_indicators = ['implement', 'giải thích', 'so sánh', 'phân biệt', 
                                 'ưu nhược điểm', 'advantages', 'disadvantages']
        
        # Advanced indicators
        advanced_indicators = ['optimize', 'tối ưu', 'performance', 'hiệu năng',
                             'architecture', 'thiết kế', 'pattern', 'best practice',
                             'scalability', 'mở rộng']
        
        if any(indicator in question_lower for indicator in advanced_indicators):
            return 'advanced'
        elif any(indicator in question_lower for indicator in intermediate_indicators):
            return 'intermediate'
        elif any(indicator in question_lower for indicator in basic_indicators):
            return 'basic'
        else:
            # Estimate based on question length and technical terms
            if len(question.split()) > 15:
                return 'intermediate'
            else:
                return 'basic'
    
    def extract_topic_keywords(self, question: str) -> List[str]:
        """Extract technical keywords and topics from the question"""
        question_lower = question.lower()
        found_keywords = []
        
        # Check technical terms by domain
        for domain, terms in self.technical_terms.items():
            for term in terms:
                if term.lower() in question_lower:
                    found_keywords.append(term)
        
        # Extract programming language mentions
        programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'php', 
                                'ruby', 'swift', 'kotlin', 'go', 'rust']
        
        for lang in programming_languages:
            if lang in question_lower:
                found_keywords.append(lang)
        
        # Extract common CS concepts
        cs_concepts = ['algorithm', 'thuật toán', 'data structure', 'cấu trúc dữ liệu',
                      'complexity', 'độ phức tạp', 'recursion', 'đệ quy',
                      'inheritance', 'kế thừa', 'polymorphism', 'đa hình',
                      'encapsulation', 'đóng gói', 'abstraction', 'trừu tượng']
        
        for concept in cs_concepts:
            if concept in question_lower:
                found_keywords.append(concept)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def is_homework_question(self, question: str) -> bool:
        """Detect if question might be a homework/assignment question"""
        homework_indicators = [
            'bài tập', 'homework', 'assignment', 'đề bài', 'problem set',
            'exercise', 'lab', 'thí nghiệm', 'project', 'dự án',
            'submit', 'nộp bài', 'deadline', 'hạn chót',
            'grade', 'điểm', 'test', 'kiểm tra', 'exam', 'thi'
        ]
        
        question_lower = question.lower()
        
        # Check for direct homework indicators
        homework_score = sum(1 for indicator in homework_indicators 
                           if indicator in question_lower)
        
        # Check for suspicious patterns (asking for complete solutions)
        suspicious_patterns = [
            r'.*viết.*chương trình.*hoàn chỉnh.*',
            r'.*write.*complete.*program.*',
            r'.*code.*toàn bộ.*',
            r'.*full.*solution.*',
            r'.*làm.*hộ.*',
            r'.*help.*me.*solve.*'
        ]
        
        pattern_matches = sum(1 for pattern in suspicious_patterns 
                            if re.search(pattern, question_lower))
        
        # Return True if strong indicators of homework
        return homework_score >= 2 or pattern_matches >= 1
    
    def suggest_learning_approach(self, question_type: str, complexity: str) -> str:
        """Suggest appropriate learning approach based on question type and complexity"""
        
        approaches = {
            'theory_explanation': {
                'basic': 'Bắt đầu với định nghĩa cơ bản, sau đó tìm hiểu ví dụ thực tế',
                'intermediate': 'Nghiên cứu nguyên lý hoạt động và so sánh với các khái niệm tương tự',
                'advanced': 'Phân tích ưu nhược điểm, các trường hợp sử dụng và best practices'
            },
            'code_debugging': {
                'basic': 'Kiểm tra syntax errors trước, sau đó trace qua từng dòng code',
                'intermediate': 'Sử dụng debugger và print statements để tìm logic errors',
                'advanced': 'Phân tích performance, memory usage và code quality'
            },
            'exercise': {
                'basic': 'Chia nhỏ bài toán, viết pseudo-code trước khi implement',
                'intermediate': 'Thiết kế algorithm, xác định data structures phù hợp',
                'advanced': 'Tối ưu hóa solution, xem xét edge cases và scalability'
            },
            'definition': {
                'basic': 'Tìm hiểu định nghĩa chính thức và ví dụ đơn giản',
                'intermediate': 'So sánh với các khái niệm liên quan và ứng dụng thực tế',
                'advanced': 'Nghiên cứu evolution của khái niệm và current trends'
            }
        }
        
        return approaches.get(question_type, {}).get(complexity, 
            'Tiếp cận từng bước một, từ cơ bản đến nâng cao')
