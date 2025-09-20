# 🎓 AI Virtual Mentor - Comprehensive CS Education Assistant

## 🌟 Overview

**AI Virtual Mentor** là hệ thống gia sư thông minh toàn diện dành cho sinh viên Công nghệ Thông tin, được thiết kế với **dual-interface architecture** bao gồm cả **Streamlit** và **Flask** web applications. Hệ thống sử dụng **Retrieval-Augmented Generation (RAG)** với **Gemini AI** để cung cấp hỗ trợ học tập chính xác, có trích dẫn, và tuân thủ nguyên tắc học tập trung thực.

### 🏗️ Architecture Overview

```
AI Virtual Mentor/
├── 📱 apps/
│   ├── flask/          # Modern Web Interface (HTML/CSS/JS + Flask)
│   └── streamlit/      # Data-Driven Interface (Streamlit)
├── 🧠 mentor_core/     # Shared RAG Engine
│   ├── document_processor.py    # Multi-format document processing
│   ├── vector_search.py        # Hybrid search with Qdrant + TF-IDF
│   ├── gemini_client.py        # LLM integration with integrity controls
│   ├── citation_formatter.py   # Academic citation system
│   └── embeddings.py           # Multilingual embeddings
├── 📊 evaluation/      # Quality assurance framework
└── 🛠️ utils/          # Supporting utilities
```

## 🌐 Dual Interface Design

### 🖥️ Flask Web Application (`apps/flask/`)
**Modern, professional web interface** với responsive design:

**🎯 Key Features:**
- **Interactive Web UI**: HTML5/CSS3/ES6+ với modern UX patterns
- **Citation Tooltips**: Hover [●] markers để xem full source details  
- **Real-time Chat**: Instant messaging với typing indicators
- **Drag & Drop Upload**: Intuitive file management
- **Session Persistence**: Documents persist across page reloads
- **Mobile Responsive**: Optimized cho all screen sizes

**👥 Best For:** 
- Production deployment
- Public-facing educational platforms
- Students preferring traditional web interfaces
- Integration with existing web systems

---

### 📊 Streamlit Application (`apps/streamlit/`)
**Data-driven interface** với rich analytics và rapid prototyping:

**🎯 Key Features:**
- **Interactive Widgets**: Native Streamlit components
- **Real-time Analytics**: Session metrics và learning progress
- **Profile Management**: Comprehensive student profiles
- **A/B Testing UI**: Configuration testing interface
- **Debug Console**: Development và evaluation tools

**👥 Best For:**
- Research và development
- Educational analytics
- Rapid feature prototyping  
- Data science workflows

## 🧠 Shared RAG Engine (`mentor_core/`)

### 🔍 Advanced Document Processing
- **Multi-format Support**: PDF, Markdown, Code files (Python, Java, C++, JavaScript)
- **Heading-aware Chunking**: Preserves semantic structure (400-800 tokens)
- **Code Intelligence**: Function/class-level chunking với syntax awareness
- **Metadata Extraction**: Page numbers, sections, file types

### 🔎 Hybrid Search System
- **Vector Search**: Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **TF-IDF Fallback**: Robust search khi embeddings unavailable
- **Qdrant Integration**: Persistent vector storage
- **MMR Diversity**: Balanced relevance và diversity
- **BGE Reranking**: Enhanced result quality

### 🤖 Gemini AI Integration
- **Academic Integrity**: 3-level system (Normal/Academic/Exam)
- **Question Classification**: Theory, Debugging, Exercise, Definition
- **Multilingual**: Vietnamese/English với auto-detection
- **Context-aware**: Responses grounded in uploaded documents

### 📝 Citation System
- **Academic Format**: `[Nguồn X: filename, Trang Y, Mục "Section"]`
- **Source Attribution**: Automatic extraction từ retrieved documents
- **Interactive Citations**: Tooltip displays cho detailed references
- **Word Boundary Detection**: Proper spacing và formatting

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- 4GB+ RAM recommended

### Installation
```bash
# Clone repository
git clone [your-repo-url]
cd AI-Virtual-Mentor

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_gemini_api_key_here"
export SESSION_SECRET="your_secure_session_secret"
```

### Run Applications

**🌐 Flask Web Interface:**
```bash
cd apps/flask
python web_app.py
# Access: http://localhost:5000
```

**📊 Streamlit Interface:**
```bash
cd apps/streamlit  
streamlit run app.py --server.port 5000
# Access: http://localhost:5000
```

## 💡 Usage Examples

### 📖 Theory Questions
```
"TCP và UDP khác nhau như thế nào?"
"Giải thích thuật toán QuickSort và độ phức tạp"
```

### 🐛 Code Debugging  
```
"Tại sao code Java này bị NullPointerException?"
"Làm sao sửa lỗi connection timeout?"
```

### 📝 Academic Exercises (Integrity Mode)
```
"Hướng dẫn tôi approach bài RMI programming"
"Gợi ý steps để solve multithreading assignment"
```

### 📚 Concept Definitions
```
"RMI là gì và hoạt động như thế nào?"
"Serialization trong network programming"
```

## 📊 Quality Assurance

### 🎯 Evaluation Framework
- **Golden Q&A Dataset**: 25+ curated questions across CS topics
- **RAGAS Metrics**: Faithfulness, Relevance, Context Quality
- **A/B Testing**: Statistical significance testing
- **Academic Compliance**: Integrity mode effectiveness measurement

### 📈 Performance Benchmarks
| Metric | Target | Current Status |
|--------|---------|---------------|
| **Answer Relevance** | ≥ 0.80 | 🔄 Monitoring |
| **Citation Accuracy** | ≥ 0.85 | ✅ Achieved |
| **Integrity Compliance** | ≥ 0.95 | ✅ Achieved |
| **Response Time** | < 8s | ✅ Achieved |

### 🧪 Continuous Testing
```bash
# Run Golden Q&A evaluation
python evaluation/ragas_evaluator.py

# Execute A/B tests
python evaluation/ab_testing.py
```

## 🔒 Academic Integrity

### 🛡️ Integrity Modes
- **Normal**: Complete solutions với detailed explanations
- **Academic**: Guidance và hints, no direct answers
- **Exam**: Minimal assistance, encourage independent thinking

### 📋 Compliance Features
- **Source Attribution**: Mandatory citations cho all information
- **No Direct Solutions**: Academic modes avoid complete answers
- **Learning Focus**: Promotes step-by-step understanding
- **Context Validation**: Responses grounded in provided materials

## 🛠️ Technical Details

### 🔧 Core Dependencies
```
Flask==2.3.2              # Web framework
Streamlit==1.25.0         # Data app framework
google-generativeai==0.3.0 # Gemini AI integration
qdrant-client==1.4.0      # Vector database
sentence-transformers==2.2.2 # Multilingual embeddings
scikit-learn==1.3.0       # TF-IDF fallback
pdfplumber==0.9.0         # PDF processing
```

### ⚙️ Configuration Files
```
.streamlit/config.toml     # Streamlit server settings
mentor_core/__init__.py    # Shared components
pyproject.toml            # Package management
replit.md                 # Project documentation
```

### 🗂️ Data Management
- **Document Storage**: `data/documents/` - Uploaded files
- **Embeddings**: `data/embeddings/` - Vector indices
- **Sessions**: `apps/flask/runtime/session/` - User sessions
- **Evaluation**: `evaluation/` - Testing datasets và results

## 🔧 Development Setup

### 🏗️ Architecture Decisions
- **Shared Core**: Consistent RAG functionality across interfaces
- **Modular Design**: Independent deployment của Flask/Streamlit apps
- **Fallback Systems**: Robust operation với degraded capabilities
- **Session Management**: Persistent state across interactions

### 🧪 Testing Strategy
- **Unit Tests**: Core module functionality
- **Integration Tests**: End-to-end workflows
- **UI Tests**: Interface interactions với Playwright
- **Evaluation Tests**: Academic quality metrics

### 📝 Code Standards
- **Type Hints**: Full typing cho all functions
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Graceful degradation
- **Security**: Input validation và safe file handling

## 🔮 Roadmap

### Version 2.0 (Planning)
- [ ] **Multi-modal RAG**: Image và diagram processing
- [ ] **Advanced Analytics**: Learning pattern analysis
- [ ] **Collaborative Features**: Document sharing
- [ ] **Mobile Apps**: React Native interfaces

### Technology Upgrades  
- [ ] **Latest Embeddings**: Upgrade to newest multilingual models
- [ ] **Local LLM Support**: Self-hosted alternatives
- [ ] **Graph RAG**: Knowledge graph enhancements
- [ ] **Microservices**: Scalable architecture

## 🤝 Contributing

### Development Process
1. **Fork Repository**: Create your feature branch
2. **Follow Standards**: Code style, testing, documentation
3. **Add Tests**: Unit và integration coverage
4. **Update Docs**: README và inline documentation
5. **Submit PR**: Detailed description với test results

### 🐛 Bug Reports
- Use GitHub Issues với detailed reproduction steps
- Include system info, error logs, và expected behavior
- Add labels cho priority và component affected

## 📄 License & Academic Use

**Educational License**: Designed cho academic environments với respect for institutional policies. Commercial use requires permission.

### 🎓 Academic Guidelines
- **Cite Sources**: Always attribute AI assistance in assignments
- **Follow Policies**: Respect your institution's academic integrity rules
- **Use Responsibly**: Tool for learning enhancement, not replacement
- **Report Issues**: Help improve educational AI safety

## 📞 Support & Community

- **Documentation**: Comprehensive guides trong repo wiki
- **Issues**: GitHub Issues cho bug reports và feature requests  
- **Discussions**: Community forum cho usage questions
- **Updates**: Follow releases cho latest improvements

---

## 🙏 Acknowledgments

- **Google AI**: Gemini API cho intelligent responses
- **Streamlit Team**: Excellent data app framework
- **Flask Community**: Robust web development tools
- **Open Source**: Sentence Transformers, Qdrant, và supporting libraries
- **Educational Community**: Feedback và testing từ students và educators

---

**🎯 Built with ❤️ for Vietnamese Computer Science Education**

**📅 Last Updated**: September 2025  
**🔖 Version**: 2.0.0  
**✅ Status**: Production Ready với Dual Interface Support