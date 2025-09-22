# 🎓 AI Virtual Mentor - Hệ thống hỗ trợ học tập cho sinh viên

## Tổng quan

**AI Virtual Mentor** là hệ thống gia sư thông minh toàn diện dành cho sinh viên. Hệ thống sử dụng **Retrieval-Augmented Generation (RAG)** với **Gemini AI** để cung cấp hỗ trợ học tập chính xác, có trích dẫn và tuân thủ nguyên tắc học tập trung thực.

### Tổng quan Kiến trúc

```
AI Virtual Mentor/
├── apps/
│   ├── flask/          
│   └── streamlit/    
├── mentor_core/    
│   ├── document_processor.py   
│   ├── vector_search.py        
│   ├── gemini_client.py       
│   ├── citation_formatter.py  
│   └── embeddings.py         
├── evaluation/    
└── utils/      
```

## Thiết kế Giao diện Kép

### Ứng dụng Web Flask (`apps/flask/`)
**Giao diện web hiện đại, chuyên nghiệp** với thiết kế responsive:

**Tính năng chính:**
- **Giao diện Web tương tác**: HTML5/CSS3/ES6+ với các mẫu UX hiện đại
- **Tooltip trích dẫn**: Di chuột qua dấu [●] để xem chi tiết nguồn đầy đủ
- **Trò chuyện thời gian thực**: Nhắn tin tức thì với chỉ báo đang gõ
- **Tải lên kéo & thả**: Quản lý tệp trực quan
- **Duy trì phiên**: Tài liệu tồn tại qua các lần tải lại trang
- **Responsive di động**: Tối ưu hóa cho mọi kích thước màn hình

**Phù hợp nhất cho:**
- Triển khai sản xuất
- Nền tảng giáo dục công cộng
- Sinh viên ưa thích giao diện web truyền thống
- Tích hợp với hệ thống web hiện có

---

###  Ứng dụng Streamlit (`apps/streamlit/`)
Đang upadte

### Xử lý tài liệu nâng cao
- **Hỗ trợ đa định dạng**: PDF, Markdown, tệp Code (Python, Java, C++, JavaScript)
- **Phân đoạn nhận biết tiêu đề**: Bảo tồn cấu trúc ngữ nghĩa (400-800 tokens)
- **Trí tuệ Code**: Phân đoạn cấp function/class với nhận biết cú pháp
- **Trích xuất Metadata**: Số trang, phần, loại tệp

### Hệ thống Tìm kiếm Kết hợp
- **Tìm kiếm Vector**: Embeddings đa ngôn ngữ (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Dự phòng TF-IDF**: Tìm kiếm mạnh mẽ khi embeddings không khả dụng
- **Tích hợp Qdrant**: Lưu trữ vector bền vững
- **Đa dạng MMR**: Cân bằng độ liên quan và đa dạng
- **Xếp hạng lại BGE**: Chất lượng kết quả được cải thiện

### Tích hợp Gemini AI
- **Tính toàn vẹn học thuật**: Hệ thống 3 cấp (Bình thường/Học thuật/Thi)
- **Phân loại câu hỏi**: Lý thuyết, Debug, Bài tập, Định nghĩa
- **Đa ngôn ngữ**: Tiếng Việt/Tiếng Anh với tự động phát hiện
- **Nhận biết ngữ cảnh**: Phản hồi dựa trên tài liệu được tải lên

### Hệ thống Trích dẫn
- **Định dạng học thuật**: `[Nguồn X: filename, Trang Y, Mục "Section"]`
- **Ghi nhận nguồn**: Trích xuất tự động từ tài liệu được truy xuất
- **Trích dẫn tương tác**: Tooltip hiển thị cho tham chiếu chi tiết
- **Phát hiện ranh giới từ**: Khoảng cách và định dạng phù hợp

## Bắt đầu nhanh

### Yêu cầu tiên quyết
- Python 3.8+
- Khóa API Google Gemini
- Khuyến nghị 4GB+ RAM

### Cài đặt
```bash
# Clone repository
git clone https://github.com/VietHann/ai-learning-mentor.git
cd ai-learning-mentor

# Cài đặt dependencies
pip install -r "apps/flask/requirements.txt"

# Đặt biến môi trường
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Chạy Ứng dụng

**Giao diện Web Flask:**
```bash
python -m apps.flask.web_app
# Truy cập: http://localhost:5000
```

## Ví dụ Sử dụng

### Câu hỏi Lý thuyết
```
"TCP và UDP khác nhau như thế nào?"
"Giải thích thuật toán QuickSort và độ phức tạp"
```

### Debug Code
```
"Tại sao code Java này bị NullPointerException?"
"Làm sao sửa lỗi connection timeout?"
```

### Bài tập Học thuật (Chế độ Tính toàn vẹn)
```
"Hướng dẫn tôi approach bài RMI programming"
"Gợi ý steps để solve multithreading assignment"
```

### Định nghĩa Khái niệm
```
"RMI là gì và hoạt động như thế nào?"
"Serialization trong network programming"
```

## Đảm bảo Chất lượng

### Khung Đánh giá
- **Bộ dữ liệu Q&A Vàng**: 25+ câu hỏi được tuyển chọn qua các chủ đề CNTT
- **Số liệu RAGAS**: Tính trung thực, Độ liên quan, Chất lượng ngữ cảnh
- **A/B Testing**: Kiểm tra ý nghĩa thống kê
- **Tuân thủ học thuật**: Đo lường hiệu quả chế độ tính toàn vẹn

### Điểm chuẩn Hiệu suất
| Số liệu | Mục tiêu | Trạng thái hiện tại |
|---------|----------|-------------------|
| **Độ liên quan câu trả lời** | ≥ 0.80 | ✅ Đã đạt |
| **Độ chính xác trích dẫn** | ≥ 0.85 | ✅ Đã đạt |
| **Tuân thủ tính toàn vẹn** | ≥ 0.95 | ✅ Đã đạt |
| **Thời gian phản hồi** | < 8s | ✅ Đã đạt |
| **Cải tiến thời gian phản hồi** | < 4s | ✅ Đã đạt |

### Kiểm tra Liên tục
```bash
# Chạy đánh giá Q&A Vàng
python evaluation/ragas_evaluator.py

# Thực hiện kiểm tra A/B
python evaluation/ab_testing.py
```

## Tính toàn vẹn Học thuật

### Chế độ Tính toàn vẹn
- **Bình thường**: Giải pháp hoàn chỉnh với giải thích chi tiết
- **Học thuật**: Hướng dẫn và gợi ý, không có câu trả lời trực tiếp
- **Thi**: Hỗ trợ tối thiểu, khuyến khích tư duy độc lập

### Tính năng Tuân thủ
- **Ghi nhận nguồn**: Trích dẫn bắt buộc cho mọi thông tin
- **Không giải pháp trực tiếp**: Chế độ học thuật tránh câu trả lời hoàn chỉnh
- **Tập trung học tập**: Khuyến khích hiểu biết từng bước
- **Xác thực ngữ cảnh**: Phản hồi dựa trên tài liệu được cung cấp
  
### Quản lý Dữ liệu
- **Lưu trữ tài liệu**: `data/documents/` - Tệp được tải lên
- **Embeddings**: `data/embeddings/` - Chỉ mục vector
- **Phiên**: `apps/flask/runtime/session/` - Phiên người dùng
- **Đánh giá**: `evaluation/` - Bộ dữ liệu kiểm tra và kết quả

## Thiết lập Phát triển

### Quyết định Kiến trúc
- **Nhân được chia sẻ**: Chức năng RAG nhất quán qua các giao diện
- **Thiết kế Modular**: Triển khai độc lập của ứng dụng Flask/Streamlit
- **Hệ thống Dự phòng**: Hoạt động mạnh mẽ với khả năng bị suy giảm
- **Quản lý Phiên**: Trạng thái bền vững qua các tương tác

### Chiến lược Kiểm tra
- **Unit Tests**: Chức năng module cốt lõi
- **Integration Tests**: Quy trình end-to-end
- **UI Tests**: Tương tác giao diện với Playwright
- **Evaluation Tests**: Số liệu chất lượng học thuật
