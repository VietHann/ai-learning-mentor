# AI Virtual Mentor - Flask Web Interface

## API Endpoints

### Core Application Routes
```http
GET  /                    # Main application interface
GET  /api/status         # System health và component status
GET  /api/documents      # List of processed documents (NEW!)
```

### User Management  
```http
GET|POST /api/profile    # User profile management
GET|POST /api/settings   # Application settings
```

### Document Processing
```http
POST /api/upload         # Multi-file upload với validation
POST /api/clear          # Clear session data
```

### AI Interaction
```http
POST /api/ask           # Question submission với academic integrity
GET  /api/conversation  # Conversation history với citations
```

### Response Format
All API endpoints return consistent JSON:
```json
{
    "success": true,
    "data": { /* response data */ },
    "error": null,
    "processed_count": 0,    // for uploads
    "documents": []          // document list
}
```

## Frontend Architecture

### Modern JavaScript (ES6+)
**Class-based Architecture** (`AIVirtualMentor` class):
```javascript
class AIVirtualMentor {
    constructor() {
        this.isLoading = false;
        this.documents = [];
        this.init();
    }

    // Core initialization
    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.loadDocumentsList();  // NEW: Auto-load documents
    }
}
```


## Security & Data Management

### Implemented Security
- **CSRF Protection**: Flask sessions với signed cookies
- **File Validation**: Type, size, và content validation
- **Path Traversal Prevention**: Secure file handling  
- **Session Security**: HTTPOnly cookies với SameSite protection
- **Input Sanitization**: XSS prevention trong user inputs

### Session Management
```python
# Flask session configuration
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR = './runtime/session'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_PERMANENT = True
```

### Data Persistence
- **Document Storage**: Local file storage với metadata
- **Session State**: Persistent across browser sessions
- **Vector Indices**: Cached embeddings for performance
- **Upload Management**: Secure temporary file handling

## User Experience Features

### Interactive Chat Interface
- **Real-time Messaging**: Instant question/answer display
- **Citation Integration**: Inline [●] markers với detailed tooltips
- **Message History**: Persistent conversation với timestamps
- **Input Enhancement**: Auto-resizing textarea với smart formatting

### Document Management
- **Visual Upload Area**: Drag-and-drop với visual feedback
- **Processing Progress**: Real-time upload và processing status
- **Document List**: Visual display của uploaded files với icons
- **Persistence Fix**: Documents remain visible after page reload ✅

### Configuration Panel
- **Profile Management**: Inline editing với immediate save
- **Integrity Modes**: Visual mode selection với descriptions
- **Settings Panel**: Collapsible configuration với instant apply
- **Analytics Display**: Session statistics với visual indicators

### API Testing
```bash
# Test system status
curl http://localhost:5000/api/status

# Test document list (NEW endpoint)
curl http://localhost:5000/api/documents

# Test profile management
curl -X POST http://localhost:5000/api/profile \
  -H "Content-Type: application/json" \
  -d '{"profile": {"name": "Test Student", "major": "CS"}}'

# Test document upload
curl -X POST http://localhost:5000/api/upload \
  -F "files=@test_document.pdf"
```

### Debug Mode
```bash
# Enable Flask debug mode
export FLASK_DEBUG=1
export FLASK_ENV=development
python web_app.py
```

## Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY="your_google_ai_studio_api_key"

# Optional (có defaults)  
SESSION_SECRET="your_secure_random_secret_key"
FLASK_DEBUG="1"  # for development
FLASK_ENV="development"  # for development
```

### Production Configuration
```python
# For production deployment
SESSION_COOKIE_SECURE = True      # HTTPS only
CORS_ORIGINS = ["your-domain.com"] # Restrict CORS
UPLOAD_MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB limit
```

## Troubleshooting

### Common Issues

** Document Persistence Issue (FIXED)**  
```
Problem: Documents disappeared after page reload
Solution: Added /api/documents endpoint và loadDocumentsList() function
Status: Resolved - documents now persist across page reloads
```

**Import Errors**
```bash
# Solution: Run from project root để resolve mentor_core imports
cd /AI-Virtual-Mentor
cd apps/flask && python web_app.py
```

**Port 5000 In Use**
```bash
# Check what's using port 5000
lsof -i :5000

# Use different port
flask --app apps/flask/web_app.py run --port 8000
```

**Session Issues**  
```bash
# Clear session data
rm -rf apps/flask/runtime/session/*
```

**Upload Failures**
```bash
# Check directory permissions
chmod 755 apps/flask/runtime/uploads/
```

### Debug Commands
```bash
# Check system status
curl http://localhost:5000/api/status

# Monitor session files
ls -la apps/flask/runtime/session/

# Check upload directory
ls -la apps/flask/runtime/uploads/

# View recent logs
tail -f /tmp/logs/Flask_AI_Virtual_Mentor_*.log
```