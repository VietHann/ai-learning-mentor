import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect, url_for, flash
from flask_cors import CORS
from flask_session import Session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

# Add parent directory to path for mentor_core imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Import core components
from mentor_core.document_processor import DocumentProcessor
from mentor_core.embeddings import EmbeddingGenerator
from mentor_core.vector_search import VectorSearch
from mentor_core.gemini_client import GeminiClient
from mentor_core.question_classifier import QuestionClassifier
from mentor_core.citation_formatter import CitationFormatter

# Import database models
from .models import db, bcrypt, User, ChatSession, UserDocument

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, origins="*", allow_headers="*", methods="*")

# Configure Flask app
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'ai-virtual-mentor-2025')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ai_virtual_mentor.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = './runtime/session'
app.config['SESSION_FILE_THRESHOLD'] = 500
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)
Session(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Global components
components = None

def initialize_components():
    """Initialize all system components"""
    global components
    
    try:
        doc_processor = DocumentProcessor()
        
        # Initialize embedding generator with fallback
        try:
            embedding_gen = EmbeddingGenerator()
        except Exception as e:
            print(f"Advanced embeddings not available, using simple fallback: {str(e)}")
            embedding_gen = EmbeddingGenerator()
        
        vector_search = VectorSearch()
        
        # Check if Gemini API key is available
        try:
            gemini_client = GeminiClient()
        except Exception as e:
            print(f"Gemini API initialization failed: {str(e)}")
            return None
            
        question_classifier = QuestionClassifier()
        citation_formatter = CitationFormatter()
        
        components = {
            'doc_processor': doc_processor,
            'embedding_gen': embedding_gen,
            'vector_search': vector_search,
            'gemini_client': gemini_client,
            'question_classifier': question_classifier,
            'citation_formatter': citation_formatter
        }
        
        return components
        
    except Exception as e:
        print(f"Critical initialization error: {str(e)}")
        return None

def initialize_session():
    """Initialize session state based on authentication status"""
    
    if current_user.is_authenticated:
        # For authenticated users, load from database
        if 'user_session_id' not in session:
            session['user_session_id'] = get_user_session_id()
        
        # Load user profile from database
        session['user_profile'] = current_user.get_profile_dict()
        
        # Load user documents
        user_docs = UserDocument.query.filter_by(user_id=current_user.id).all()
        session['documents_processed'] = [doc.get_dict() for doc in user_docs]
        
        # Load conversation history (recent sessions)
        recent_chats = ChatSession.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatSession.created_at.desc()).limit(50).all()
        
        session['conversation_history'] = [chat.get_dict() for chat in recent_chats]
        
        # Set authenticated user embeddings ready if they have documents
        session['embeddings_ready'] = len(user_docs) > 0
    
    else:
        # For anonymous users, use legacy session system
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        if 'documents_processed' not in session:
            session['documents_processed'] = []
        if 'user_profile' not in session:
            session['user_profile'] = {
                'name': '',
                'email': '',
                'major': '',
                'academic_year': '',
                'preferred_language': 'Vietnamese'
            }
        if 'embeddings_ready' not in session:
            session['embeddings_ready'] = False
        if 'academic_mode' not in session:
            session['academic_mode'] = True
    
    # Initialize session analytics and conversation context for all users
    if 'session_analytics' not in session:
        if current_user.is_authenticated:
            # Calculate analytics from database for authenticated users
            total_chats = ChatSession.query.filter_by(user_id=current_user.id).count()
            unique_topics = db.session.query(ChatSession.question_type).filter_by(
                user_id=current_user.id
            ).distinct().all()
            topics_list = [topic[0] for topic in unique_topics if topic[0]]
            
            session['session_analytics'] = {
                'questions_asked': total_chats,
                'total_responses': total_chats,
                'topics_explored': topics_list,
                'documents_uploaded': len(session.get('documents_processed', [])),
                'question_types_count': {},
                'average_response_time': 0.0,
                'session_start': datetime.utcnow().isoformat()
            }
        else:
            # Default analytics for anonymous users
            session['session_analytics'] = {
                'questions_asked': 0,
                'total_responses': 0,
                'topics_explored': [],
                'documents_uploaded': 0,
                'question_types_count': {},
                'average_response_time': 0.0,
                'session_start': datetime.utcnow().isoformat()
            }
    
    if 'conversation_context' not in session:
        session['conversation_context'] = {
            'current_topic': '',
            'follow_up_suggestions': []
        }

def get_user_session_id():
    """Get user-specific session ID"""
    if current_user.is_authenticated:
        return f"user_{current_user.id}"
    else:
        if 'anonymous_session_id' not in session:
            session['anonymous_session_id'] = str(uuid.uuid4())
        return f"anonymous_{session['anonymous_session_id']}"

def create_tables():
    """Create database tables"""
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Error creating database tables: {str(e)}")

# Initialize database tables on startup
create_tables()

def generate_follow_up_suggestions(question_type: str, question: str) -> List[str]:
    """Generate contextual follow-up suggestions based on question type"""
    suggestions_map = {
        'theory_explanation': [
            "Ví dụ thực tế về khái niệm này",
            "Ứng dụng của lý thuyết trong thực tế",
            "So sánh với các khái niệm tương tự"
        ],
        'code_debugging': [
            "Cách phòng tránh lỗi tương tự",
            "Best practices cho loại code này",
            "Tools để debug hiệu quả hơn"
        ],
        'exercise': [
            "Bài tập nâng cao về chủ đề này",
            "Cách tối ưu giải pháp",
            "Test cases để kiểm tra"
        ],
        'definition': [
            "Khái niệm liên quan",
            "Lịch sử phát triển",
            "Xu hướng tương lai"
        ]
    }
    
    return suggestions_map.get(question_type, ["Chủ đề liên quan", "Khái niệm mở rộng"])

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        full_name = data.get('full_name', '').strip()
        major = data.get('major', '').strip()
        
        # Validation
        if not all([username, email, password]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
        
        if not User.validate_email(email):
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400
        
        # Check for existing user
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        try:
            # Create new user
            user = User(username=username, email=email, full_name=full_name, major=major)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            # Auto login after registration
            login_user(user, remember=True)
            
            return jsonify({
                'success': True,
                'message': 'Registration successful',
                'user': user.get_profile_dict()
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': f'Registration failed: {str(e)}'}), 500
    
    return render_template('auth.html', mode='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username_or_email = data.get('username', '').strip()
        password = data.get('password', '').strip()
        remember_me = data.get('remember', False)
        
        if not all([username_or_email, password]):
            return jsonify({'success': False, 'error': 'Username/email and password are required'}), 400
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if user and user.check_password(password):
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Login user
            login_user(user, remember=remember_me)
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': user.get_profile_dict()
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('auth.html', mode='login')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def index():
    """Main application page"""
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    initialize_session()
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'components_ready': components is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/documents')
@login_required
def api_documents():
    """Get list of processed documents (user-specific)"""
    initialize_session()
    
    # Get documents from database for current user only
    user_docs = UserDocument.query.filter_by(user_id=current_user.id).all()
    documents = [doc.get_dict() for doc in user_docs]
    
    return jsonify({
        'documents': documents,
        'count': len(documents)
    })

@app.route('/api/profile', methods=['GET', 'POST'])
@login_required
def api_profile():
    """User profile management (user-specific)"""
    initialize_session()
    
    if request.method == 'GET':
        # Calculate analytics from database
        total_chats = ChatSession.query.filter_by(user_id=current_user.id).count()
        unique_topics = db.session.query(ChatSession.question_type).filter_by(
            user_id=current_user.id
        ).distinct().count()
        
        return jsonify({
            'profile': current_user.get_profile_dict(),
            'analytics': {
                'questions_asked': total_chats,
                'topics_explored': unique_topics,
                'total_responses': total_chats,
                'documents_uploaded': UserDocument.query.filter_by(user_id=current_user.id).count()
            }
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        # Update user profile in database
        if 'profile' in data:
            current_user.update_profile(data['profile'])
            db.session.commit()
            
            # Update session cache
            session['user_profile'] = current_user.get_profile_dict()
            session.permanent = True
        
        return jsonify({
            'success': True,
            'profile': current_user.get_profile_dict()
        })

@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    """Document upload and processing (user-specific)"""
    initialize_session()
    
    if not components:
        return jsonify({'error': 'System components not initialized'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Process documents
        doc_processor = components['doc_processor']
        embedding_gen = components['embedding_gen']
        vector_search = components['vector_search']
        
        processed_docs = []
        saved_documents = []
        
        for file in files:
            if file.filename:
                # Create file hash for deduplication
                file_content = file.read()
                file_hash = hashlib.sha256(file_content).hexdigest()
                file.seek(0)  # Reset file pointer
                
                # Check if user already has this document
                existing_doc = UserDocument.query.filter_by(
                    user_id=current_user.id,
                    file_hash=file_hash
                ).first()
                
                if existing_doc:
                    continue  # Skip duplicate files
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
                file.save(temp_path)
                
                try:
                    # Process the document
                    docs = doc_processor.process_documents([temp_path])
                    
                    if docs:
                        # Create user document record
                        user_doc = UserDocument(
                            user_id=current_user.id,
                            filename=f"user_{current_user.id}_{file.filename}",
                            original_filename=file.filename,
                            file_type=file.filename.split('.')[-1].upper(),
                            file_size=len(file_content),
                            file_hash=file_hash,
                            chunks_count=len(docs),
                            processing_status='processed',
                            processed_at=datetime.utcnow()
                        )
                        db.session.add(user_doc)
                        
                        # Tag documents with user ID for vector search isolation
                        for doc in docs:
                            doc['user_id'] = current_user.id
                            doc['document_id'] = user_doc.id
                        
                        processed_docs.extend(docs)
                        saved_documents.append(user_doc)
                    
                except Exception as doc_error:
                    # Mark document as error in database
                    user_doc = UserDocument(
                        user_id=current_user.id,
                        filename=f"user_{current_user.id}_{file.filename}",
                        original_filename=file.filename,
                        file_type=file.filename.split('.')[-1].upper(),
                        file_size=len(file_content),
                        file_hash=file_hash,
                        processing_status='error',
                        error_message=str(doc_error)
                    )
                    db.session.add(user_doc)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        if processed_docs:
            # Generate embeddings from full documents 
            embeddings = embedding_gen.generate_embeddings(processed_docs)
            
            # Add user-specific documents to vector search
            vector_search.add_documents(processed_docs, embeddings)
            
            session['embeddings_ready'] = True
            session.permanent = True
        
        # Commit all document records
        db.session.commit()
        
        # Update session with user documents
        user_docs = UserDocument.query.filter_by(user_id=current_user.id).all()
        session['documents_processed'] = [doc.get_dict() for doc in user_docs]
        
        return jsonify({
            'success': True,
            'processed_count': len(processed_docs),
            'documents': session['documents_processed'],
            'new_documents': len(saved_documents)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Handle user questions"""
    initialize_session()
    
    if not components:
        return jsonify({'error': 'System components not initialized'}), 500
    
    data = request.get_json()
    question = data.get('question', '').strip()
    language = data.get('language', 'Vietnamese')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        start_time = datetime.now()
        
        # Get components
        vector_search = components['vector_search']
        gemini_client = components['gemini_client']
        question_classifier = components['question_classifier']
        citation_formatter = components['citation_formatter']
        embedding_gen = components['embedding_gen']
        
        # Classify question
        question_type = question_classifier.classify(question)
        
        # Search for relevant documents
        # First generate embedding for the question
        query_embedding = embedding_gen.generate_query_embedding(question)
        
        # Use hybrid search for better results (combines vector + BM25)
        relevant_docs = vector_search.hybrid_search(query_embedding, question, top_k=6)
        
        if not relevant_docs:
            return jsonify({
                'error': 'No relevant documents found. Please upload some educational materials first.'
            }), 400
        
        # Generate response
        integrity_mode = session.get('integrity_mode', 'academic')
        academic_mode = session.get('academic_mode', True)
        
        response, citations = gemini_client.generate_educational_response(
            question=question,
            question_type=question_type,
            relevant_docs=relevant_docs,
            academic_mode=academic_mode,
            integrity_mode=integrity_mode,
            language=language
        )
        
        # Format citations
        formatted_citations = citation_formatter.format_citations(citations)
        
        # Enhanced conversation history with user context
        conversation_entry = {
            'id': str(uuid.uuid4()),
            'question': question,
            'response': response,
            'citations': formatted_citations,
            'question_type': question_type,
            'timestamp': datetime.now().isoformat(),
            'user_profile_snapshot': {
                'name': session['user_profile'].get('name', 'Anonymous'),
                'major': session['user_profile'].get('major', ''),
                'level': session['user_profile'].get('level', ''),
                'year': session['user_profile'].get('year', '')
            },
            'integrity_mode': integrity_mode,
            'language': language,
            'relevant_docs_count': len(relevant_docs),
            'response_length': len(response)
        }
        
        session['conversation_history'].append(conversation_entry)
        
        # Update session analytics
        session['session_analytics']['questions_asked'] += 1
        session['session_analytics']['total_responses'] += 1
        
        if question_type not in session['session_analytics']['topics_explored']:
            session['session_analytics']['topics_explored'].append(question_type)
        
        # Update question type count
        if question_type in session['session_analytics']['question_types_count']:
            session['session_analytics']['question_types_count'][question_type] += 1
        else:
            session['session_analytics']['question_types_count'][question_type] = 1
        
        # Update conversation context
        session['conversation_context']['current_topic'] = question_type.replace('_', ' ').title()
        
        # Generate follow-up suggestions
        follow_up_suggestions = generate_follow_up_suggestions(question_type, question)
        if follow_up_suggestions:
            session['conversation_context']['follow_up_suggestions'].extend(follow_up_suggestions)
        
        session.permanent = True
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'success': True,
            'response': response,
            'citations': formatted_citations,
            'question_type': question_type,
            'execution_time': execution_time,
            'conversation_id': conversation_entry['id'],
            'follow_up_suggestions': follow_up_suggestions[:3]  # Limit to 3 suggestions
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

@app.route('/api/conversation')
@login_required
def api_conversation():
    """Get conversation history (user-specific)"""
    initialize_session()
    
    # Get recent chat sessions for current user
    recent_chats = ChatSession.query.filter_by(
        user_id=current_user.id
    ).order_by(ChatSession.created_at.desc()).limit(50).all()
    
    conversation_history = [chat.get_dict() for chat in recent_chats]
    
    return jsonify({
        'history': conversation_history,
        'context': session.get('conversation_context', {})
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Manage application settings"""
    initialize_session()
    
    if request.method == 'GET':
        return jsonify({
            'integrity_mode': session.get('integrity_mode', 'academic'),
            'academic_mode': session.get('academic_mode', True),
            'preferred_language': session['user_profile'].get('preferred_language', 'Vietnamese')
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'integrity_mode' in data:
            session['integrity_mode'] = data['integrity_mode']
            session['academic_mode'] = (data['integrity_mode'] != 'normal')
        
        if 'preferred_language' in data:
            session['user_profile']['preferred_language'] = data['preferred_language']
        
        session.permanent = True
        
        return jsonify({
            'success': True,
            'settings': {
                'integrity_mode': session.get('integrity_mode'),
                'academic_mode': session.get('academic_mode'),
                'preferred_language': session['user_profile'].get('preferred_language')
            }
        })

@app.route('/api/clear', methods=['POST'])
def api_clear():
    """Clear session data"""
    initialize_session()
    
    # Clear conversation and documents but preserve user profile
    session['conversation_history'] = []
    session['documents_processed'] = []
    session['embeddings_ready'] = False
    
    # Reset analytics but preserve user profile
    session['session_analytics'] = {
        'questions_asked': 0,
        'topics_explored': [],
        'session_start': datetime.now().isoformat(),
        'total_responses': 0,
        'question_types_count': {},
        'average_response_time': 0.0
    }
    
    # Reset conversation context
    session['conversation_context'] = {
        'current_topic': None,
        'context_documents': [],
        'learning_path': [],
        'follow_up_suggestions': []
    }
    
    # Clear vector search data
    if components and 'vector_search' in components:
        components['vector_search'].clear()
    
    session.permanent = True
    
    return jsonify({'success': True, 'message': 'Session data cleared successfully'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Initializing AI Virtual Mentor Web Application...")
    
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('flask_session', exist_ok=True)
    
    # Initialize system components
    initialize_components()
    
    if components:
        print("✅ All components initialized successfully")
        print("🌐 Starting Flask web server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Component initialization failed")
        print("Please check your API keys and dependencies")