import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_session import Session

# Import core components
from mentor_core.document_processor import DocumentProcessor
from mentor_core.embeddings import EmbeddingGenerator
from mentor_core.vector_search import VectorSearch
from mentor_core.gemini_client import GeminiClient
from mentor_core.question_classifier import QuestionClassifier
from mentor_core.citation_formatter import CitationFormatter

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, origins="*", allow_headers="*", methods="*")

# Configure Flask session
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'ai-virtual-mentor-2025')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = './runtime/session'
app.config['SESSION_FILE_THRESHOLD'] = 500
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Initialize session
Session(app)

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
    """Initialize comprehensive session state"""
    
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    if 'documents_processed' not in session:
        session['documents_processed'] = []
    if 'embeddings_ready' not in session:
        session['embeddings_ready'] = False
    if 'academic_mode' not in session:
        session['academic_mode'] = True
    if 'integrity_mode' not in session:
        session['integrity_mode'] = "academic"
    
    # Enhanced User Profile System
    if 'user_profile' not in session:
        session['user_profile'] = {
            'name': '',
            'student_id': '',
            'major': 'Công nghệ Thông tin',
            'year': 'Năm 2',
            'level': 'Trung bình',
            'preferred_language': 'Vietnamese',
            'learning_goals': [],
            'favorite_topics': [],
            'created_at': datetime.now().isoformat()
        }
    
    # Session analytics and learning tracking
    if 'session_analytics' not in session:
        session['session_analytics'] = {
            'questions_asked': 0,
            'topics_explored': [],
            'session_start': datetime.now().isoformat(),
            'total_responses': 0,
            'question_types_count': {},
            'average_response_time': 0.0
        }
    
    # Enhanced conversation context
    if 'conversation_context' not in session:
        session['conversation_context'] = {
            'current_topic': None,
            'context_documents': [],
            'learning_path': [],
            'follow_up_suggestions': []
        }

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

@app.route('/')
def index():
    """Main application page"""
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
def api_documents():
    """Get list of processed documents"""
    initialize_session()
    
    return jsonify({
        'documents': session['documents_processed'],
        'count': len(session['documents_processed'])
    })

@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    """User profile management"""
    initialize_session()
    
    if request.method == 'GET':
        return jsonify({
            'profile': session['user_profile'],
            'analytics': {
                'questions_asked': session['session_analytics']['questions_asked'],
                'topics_explored': len(set(session['session_analytics']['topics_explored'])),
                'total_responses': session['session_analytics']['total_responses']
            }
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        # Update user profile
        if 'profile' in data:
            session['user_profile'].update(data['profile'])
            session.permanent = True
        
        return jsonify({
            'success': True,
            'profile': session['user_profile']
        })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Document upload and processing"""
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
        
        for file in files:
            if file.filename:
                # Save uploaded file temporarily
                temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
                file.save(temp_path)
                
                try:
                    # Process the document
                    docs = doc_processor.process_documents([temp_path])
                    processed_docs.extend(docs)
                    
                    # Check for duplicates before adding to session tracking
                    existing_docs = [doc['name'] for doc in session['documents_processed']]
                    if file.filename not in existing_docs:
                        session['documents_processed'].append({
                            'name': file.filename,
                            'type': file.filename.split('.')[-1].upper(),
                            'chunks': len(docs),
                            'processed_at': datetime.now().isoformat()
                        })
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        if processed_docs:
            # Generate embeddings from full documents 
            embeddings = embedding_gen.generate_embeddings(processed_docs)
            
            # Add documents to vector search with embeddings
            vector_search.add_documents(processed_docs, embeddings)
            
            session['embeddings_ready'] = True
            session.permanent = True
        
        return jsonify({
            'success': True,
            'processed_count': len(processed_docs),
            'documents': session['documents_processed']
        })
        
    except Exception as e:
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
def api_conversation():
    """Get conversation history"""
    initialize_session()
    
    return jsonify({
        'history': session['conversation_history'],
        'context': session['conversation_context']
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