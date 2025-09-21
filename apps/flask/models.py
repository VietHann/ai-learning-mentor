from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_bcrypt import Bcrypt
from email_validator import validate_email, EmailNotValidError

# Initialize extensions
db = SQLAlchemy()
bcrypt = Bcrypt()

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    
    # Profile information
    full_name = db.Column(db.String(200))
    major = db.Column(db.String(100))
    academic_year = db.Column(db.String(20))
    preferred_language = db.Column(db.String(10), default='Vietnamese')
    
    # Settings
    integrity_mode = db.Column(db.String(20), default='Normal')
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    documents = db.relationship('UserDocument', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password: str):
        """Hash and set password"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """Check password against hash"""
        return bcrypt.check_password_hash(self.password_hash, password)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def get_profile_dict(self) -> dict:
        """Get profile as dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'major': self.major,
            'academic_year': self.academic_year,
            'preferred_language': self.preferred_language,
            'integrity_mode': self.integrity_mode,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def update_profile(self, profile_data: dict):
        """Update user profile"""
        allowed_fields = ['full_name', 'major', 'academic_year', 'preferred_language', 'integrity_mode']
        for field in allowed_fields:
            if field in profile_data:
                setattr(self, field, profile_data[field])
    
    def __repr__(self):
        return f'<User {self.username}>'


class ChatSession(db.Model):
    """Chat session model for conversation history"""
    
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    session_id = db.Column(db.String(100), nullable=False, index=True)
    
    # Message content
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(50))
    
    # Context and citations
    citations = db.Column(db.JSON)
    context_used = db.Column(db.JSON)
    
    # Metadata
    integrity_mode = db.Column(db.String(20))
    language = db.Column(db.String(10))
    processing_time = db.Column(db.Float)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def get_dict(self) -> dict:
        """Get chat session as dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'question': self.question,
            'response': self.response,
            'question_type': self.question_type,
            'citations': self.citations,
            'context_used': self.context_used,
            'integrity_mode': self.integrity_mode,
            'language': self.language,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<ChatSession {self.id} - User {self.user_id}>'


class UserDocument(db.Model):
    """User document model for uploaded files"""
    
    __tablename__ = 'user_documents'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Document metadata
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    file_size = db.Column(db.Integer)
    file_hash = db.Column(db.String(64), index=True)  # For deduplication
    
    # Processing info
    chunks_count = db.Column(db.Integer, default=0)
    processing_status = db.Column(db.String(20), default='pending')  # pending, processed, error
    error_message = db.Column(db.Text)
    
    # Vector storage reference
    vector_id = db.Column(db.String(100))  # Reference to vector database
    
    # Timestamps
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    
    def get_dict(self) -> dict:
        """Get document as dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'chunks_count': self.chunks_count,
            'processing_status': self.processing_status,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }
    
    def __repr__(self):
        return f'<UserDocument {self.filename} - User {self.user_id}>'