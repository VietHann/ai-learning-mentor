/**
 * AI Virtual Mentor - Web Application JavaScript
 * Modern interactive functionality for the CS education platform
 */

class AIVirtualMentor {
    constructor() {
        this.isLoading = false;
        this.currentConversationId = null;
        this.profile = {};
        this.settings = {};
        this.documents = [];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.setupDragAndDrop();
        this.loadProfile();
        this.loadConversationHistory();
        this.loadDocumentsList();
    }

    setupEventListeners() {
        // Question form submission
        document.getElementById('question-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleQuestion();
        });

        // Profile form submission
        document.getElementById('profile-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveProfile();
        });

        // Settings change listeners
        document.getElementById('integrity-mode').addEventListener('change', (e) => {
            this.updateSettings({ integrity_mode: e.target.value });
        });

        document.getElementById('preferred-language').addEventListener('change', (e) => {
            this.updateSettings({ preferred_language: e.target.value });
            this.profile.preferred_language = e.target.value;
        });

        // File upload
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Textarea auto-resize
        const textarea = document.getElementById('question-input');
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        });

        // Enter to submit (Ctrl+Enter for new line)
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault();
                if (!this.isLoading) {
                    this.handleQuestion();
                }
            }
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            }, false);
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFileUpload(files);
        }, false);

        uploadArea.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async loadInitialData() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.components_ready) {
                console.log('✅ System components ready');
            } else {
                this.showToast('System components not fully initialized', 'warning');
            }
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showToast('Failed to connect to server', 'error');
        }
    }

    async loadProfile() {
        try {
            const response = await fetch('/api/profile');
            const data = await response.json();
            
            this.profile = data.profile;
            this.updateProfileDisplay();
            this.updateProfileForm();
            this.updateAnalytics(data.analytics);
        } catch (error) {
            console.error('Failed to load profile:', error);
        }
    }

    async saveProfile() {
        const profileData = {
            name: document.getElementById('profile-name').value,
            student_id: document.getElementById('profile-student-id').value,
            major: document.getElementById('profile-major').value,
            year: document.getElementById('profile-year').value,
            level: document.getElementById('profile-level').value,
            preferred_language: document.getElementById('preferred-language').value
        };

        try {
            const response = await fetch('/api/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile: profileData })
            });

            const data = await response.json();
            
            if (data.success) {
                this.profile = data.profile;
                this.updateProfileDisplay();
                this.cancelProfileEdit();
                this.showToast('Profile saved successfully!', 'success');
            }
        } catch (error) {
            console.error('Failed to save profile:', error);
            this.showToast('Failed to save profile', 'error');
        }
    }

    updateProfileDisplay() {
        const greeting = document.getElementById('user-greeting');
        const details = document.getElementById('profile-details-text');

        if (this.profile.name) {
            greeting.textContent = `👋 Hi ${this.profile.name}!`;
            details.textContent = `${this.profile.major} - ${this.profile.year} - ${this.profile.level}`;
        } else {
            greeting.textContent = 'Welcome, Student!';
            details.textContent = 'Complete your profile to get personalized experience';
        }
    }

    updateProfileForm() {
        document.getElementById('profile-name').value = this.profile.name || '';
        document.getElementById('profile-student-id').value = this.profile.student_id || '';
        document.getElementById('profile-major').value = this.profile.major || 'Công nghệ Thông tin';
        document.getElementById('profile-year').value = this.profile.year || 'Năm 2';
        document.getElementById('profile-level').value = this.profile.level || 'Trung bình';
        document.getElementById('preferred-language').value = this.profile.preferred_language || 'Vietnamese';
    }

    updateAnalytics(analytics) {
        document.getElementById('questions-count').textContent = analytics.questions_asked;
        document.getElementById('topics-count').textContent = analytics.topics_explored;
        document.getElementById('responses-count').textContent = analytics.total_responses;
    }

    async updateSettings(newSettings) {
        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newSettings)
            });

            const data = await response.json();
            if (data.success) {
                this.settings = { ...this.settings, ...data.settings };
                this.showToast('Settings updated', 'success');
            }
        } catch (error) {
            console.error('Failed to update settings:', error);
            this.showToast('Failed to update settings', 'error');
        }
    }

    async handleFileUpload(files) {
        if (!files || files.length === 0) return;

        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });

        const progressContainer = document.getElementById('upload-progress');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        try {
            progressContainer.style.display = 'block';
            progressText.textContent = 'Uploading files...';
            
            // Simulate upload progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 10;
                progressFill.style.width = progress + '%';
                if (progress >= 50) clearInterval(progressInterval);
            }, 100);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                progressFill.style.width = '100%';
                progressText.textContent = `Processed ${data.processed_count} documents`;
                
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 2000);

                this.updateDocumentsList(data.documents);
                this.showToast(`Successfully processed ${data.processed_count} documents!`, 'success');
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload failed:', error);
            progressContainer.style.display = 'none';
            this.showToast(`Upload failed: ${error.message}`, 'error');
        }

        // Clear file input
        document.getElementById('file-input').value = '';
    }

    updateDocumentsList(documents) {
        const container = document.getElementById('documents-container');
        
        if (documents.length === 0) {
            container.innerHTML = '<p class="no-documents">No documents uploaded yet</p>';
            return;
        }

        container.innerHTML = documents.map(doc => `
            <div class="document-item">
                <i class="fas ${this.getFileIcon(doc.type)}"></i>
                <span>${doc.name}</span>
                <small>(${doc.chunks} chunks)</small>
            </div>
        `).join('');
    }

    getFileIcon(type) {
        const icons = {
            'PDF': 'fa-file-pdf',
            'MD': 'fa-file-text',
            'PY': 'fa-file-code',
            'JAVA': 'fa-file-code',
            'CPP': 'fa-file-code',
            'JS': 'fa-file-code',
            'HTML': 'fa-file-code',
            'CSS': 'fa-file-code'
        };
        return icons[type] || 'fa-file';
    }

    async handleQuestion() {
        if (this.isLoading) return;

        const questionInput = document.getElementById('question-input');
        const question = questionInput.value.trim();

        if (!question) {
            this.showToast('Please enter a question', 'warning');
            return;
        }

        this.isLoading = true;
        this.showLoading(true);
        
        // Add user message immediately
        this.addMessageToConversation({
            type: 'user',
            content: question,
            timestamp: new Date().toISOString()
        });

        // Clear input
        questionInput.value = '';
        questionInput.style.height = 'auto';

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    language: this.profile.preferred_language || 'Vietnamese'
                })
            });

            const data = await response.json();

            if (data.success) {
                // Clear any previous error messages on successful response
                this.clearErrorMessages();
                
                // Add assistant response
                this.addMessageToConversation({
                    type: 'assistant',
                    content: data.response,
                    citations: data.citations,
                    timestamp: new Date().toISOString(),
                    questionType: data.question_type,
                    executionTime: data.execution_time
                });

                // Update context and suggestions
                this.updateLearningContext(data);
                
                // Refresh analytics
                this.loadProfile();
                
                this.showToast('Response generated successfully!', 'success');
            } else {
                throw new Error(data.error || 'Failed to generate response');
            }
        } catch (error) {
            console.error('Question failed:', error);
            this.addMessageToConversation({
                type: 'assistant',
                content: `Sorry, I encountered an error: ${error.message}. Please try again or upload some educational documents first.`,
                timestamp: new Date().toISOString(),
                isError: true
            });
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.isLoading = false;
            this.showLoading(false);
        }
    }

    addMessageToConversation(message) {
        const history = document.getElementById('conversation-history');
        const messageElement = this.createMessageElement(message);
        
        history.appendChild(messageElement);
        
        // Scroll to bottom
        messageElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    createMessageElement(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-container';

        const isUser = message.type === 'user';
        const avatarIcon = isUser ? 'fa-user' : 'fa-robot';
        const messageClass = isUser ? 'user-message' : 'assistant-message';

        let citationsHtml = '';
        if (message.citations && message.citations.length > 0) {
            citationsHtml = `
                <div class="citations">
                    <h4><i class="fas fa-book"></i> Sources</h4>
                    <ul>
                        ${message.citations.map(citation => `<li>${citation}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        let metaInfo = '';
        if (message.questionType && !isUser) {
            metaInfo = `<small style="color: var(--gray-500); font-size: var(--font-size-xs);">
                Type: ${message.questionType.replace('_', ' ').toUpperCase()} 
                ${message.executionTime ? `• ${message.executionTime.toFixed(2)}s` : ''}
            </small>`;
        }

        messageDiv.innerHTML = `
            <div class="message ${messageClass}">
                <div class="message-avatar">
                    <i class="fas ${avatarIcon}"></i>
                </div>
                <div class="message-content ${message.isError ? 'error' : ''}">
                    ${this.formatMessageContent(message.content)}
                    ${citationsHtml}
                    ${metaInfo}
                </div>
            </div>
        `;

        return messageDiv;
    }

    formatMessageContent(content) {
        // First handle code blocks
        content = content.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, language, code) => {
            const lang = language || 'text';
            const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
            return `<div class="code-block">
                <div class="code-header">
                    <span class="code-language">${lang}</span>
                    <button class="copy-btn" onclick="copyCode('${codeId}')">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
                <pre><code id="${codeId}" class="language-${lang}">${this.escapeHtml(code.trim())}</code></pre>
            </div>`;
        });
        
        // Handle inline code
        content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
        
        // Basic formatting for better readability BEFORE citations to avoid conflicts
        content = content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^(.+)$/, '<p>$1</p>');
        
        // Handle inline citations AFTER markdown to avoid conflicts with * processing
        // Use word boundary approach to ensure proper spacing
        content = content.replace(/(\w)(\s*\["?Nguồn[^\]"]*"?\]\s*)/g, (match, precedingChar, citation) => {
            console.log('Processing citation with word boundary:', match, 'Preceding:', precedingChar, 'Citation:', citation);
            let cleanedCitation = citation.trim();
            // Remove outer brackets and quotes
            cleanedCitation = cleanedCitation.replace(/^\["?/, '').replace(/"?\]$/, '');
            // Remove "Nguồn:" or "Nguồn " prefix if present
            cleanedCitation = cleanedCitation.replace(/^Nguồn:?\s*/, '');
            
            console.log('Cleaned citation:', cleanedCitation);
            const citationText = this.escapeHtml(cleanedCitation).replace(/'/g, "\\'");
            
            // Preserve preceding character and add explicit space before citation
            return `${precedingChar} <span class="citation-ref" onmouseenter="showTooltip(this, '${citationText}')" onmouseleave="hideTooltip()"><sup class="citation-marker">[●]</sup></span>`;
        });
        
        // Handle remaining citations that weren't caught by word boundary
        content = content.replace(/\["?Nguồn[^\]"]*"?\]/g, (match) => {
            console.log('Processing remaining citation match:', match);
            let cleanedCitation = match.trim();
            // Remove outer brackets and quotes
            cleanedCitation = cleanedCitation.replace(/^\["?/, '').replace(/"?\]$/, '');
            // Remove "Nguồn:" or "Nguồn " prefix if present
            cleanedCitation = cleanedCitation.replace(/^Nguồn:?\s*/, '');
            
            console.log('Cleaned remaining citation:', cleanedCitation);
            const citationText = this.escapeHtml(cleanedCitation).replace(/'/g, "\\'");
            
            return ` <span class="citation-ref" onmouseenter="showTooltip(this, '${citationText}')" onmouseleave="hideTooltip()"><sup class="citation-marker">[●]</sup></span>`;
        });
        
        // Handle simple [Nguồn: ...] format with word boundary  
        content = content.replace(/(\w)(\s*\[Nguồn[^\]]*\]\s*)/g, (match, precedingChar, citation) => {
            console.log('Processing simple citation with word boundary:', match, 'Preceding:', precedingChar, 'Citation:', citation);
            let cleanedCitation = citation.trim().replace(/^\[/, '').replace(/\]$/, '');
            cleanedCitation = cleanedCitation.replace(/^Nguồn:?\s*/, '');
            
            console.log('Cleaned simple citation:', cleanedCitation);
            const citationText = this.escapeHtml(cleanedCitation).replace(/'/g, "\\'");
            
            return `${precedingChar} <span class="citation-ref" onmouseenter="showTooltip(this, '${citationText}')" onmouseleave="hideTooltip()"><sup class="citation-marker">[●]</sup></span>`;
        });
        
        // Handle remaining simple citations
        content = content.replace(/\[Nguồn[^\]]*\]/g, (match) => {
            console.log('Processing remaining simple citation match:', match);
            let cleanedCitation = match.trim().replace(/^\[/, '').replace(/\]$/, '');
            cleanedCitation = cleanedCitation.replace(/^Nguồn:?\s*/, '');
            
            console.log('Cleaned remaining simple citation:', cleanedCitation);
            const citationText = this.escapeHtml(cleanedCitation).replace(/'/g, "\\'");
            
            return ` <span class="citation-ref" onmouseenter="showTooltip(this, '${citationText}')" onmouseleave="hideTooltip()"><sup class="citation-marker">[●]</sup></span>`;
        });
        
        return content;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearErrorMessages() {
        // Remove all error messages from conversation history
        const history = document.getElementById('conversation-history');
        const errorMessages = history.querySelectorAll('.message-content.error');
        
        errorMessages.forEach(errorMessage => {
            const messageContainer = errorMessage.closest('.message-container');
            if (messageContainer) {
                messageContainer.remove();
            }
        });
        
        console.log('Cleared', errorMessages.length, 'error messages from conversation');
    }

    updateLearningContext(data) {
        // Update current topic
        const currentTopic = document.getElementById('current-topic');
        const topicName = document.getElementById('topic-name');
        
        if (data.question_type) {
            currentTopic.style.display = 'block';
            topicName.textContent = data.question_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        }

        // Update suggestions
        const suggestionsDiv = document.getElementById('suggestions');
        const suggestionsList = document.getElementById('suggestions-list');
        
        if (data.follow_up_suggestions && data.follow_up_suggestions.length > 0) {
            suggestionsDiv.style.display = 'block';
            suggestionsList.innerHTML = data.follow_up_suggestions
                .map(suggestion => `<li onclick="mentor.insertSuggestion('${suggestion}')">${suggestion}</li>`)
                .join('');
        }
    }

    async loadConversationHistory() {
        try {
            const response = await fetch('/api/conversation');
            const data = await response.json();
            
            // Display existing conversation history
            data.history.forEach(entry => {
                this.addMessageToConversation({
                    type: 'user',
                    content: entry.question,
                    timestamp: entry.timestamp
                });
                
                this.addMessageToConversation({
                    type: 'assistant',
                    content: entry.response,
                    citations: entry.citations,
                    timestamp: entry.timestamp,
                    questionType: entry.question_type
                });
            });

            // Update context
            if (data.context.current_topic) {
                document.getElementById('current-topic').style.display = 'block';
                document.getElementById('topic-name').textContent = data.context.current_topic;
            }

            if (data.context.follow_up_suggestions && data.context.follow_up_suggestions.length > 0) {
                document.getElementById('suggestions').style.display = 'block';
                document.getElementById('suggestions-list').innerHTML = 
                    data.context.follow_up_suggestions.slice(-3)
                        .map(suggestion => `<li onclick="mentor.insertSuggestion('${suggestion}')">${suggestion}</li>`)
                        .join('');
            }
        } catch (error) {
            console.error('Failed to load conversation history:', error);
        }
    }

    async loadDocumentsList() {
        try {
            const response = await fetch('/api/documents');
            const data = await response.json();
            
            if (data.documents) {
                this.documents = data.documents;
                this.updateDocumentsList(data.documents);
                console.log(`📄 Loaded ${data.count} documents from session`);
            }
        } catch (error) {
            console.error('Failed to load documents list:', error);
        }
    }

    insertSuggestion(suggestion) {
        const input = document.getElementById('question-input');
        input.value = suggestion;
        input.focus();
        input.style.height = 'auto';
        input.style.height = input.scrollHeight + 'px';
    }

    insertExample(type) {
        const examples = {
            theory: "Machine Learning là gì và có những loại chính nào?",
            debug: "Tại sao code Java này bị lỗi NullPointerException?",
            exercise: "Hướng dẫn tôi cách tiếp cận bài tập về TCP Socket programming"
        };

        const input = document.getElementById('question-input');
        const exampleText = examples[type] || examples.theory;
        
        // Set value and trigger events
        input.value = exampleText;
        
        // Trigger input and change events to ensure the form recognizes the value
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        
        input.focus();
        input.style.height = 'auto';
        input.style.height = input.scrollHeight + 'px';
        
        // Debug logging
        console.log('insertExample called with type:', type);
        console.log('Set input value to:', input.value);
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        const submitBtn = document.getElementById('submit-btn');
        
        overlay.style.display = show ? 'flex' : 'none';
        
        if (show) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Thinking...</span>';
        } else {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i><span>Ask</span>';
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: var(--spacing-sm);">
                <i class="fas ${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.style.animation = 'slideInRight 0.3s ease reverse';
            setTimeout(() => {
                if (container.contains(toast)) {
                    container.removeChild(toast);
                }
            }, 300);
        }, 5000);
    }

    getToastIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-triangle',
            warning: 'fa-exclamation-circle',
            info: 'fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    async clearAllData() {
        if (!confirm('Are you sure you want to clear all conversation history and documents? Your profile will be preserved.')) {
            return;
        }

        try {
            const response = await fetch('/api/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                // Clear UI
                document.getElementById('conversation-history').innerHTML = `
                    <div class="welcome-message">
                        <div class="message-container">
                            <div class="message assistant-message">
                                <div class="message-avatar">
                                    <i class="fas fa-robot"></i>
                                </div>
                                <div class="message-content">
                                    <h3>👋 Welcome back!</h3>
                                    <p>All data has been cleared. Upload some documents to get started again.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                document.getElementById('documents-container').innerHTML = 
                    '<p class="no-documents">No documents uploaded yet</p>';
                
                document.getElementById('current-topic').style.display = 'none';
                document.getElementById('suggestions').style.display = 'none';

                // Reset analytics
                document.getElementById('questions-count').textContent = '0';
                document.getElementById('topics-count').textContent = '0';
                document.getElementById('responses-count').textContent = '0';

                this.showToast('All data cleared successfully!', 'success');
            }
        } catch (error) {
            console.error('Failed to clear data:', error);
            this.showToast('Failed to clear data', 'error');
        }
    }
}

// Global functions for HTML onclick events
function toggleProfileEdit() {
    const editDiv = document.getElementById('profile-edit');
    const isHidden = editDiv.style.display === 'none';
    editDiv.style.display = isHidden ? 'block' : 'none';
}

// Citation tooltip functions
let activeTooltip = null;

function showTooltip(element, text) {
    console.log('showTooltip called with:', element, text);
    
    // Remove any existing tooltip
    hideTooltip();
    
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'citation-tooltip';
    tooltip.textContent = text;
    tooltip.style.position = 'fixed';
    tooltip.style.opacity = '0';
    tooltip.style.visibility = 'hidden';
    tooltip.style.zIndex = '10000';
    tooltip.style.background = '#2d3748';
    tooltip.style.color = 'white';
    tooltip.style.padding = '8px 12px';
    tooltip.style.borderRadius = '8px';
    tooltip.style.fontSize = '12px';
    tooltip.style.maxWidth = '300px';
    tooltip.style.lineHeight = '1.4';
    
    console.log('Created tooltip element:', tooltip);
    
    // Append to body for better positioning
    document.body.appendChild(tooltip);
    
    // Position tooltip relative to element
    const rect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    console.log('Element rect:', rect);
    console.log('Tooltip rect:', tooltipRect);
    
    // Position above the element
    const left = Math.max(10, Math.min(
        window.innerWidth - tooltipRect.width - 10,
        rect.left + (rect.width - tooltipRect.width) / 2
    ));
    const top = rect.top - tooltipRect.height - 8;
    
    console.log('Positioning tooltip at:', left, top);
    
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    tooltip.style.opacity = '1';
    tooltip.style.visibility = 'visible';
    
    activeTooltip = tooltip;
    
    console.log('Tooltip should now be visible');
}

function hideTooltip() {
    if (activeTooltip) {
        activeTooltip.remove();
        activeTooltip = null;
    }
}

function copyCode(codeId) {
    const codeElement = document.getElementById(codeId);
    if (codeElement) {
        const text = codeElement.textContent;
        navigator.clipboard.writeText(text).then(() => {
            // Show feedback
            const button = codeElement.closest('.code-block').querySelector('.copy-btn');
            const originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check"></i> Copied!';
            button.style.background = 'rgba(34, 197, 94, 0.2)';
            button.style.borderColor = 'rgba(34, 197, 94, 0.3)';
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.style.background = 'rgba(255, 255, 255, 0.1)';
                button.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy code:', err);
        });
    }
}

function cancelProfileEdit() {
    document.getElementById('profile-edit').style.display = 'none';
    mentor.updateProfileForm(); // Reset form to current values
}

function clearAllData() {
    mentor.clearAllData();
}

function insertExample(type) {
    console.log('Global insertExample called with type:', type);
    if (window.mentor && window.mentor.insertExample) {
        window.mentor.insertExample(type);
    } else {
        console.error('mentor instance not available or insertExample method missing');
    }
}

// Initialize the application
const mentor = new AIVirtualMentor();

// Export for global access
window.mentor = mentor;