"""
Session Management Middleware and Utilities
Handles user session tracking, activity monitoring, and session cleanup
"""

from datetime import datetime, timedelta
from django.contrib.sessions.models import Session
from django.contrib.auth.models import User
from django.utils.timezone import now


class SessionActivityTracker:
    """Tracks and manages user session activity"""
    
    SESSION_TIMEOUT_MINUTES = 60  # 1 hour inactivity timeout
    
    @staticmethod
    def initialize_session(request):
        """Initialize user session with tracking info"""
        if 'user_session_id' not in request.session:
            request.session['user_session_id'] = request.session.session_key
            request.session['user_session_start'] = datetime.now().isoformat()
            request.session['queries_in_session'] = 0
            request.session['question_corrections'] = {}
            request.session['session_ip'] = get_client_ip(request)
            request.session.modified = True
    
    @staticmethod
    def update_activity(request):
        """Update last activity timestamp"""
        request.session['last_activity'] = datetime.now().isoformat()
        request.session.modified = True
    
    @staticmethod
    def increment_query_count(request):
        """Increment query count for session"""
        if 'queries_in_session' not in request.session:
            request.session['queries_in_session'] = 0
        request.session['queries_in_session'] += 1
        request.session.modified = True
    
    @staticmethod
    def add_correction(request, original, corrected):
        """Record a typo correction in session"""
        if 'question_corrections' not in request.session:
            request.session['question_corrections'] = {}
        request.session['question_corrections'][original] = corrected
        request.session.modified = True
    
    @staticmethod
    def get_session_stats(request):
        """Get comprehensive session statistics"""
        return {
            'session_id': request.session.get('user_session_id'),
            'session_start': request.session.get('user_session_start'),
            'last_activity': request.session.get('last_activity'),
            'total_queries': request.session.get('queries_in_session', 0),
            'corrections_applied': len(request.session.get('question_corrections', {})),
            'user_ip': request.session.get('session_ip'),
        }
    
    @staticmethod
    def is_session_inactive(request, timeout_minutes=None):
        """Check if session has been inactive for too long"""
        if timeout_minutes is None:
            timeout_minutes = SessionActivityTracker.SESSION_TIMEOUT_MINUTES
        
        last_activity_str = request.session.get('last_activity')
        if not last_activity_str:
            return False
        
        try:
            last_activity = datetime.fromisoformat(last_activity_str)
            inactive_duration = datetime.now() - last_activity
            return inactive_duration > timedelta(minutes=timeout_minutes)
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def cleanup_old_sessions(hours=24):
        """Clean up sessions older than specified hours"""
        expire_date = now() - timedelta(hours=hours)
        Session.objects.filter(expire_date__lt=expire_date).delete()


class SessionMiddleware:
    """Custom middleware for session tracking and management"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Initialize session on authenticated requests
        if request.user.is_authenticated:
            SessionActivityTracker.initialize_session(request)
            SessionActivityTracker.update_activity(request)
        
        response = self.get_response(request)
        return response


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def get_user_session_info(user):
    """Get all active sessions for a user"""
    user_sessions = []
    for session in Session.objects.all():
        session_data = session.get_decoded()
        if session_data.get('_auth_user_id') == str(user.id):
            user_sessions.append({
                'session_key': session.session_key,
                'created': session_data.get('user_session_start'),
                'last_activity': session_data.get('last_activity'),
                'queries': session_data.get('queries_in_session', 0),
                'expire_date': session.expire_date,
            })
    return user_sessions


def invalidate_user_sessions(user, exclude_session_key=None):
    """Invalidate all sessions for a user except the specified one"""
    count = 0
    for session in Session.objects.all():
        session_data = session.get_decoded()
        if session_data.get('_auth_user_id') == str(user.id):
            if exclude_session_key and session.session_key == exclude_session_key:
                continue
            session.delete()
            count += 1
    return count
