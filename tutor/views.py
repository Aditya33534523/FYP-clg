import json
from datetime import datetime
import markdown as md
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.contrib.auth.models import User

from .forms import RegisterForm, QuestionForm
from .models import ChatHistory
from .typo_corrector import correct_query
from .session_manager import SessionActivityTracker


# ── HELPER: apply strict no-cache headers to any response ──
def _no_cache(response):
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, private'
    response['Pragma']        = 'no-cache'
    response['Expires']       = '0'
    response['X-Auth-State']  = 'logged-out'   # used by JS guard
    return response


def _no_cache_auth(response):
    """Same but marks page as requiring authentication."""
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, private'
    response['Pragma']        = 'no-cache'
    response['Expires']       = '0'
    response['X-Auth-State']  = 'logged-in'    # used by JS guard
    return response


def index(request):
    if request.user.is_authenticated:
        return redirect('chat')
    return redirect('login')


@never_cache
def login_view(request):
    if request.user.is_authenticated:
        return redirect('chat')
    if request.method == 'POST':
        email    = request.POST.get('username', '').strip().lower()
        password = request.POST.get('password', '').strip()
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            response = redirect('chat')
            return _no_cache(response)
        else:
            messages.error(request, 'Invalid email or password. Please try again.')
    response = render(request, 'tutor/login.html')
    return _no_cache(response)


def register_view(request):
    if request.user.is_authenticated:
        return redirect('chat')
    if request.method == 'POST':
        email     = request.POST.get('email', '').strip().lower()
        password1 = request.POST.get('password1', '').strip()
        password2 = request.POST.get('password2', '').strip()

        if not email or not password1 or not password2:
            messages.error(request, 'All fields are required.')
        elif password1 != password2:
            messages.error(request, 'Passwords do not match.')
        elif len(password1) < 8:
            messages.error(request, 'Password must be at least 8 characters.')
        elif not any(c.isupper() for c in password1):
            messages.error(request, 'Password must contain at least one uppercase letter.')
        elif not any(c.islower() for c in password1):
            messages.error(request, 'Password must contain at least one lowercase letter.')
        elif not any(c.isdigit() for c in password1):
            messages.error(request, 'Password must contain at least one digit.')
        elif not any(c in '!@#$%^&*(),.?":{}|<>' for c in password1):
            messages.error(request, 'Password must contain at least one special character.')
        elif User.objects.filter(username=email).exists():
            messages.error(request, 'An account with this email already exists. Please sign in.')
            return redirect('login')
        else:
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password1
            )
            login(request, user)
            return redirect('chat')

    response = render(request, 'tutor/register.html')
    return _no_cache(response)


@never_cache
def logout_view(request):
    # 1. Flush the entire session (kills all session data)
    request.session.flush()
    # 2. Django auth logout (clears auth cookie)
    logout(request)
    # 3. Redirect to login with aggressive no-cache headers
    response = redirect('login')
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, private'
    response['Pragma']        = 'no-cache'
    response['Expires']       = '0'
    # 4. Expire the session cookie immediately
    response.delete_cookie('sessionid')
    return response


@login_required
@never_cache
def chat_view(request):
    SessionActivityTracker.initialize_session(request)
    SessionActivityTracker.update_activity(request)

    if SessionActivityTracker.is_session_inactive(request):
        messages.warning(request, 'Your session has been inactive for a while. Please refresh.')

    history      = ChatHistory.objects.filter(user=request.user).order_by('-timestamp')
    session_stats = SessionActivityTracker.get_session_stats(request)

    response = render(request, 'tutor/chat.html', {
        'history':       history,
        'user':          request.user,
        'session_stats': session_stats,
    })
    # Absolutely no caching for the chat page
    return _no_cache_auth(response)


@login_required
@require_POST
@never_cache
def ask_view(request):
    try:
        data     = json.loads(request.body)
        question = data.get('question', '').strip()
        marks    = int(data.get('marks', 1))
    except Exception:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

    SessionActivityTracker.initialize_session(request)
    SessionActivityTracker.update_activity(request)
    SessionActivityTracker.increment_query_count(request)

    original_question = question
    question, corrections_made = correct_query(question, use_fuzzy=True, fuzzy_threshold=0.7)

    for original, corrected in corrections_made.items():
        SessionActivityTracker.add_correction(request, original, corrected)

    try:
        from .rag_engine import get_rag_engine
        engine = get_rag_engine()
        answer_raw, confidence, subject = engine.answer_question(question, marks)
        answer_html = md.markdown(answer_raw, extensions=['extra', 'nl2br'])
    except Exception as e:
        print(f"RAG Error: {e}")
        answer_html = "<p>AI engine is unavailable. Please ensure Ollama is running.</p>"
        confidence  = 0.0
        subject     = "N/A"

    chat = ChatHistory.objects.create(
        user=request.user,
        question=question,
        answer=answer_html,
        confidence=confidence,
        subject=subject,
        marks=marks
    )

    conf_label = 'High' if confidence >= 0.6 else ('Medium' if confidence >= 0.35 else 'Low')
    conf_class = 'conf-high' if confidence >= 0.6 else ('conf-medium' if confidence >= 0.35 else 'conf-low')

    correction_info = ""
    if corrections_made:
        correction_info = "<p class='correction-notice'><strong>Auto-corrected:</strong> "
        correction_parts = [f"'{orig}' → '{corr}'" for orig, corr in corrections_made.items()]
        correction_info += ", ".join(correction_parts) + "</p>"

    resp = JsonResponse({
        'id':                chat.id,
        'answer':            answer_html,
        'confidence':        confidence,
        'conf_label':        conf_label,
        'conf_class':        conf_class,
        'subject':           subject,
        'corrections':       corrections_made,
        'correction_info':   correction_info,
        'original_question': original_question,
        'corrected_question': question if corrections_made else None,
    })
    return _no_cache_auth(resp)


@login_required
@require_POST
def clear_history_view(request):
    ChatHistory.objects.filter(user=request.user).delete()
    return JsonResponse({'status': 'ok'})


@never_cache
def ping_view(request):
    """
    Lightweight session-check endpoint used by the JS back-button guard.
    Returns 200 if the user is authenticated, 401 otherwise.
    The browser never caches this response.
    """
    if request.user.is_authenticated:
        resp = JsonResponse({'authenticated': True})
    else:
        resp = JsonResponse({'authenticated': False}, status=401)
    return _no_cache(resp)