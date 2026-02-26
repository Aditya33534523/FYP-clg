import json
import markdown as md
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.contrib.auth.models import User

from .forms import RegisterForm, QuestionForm
from .models import ChatHistory


def index(request):
    if request.user.is_authenticated:
        return redirect('chat')
    return redirect('login')


# ── FIX 1 & 2: Proper login view ──
def login_view(request):
    if request.user.is_authenticated:
        return redirect('chat')
    if request.method == 'POST':
        email    = request.POST.get('username', '').strip().lower()
        password = request.POST.get('password', '').strip()
        # Authenticate using email as username (that's how we store it)
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('chat')
        else:
            messages.error(request, 'Invalid email or password. Please try again.')
    return render(request, 'tutor/login.html')


# ── FIX 1 & 2: Proper register view ──
def register_view(request):
    if request.user.is_authenticated:
        return redirect('chat')
    if request.method == 'POST':
        email    = request.POST.get('email', '').strip().lower()
        password1 = request.POST.get('password1', '').strip()
        password2 = request.POST.get('password2', '').strip()

        # Validate fields
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
            # ── KEY FIX: user already exists → redirect to login ──
            messages.error(request, 'An account with this email already exists. Please sign in.')
            return redirect('login')
        else:
            # Create user
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password1
            )
            login(request, user)
            return redirect('chat')

    return render(request, 'tutor/register.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def chat_view(request):
    history = ChatHistory.objects.filter(user=request.user).order_by('timestamp')
    return render(request, 'tutor/chat.html', {
        'history': history,
        'user': request.user,
    })


@login_required
@require_POST
def ask_view(request):
    try:
        data     = json.loads(request.body)
        question = data.get('question', '').strip()
        marks    = int(data.get('marks', 1))
    except Exception:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    if not question:
        return JsonResponse({'error': 'Question cannot be empty'}, status=400)

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

    return JsonResponse({
        'id': chat.id,
        'answer': answer_html,
        'confidence': confidence,
        'conf_label': conf_label,
        'conf_class': conf_class,
        'subject': subject,
    })


@login_required
@require_POST
def clear_history_view(request):
    ChatHistory.objects.filter(user=request.user).delete()
    return JsonResponse({'status': 'ok'})