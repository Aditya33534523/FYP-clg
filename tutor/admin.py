from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import ChatHistory


@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'question_short', 'subject', 'marks', 'confidence_pct', 'timestamp')
    list_filter = ('subject', 'marks', 'timestamp')
    search_fields = ('user__email', 'question', 'answer')
    readonly_fields = ('user', 'question', 'answer', 'confidence', 'subject', 'marks', 'timestamp')
    ordering = ('-timestamp',)

    def question_short(self, obj):
        return obj.question[:80] + '...' if len(obj.question) > 80 else obj.question
    question_short.short_description = 'Question'

    def confidence_pct(self, obj):
        return f"{obj.confidence:.0%}"
    confidence_pct.short_description = 'Confidence'
