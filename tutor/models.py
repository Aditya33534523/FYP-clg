from django.db import models
from django.contrib.auth.models import User


class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_history')
    question = models.TextField()
    answer = models.TextField()
    confidence = models.FloatField(default=0.0)
    subject = models.CharField(max_length=100, blank=True)
    marks = models.IntegerField(default=1)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Chat History'
        verbose_name_plural = 'Chat Histories'

    def __str__(self):
        return f"{self.user.email} — {self.question[:60]}"

