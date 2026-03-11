"""
Django management command to clean up expired sessions
Run: python manage.py cleanup_sessions
"""

from django.core.management.base import BaseCommand
from django.contrib.sessions.models import Session
from django.utils import timezone
from datetime import timedelta
from tutor.session_manager import SessionActivityTracker


class Command(BaseCommand):
    help = 'Clean up expired and inactive sessions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hours',
            type=int,
            default=24,
            help='Delete sessions older than this many hours (default: 24)',
        )
        parser.add_argument(
            '--inactive-hours',
            type=int,
            default=None,
            help='Delete sessions inactive for this many hours (optional)',
        )

    def handle(self, *args, **options):
        hours = options['hours']
        inactive_hours = options.get('inactive_hours')
        
        # Clean up expired sessions
        expire_date = timezone.now() - timedelta(hours=hours)
        expired_count, _ = Session.objects.filter(expire_date__lt=expire_date).delete()
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Deleted {expired_count} expired sessions (older than {hours} hours)')
        )
        
        # Clean up inactive sessions if specified
        if inactive_hours:
            inactive_count = 0
            for session in Session.objects.all():
                try:
                    session_data = session.get_decoded()
                    last_activity_str = session_data.get('last_activity')
                    
                    if last_activity_str:
                        from datetime import datetime
                        last_activity = datetime.fromisoformat(last_activity_str)
                        inactive_duration = timezone.now().replace(tzinfo=None) - last_activity
                        
                        if inactive_duration > timedelta(hours=inactive_hours):
                            session.delete()
                            inactive_count += 1
                except Exception as e:
                    self.stdout.write(f"⚠️  Error processing session {session.session_key}: {e}")
            
            self.stdout.write(
                self.style.SUCCESS(f'✅ Deleted {inactive_count} inactive sessions (inactive for {inactive_hours} hours)')
            )
        
        self.stdout.write(self.style.SUCCESS('Session cleanup completed!'))
