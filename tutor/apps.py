from django.apps import AppConfig
from django.contrib.admin import AdminSite


class RAGAdminSite(AdminSite):
    site_header = "RAG Based AI Tutor"
    site_title = "RAG AI Tutor Admin"
    index_title = "Administration Dashboard"


rag_admin_site = RAGAdminSite(name='rag_admin')


class TutorAdminConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tutor'
    verbose_name = 'RAG AI Tutor'

    def ready(self):
        pass
