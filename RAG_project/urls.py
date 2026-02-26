from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

admin.site.site_header = "RAG Based AI Tutor"
admin.site.site_title = "RAG AI Tutor Admin"
admin.site.index_title = "Administration Dashboard"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('tutor.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
