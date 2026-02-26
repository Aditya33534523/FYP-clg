from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('chat/', views.chat_view, name='chat'),
    path('api/ask/', views.ask_view, name='ask'),
    path('api/clear/', views.clear_history_view, name='clear_history'),
]
