from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'placeholder': 'you@example.com', 'class': 'form-input'})
    )
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'placeholder': 'Min 8 chars, upper, lower, digit, symbol', 'class': 'form-input'})
    )
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={'placeholder': 'Repeat password', 'class': 'form-input'})
    )

    class Meta:
        model = User
        fields = ('email', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = self.cleaned_data['email']
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class LoginForm(AuthenticationForm):
    username = forms.EmailField(
        label='Email',
        widget=forms.EmailInput(attrs={'placeholder': 'you@example.com', 'class': 'form-input', 'autofocus': True})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Your password', 'class': 'form-input'})
    )


class QuestionForm(forms.Form):
    MARKS_CHOICES = [
        (1, '🔹 Short (1 Mark)'),
        (2, '🔷 Medium (2 Marks)'),
        (5, '🔵 Detailed (5 Marks)'),
    ]
    question = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'e.g. What is backpropagation in neural networks?', 'class': 'chat-input', 'autocomplete': 'off'})
    )
    marks = forms.ChoiceField(choices=MARKS_CHOICES, initial=1)
