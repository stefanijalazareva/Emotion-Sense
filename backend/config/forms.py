from django.contrib.auth.forms import AuthenticationForm, UserCreationForm


INPUT_CLASSES = (
    'mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm '
    'focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500'
)


class UserLoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = INPUT_CLASSES


class UserRegistrationForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = INPUT_CLASSES

