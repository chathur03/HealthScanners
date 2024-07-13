# api/forms.py

from django import forms
from .models import Scan

class ScanForm(forms.ModelForm):
    class Meta:
        model = Scan
        fields = ['image']
