from django.db import models

class Scan(models.Model):
    image = models.ImageField(upload_to='scans/')
