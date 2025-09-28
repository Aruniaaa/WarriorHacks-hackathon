
from django.apps import AppConfig

class WebsiteConfig(AppConfig):
       default_auto_field = 'django.db.models.BigAutoField'
       name = 'website'
       
       def ready(self):
           from .views import initialize_phone_detection
           initialize_phone_detection()