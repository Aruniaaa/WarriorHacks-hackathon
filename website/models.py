from django.db import models
from django.utils import timezone
import uuid

def get_current_date():
    """Helper function to get current date (not datetime)"""
    return timezone.now().date()

class FocusStats(models.Model):
    user_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    # daily stats
    total_sesh_day = models.IntegerField(default=0)
    total_focus_time_day = models.FloatField(default=0.0)  # in seconds
    times_phone_stopped_day = models.IntegerField(default=0)

    # weekly stats
    total_sesh_week = models.IntegerField(default=0)
    total_focus_time_week = models.FloatField(default=0.0)  # in seconds
    times_phone_stopped_week = models.IntegerField(default=0)

    last_day_reset = models.DateField(default=get_current_date)
    last_week_reset = models.DateField(default=get_current_date)

    def __str__(self):
        return f"{self.user_id}'s Focus Stats"
    
class DailyFocusHistory(models.Model):
    """Store daily focus history for each user"""
    user_id = models.UUIDField()
    date = models.DateField()
    total_focus_time = models.FloatField(default=0.0)  # in seconds
    total_sessions = models.IntegerField(default=0)
    phone_detections = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user_id', 'date']  # One record per user per day
        
    def __str__(self):
        return f"{self.user_id} - {self.date}: {self.total_focus_time/60:.1f}min"

class Tasks(models.Model):
    user_id = models.UUIDField(default=uuid.uuid4, editable=False)
    
    # Remove the separate id field since Django auto-creates one
    task = models.CharField(max_length=50)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Task: {self.task} for {self.user_id}"