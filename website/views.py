from django.shortcuts import render
from django.http import JsonResponse
from .models import FocusStats
from django.utils import timezone
from datetime import timedelta, datetime
from .cv_detection import phone_detector
import logging
import os
import json
import time
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .models import Tasks
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
import uuid
from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)


active_timers = {}

def landing_page(request):
    return render(request, "landing_page.html", {})

def timer(request):
    return render(request, "timer.html", {})

def summarization(request):
    return render(request, "summarization.html", {})

def contact(request):
    return render(request, "contact.html", {})


summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


logger = logging.getLogger(__name__)


try:
    summarizer = pipeline(
        "summarization", 
        model="sshleifer/distilbart-cnn-12-6",
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    logger.error(f"Failed to initialize summarization model: {e}")
    summarizer = None

def chunk_text(text, max_chunk_size=800):

    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:

        sentence_tokens = len(sentence) // 4
        
        if current_length + sentence_tokens > max_chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks

def summarize_text(text):

    if not summarizer:
        raise Exception("Summarization model not available")
    
    try:

        input_length = len(text)

        max_length = min(max(int(input_length * 0.3), 50), 512)
        min_length = min(max(int(input_length * 0.1), 30), max_length - 50)
        

        estimated_tokens = len(text) // 4
        
        if estimated_tokens > 1000:
            chunks = chunk_text(text, max_chunk_size=800)
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                chunk_max_length = min(max(len(chunk) // 6, 50), 200)
                chunk_min_length = min(max(len(chunk) // 12, 30), chunk_max_length - 20)
                
                chunk_summary = summarizer(
                    chunk,
                    max_length=chunk_max_length,
                    min_length=chunk_min_length,
                    do_sample=False,
                    truncation=True
                )
                chunk_summaries.append(chunk_summary[0]['summary_text'])
            

            if len(chunk_summaries) > 1:
                combined_summary = ' '.join(chunk_summaries)

                final_max_length = min(max(len(combined_summary) // 4, 100), 400)
                final_min_length = min(max(len(combined_summary) // 8, 50), final_max_length - 50)
                
                final_summary = summarizer(
                    combined_summary,
                    max_length=final_max_length,
                    min_length=final_min_length,
                    do_sample=False,
                    truncation=True
                )
                return final_summary[0]['summary_text']
            else:
                return chunk_summaries[0]
        else:

            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
            
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")

        if "index out of range" in str(e) or "sequence length" in str(e):
            raise Exception("Text is too long or complex for summarization. Please try with shorter text.")
        else:
            raise Exception(f"Summarization failed: {str(e)}")

@csrf_exempt
@require_http_methods(["POST"])
def summarize_text_api(request):

    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({'error': 'Text cannot be empty'}, status=400)
        
        if len(text) < 100:
            return JsonResponse({'error': 'Text must be at least 100 characters long'}, status=400)
        
        if len(text) > 50000:  # Increased limit but with chunking support
            return JsonResponse({'error': 'Text is too long. Maximum 50,000 characters allowed'}, status=400)
        
        # Check if summarizer is available
        if not summarizer:
            return JsonResponse({
                'error': 'Summarization service is currently unavailable. Please try again later.'
            }, status=503)
        

        try:
            logger.info(f"Starting summarization for text of length: {len(text)}")
            summary = summarize_text(text)
            logger.info(f"Summarization completed. Summary length: {len(summary)}")
            
            return JsonResponse({
                'success': True,
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': round((1 - len(summary) / len(text)) * 100, 1)
            })
            
        except Exception as e:
            logger.error(f"Summarization model error: {str(e)}")
            return JsonResponse({
                'error': str(e)
            }, status=503)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        logger.error(f"Summarization API error: {str(e)}")
        return JsonResponse({'error': 'Internal server error'}, status=500)


def format_time_display(total_minutes):
    if total_minutes < 60:
        return f"{int(total_minutes)}m"
    else:
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        if minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {minutes}m"


def get_weekly_breakdown(stats):

    total_minutes = stats.total_focus_time_week / 60 if stats.total_focus_time_week > 0 else 0

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_minutes = [30, 120, 360, 248, 480, 500, 67]

    today = timezone.now().date()
    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)

    # Uncomment and fix this section when you want real data:
    # for i, day in enumerate(days):
    #     current_day = monday + timedelta(days=i)
    #
    #     if current_day == today:
    #         # Use today's actual focus time
    #         daily_minutes.append(stats.total_focus_time_day / 60)  # Convert seconds to minutes
    #     elif current_day < today:
    #         # Past days - get from database
    #         daily_minutes.append(0)  # Replace with actual database query
    #     else:
    #         # Future days
    #         daily_minutes.append(0)

    max_minutes = max(daily_minutes) if max(daily_minutes) > 0 else 1

    daily_chart_data = []
    for day, minutes in zip(days, daily_minutes):
        height_percent = round((minutes / max_minutes * 100) if max_minutes > 0 else 0)
        daily_chart_data.append({
            'day': day,
            'minutes': round(minutes, 1),  # Changed from 'hours' to 'minutes'
            'height_percent': height_percent,
            'is_highlight': minutes == max_minutes
        })

    return daily_chart_data


def stats(request):
    stats, user_id = get_or_create_user(request)

    reset_if_needed(stats)

    daily_data_raw = get_weekly_breakdown(stats)


    total_focus_minutes = stats.total_focus_time_week / 60 if stats.total_focus_time_week > 0 else 0
    daily_avg_minutes = total_focus_minutes / 7 if total_focus_minutes > 0 else 0


    weekly_goal_minutes = 40 * 60
    goal_completion = min(100, (total_focus_minutes / weekly_goal_minutes) * 100) if total_focus_minutes > 0 else 0
    focus_percentage = 80 if stats.times_phone_stopped_week < 5 else max(60, 80 - (stats.times_phone_stopped_week * 2))


    daily_focus_minutes = [d['minutes'] for d in daily_data_raw]
    if total_focus_minutes > 0 and daily_focus_minutes:
        max_minutes = max(daily_focus_minutes)
        min_minutes = min(daily_focus_minutes)
        best_day_index = daily_focus_minutes.index(max_minutes)
        worst_day_index = daily_focus_minutes.index(min_minutes)
        best_day = daily_data_raw[best_day_index]['day']
        worst_day = daily_data_raw[worst_day_index]['day']
        best_day_minutes = max_minutes
        worst_day_minutes = min_minutes
    else:
        best_day = worst_day = "N/A"
        best_day_minutes = worst_day_minutes = 0


    daily_chart_data = [
        {
            'label': d['day'],
            'minutes': d['minutes'],
            'height_percent': d['height_percent'],
            'is_highlight': d['minutes'] == max(daily_focus_minutes)
        }
        for d in daily_data_raw
    ]


    longest_session_minutes = 150 if stats.total_sesh_week > 0 else 0
    improvement = 15 if stats.total_sesh_week > 0 else 0
    streak_days = sum(1 for m in daily_focus_minutes if m > 0)

    context = {
        'user_stats': {
            'daily_chart_data': daily_chart_data,
            'phone_detections': stats.times_phone_stopped_week,
            'phone_detections_change_text': f'↓ {improvement}% from last week',
            'best_day': best_day,
            'best_day_time': format_time_display(best_day_minutes),
            'worst_day': worst_day,
            'worst_day_time': format_time_display(worst_day_minutes),
            'current_streak': streak_days,
            'total_focus_time': 30,
            'daily_average_time': 4,
            'longest_session_time': format_time_display(longest_session_minutes),
            'goal_completion_percent': 67,
            'sessions_completed': stats.total_sesh_week,
            'weekly_change_percent': improvement,
            'focus_percent': focus_percentage,
            'distraction_percent': 100 - focus_percentage
        },
        'user_id': user_id
    }

    response = render(request, "stats.html", context)
    response.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 365)
    return response

def get_or_create_user(request):

    user_id = request.COOKIES.get("user_id")
    if not user_id:
        stats = FocusStats.objects.create()
        user_id = str(stats.user_id)
        logger.info(f"Created new user: {user_id}")
    else:
        try:
            stats = FocusStats.objects.get(user_id=user_id)
            logger.info(f"Found existing user: {user_id}")
        except FocusStats.DoesNotExist:
            stats = FocusStats.objects.create()
            user_id = str(stats.user_id)
            logger.info(f"Created new user (old ID not found): {user_id}")
    return stats, user_id


def initialize_phone_detection():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        'keras_model1.h5',
        os.path.join(current_dir, 'keras_model1.h5'),
        os.path.join('models', 'keras_model1.h5'),
        os.path.join('models', 'phone_detection_model.h5'),
    ]

    print("=== DEBUGGING MODEL PATHS ===")
    print(f"Current views.py directory: {current_dir}")
    for i, path in enumerate(possible_paths):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"Path {i+1}: {path}")
        print(f"  Absolute: {abs_path}")
        print(f"  Exists: {exists}")
        if exists:
            print(f"  ✓ FOUND MODEL AT: {path}")
    print("=============================")
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            success = phone_detector.load_model(model_path)
            if success:
                logger.info(f"Phone detection model initialized successfully from {model_path}")
                return
            else:
                logger.error(f"Failed to initialize phone detection model from {model_path}")
    
    logger.warning("Phone detection model not found in any expected location")

def start_timer(request):

    stats, user_id = get_or_create_user(request)
    

    reset_if_needed(stats)

    active_timers[user_id] = {
        'start_time': timezone.now(),
        'paused_time': None,
        'total_paused': 0,
        'stats': stats
    }

    detection_started = False
    if phone_detector.model is not None:
        detection_started = phone_detector.start_detection(user_id)
        if not detection_started:
            logger.warning(f"Failed to start phone detection for user {user_id}")
    else:
        logger.warning("Phone detection model not loaded - skipping CV detection")
    
    response = JsonResponse({
        "status": "started",
        "user_id": user_id,
        "phone_detection": detection_started,
        "message": "Timer started successfully"
    })
    response.set_cookie("user_id", user_id, max_age=60*60*24*365)
    return response

def pause_timer(request):

    stats, user_id = get_or_create_user(request)
    
    if user_id in active_timers:
        active_timers[user_id]['paused_time'] = timezone.now()
        phone_detector.stop_detection()
        return JsonResponse({"status": "paused"})
    
    return JsonResponse({"status": "error", "message": "No active timer"})

def resume_timer(request):

    stats, user_id = get_or_create_user(request)
    
    if user_id in active_timers and active_timers[user_id]['paused_time']:

        pause_duration = (timezone.now() - active_timers[user_id]['paused_time']).total_seconds()
        active_timers[user_id]['total_paused'] += pause_duration
        active_timers[user_id]['paused_time'] = None
        

        detection_restarted = False
        if phone_detector.model is not None:
            detection_restarted = phone_detector.start_detection(user_id)
        
        return JsonResponse({
            "status": "resumed",
            "phone_detection": detection_restarted
        })
    
    return JsonResponse({"status": "error", "message": "No paused timer"})

def stop_timer(request):

    stats, user_id = get_or_create_user(request)
    

    phone_detector.stop_detection()
    
    if user_id not in active_timers:
        return JsonResponse({"status": "error", "message": "No active timer"})
    
    timer_data = active_timers[user_id]

    end_time = timezone.now()
    total_duration = (end_time - timer_data['start_time']).total_seconds()
    

    if timer_data['paused_time']:

        pause_duration = (end_time - timer_data['paused_time']).total_seconds()
        timer_data['total_paused'] += pause_duration
    

    duration = total_duration - timer_data['total_paused']
    duration = max(0, duration)

    stats.total_focus_time_day += duration
    stats.total_focus_time_week += duration
    stats.total_sesh_day += 1
    stats.total_sesh_week += 1
    stats.save()
    
    logger.info(f"Focus session completed: {duration:.1f} seconds for user {user_id}")
    

    del active_timers[user_id]
    
    return JsonResponse({
        "status": "stopped",
        "duration": round(duration),
        "duration_minutes": round(duration / 60, 1),
        "duration_hours": round(duration / 3600, 2),
        "stats": {
            "sessions_today": stats.total_sesh_day,
            "focus_time_today": round(stats.total_focus_time_day),
            "focus_time_hours": round(stats.total_focus_time_day / 3600, 2),
            "phone_interruptions_today": stats.times_phone_stopped_day
        }
    })

@csrf_exempt
def phone_detected(request):

    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    print("🚨 PHONE DETECTED VIEW CALLED!")
    

    detection_user_id = phone_detector.user_id
    
    if not detection_user_id:
        print("❌ No user_id in phone detector!")
        return JsonResponse({"error": "No active user"}, status=400)
    
    try:

        stats = FocusStats.objects.get(user_id=detection_user_id)
        print(f"📊 Found stats for user {detection_user_id}")
    except FocusStats.DoesNotExist:
        print(f"❌ No stats found for user {detection_user_id}")
        return JsonResponse({"error": "User stats not found"}, status=404)
    

    reset_if_needed(stats)
    

    old_count = stats.times_phone_stopped_day
    stats.times_phone_stopped_day += 1
    stats.times_phone_stopped_week += 1
    stats.save()
    
    print(f"📊 Updated interruption count from {old_count} to {stats.times_phone_stopped_day}")
    
    logger.info(f"Phone detected for user {detection_user_id}. "
               f"Total interruptions today: {stats.times_phone_stopped_day}")
    
    return JsonResponse({
        "status": "phone_detected",
        "user_id": detection_user_id,
        "interruptions_today": stats.times_phone_stopped_day,
        "interruptions_week": stats.times_phone_stopped_week
    })

def get_detection_status(request):

    stats, user_id = get_or_create_user(request)
    

    elapsed = time.time() - phone_detector.start_time if hasattr(phone_detector, 'start_time') else 1
    frame_rate = phone_detector.frame_count / elapsed if hasattr(phone_detector, 'frame_count') and elapsed > 0 else 0

    print(f"Detection status - interruptions_today: {stats.times_phone_stopped_day}")
    print(f"Phone confidence: {getattr(phone_detector, 'last_confidence_phone', 0)}")
    
    response_data = {
        "detection_running": phone_detector.is_running,
        "model_loaded": phone_detector.model is not None,
        "current_user": phone_detector.user_id,
        "timer_active": user_id in active_timers,
        "confidence": {
            "phone": getattr(phone_detector, 'last_confidence_phone', 0),
            "no_phone": getattr(phone_detector, 'last_confidence_no_phone', 0)
        },
        "frame_rate": round(frame_rate, 1),
        "last_detection": "Just now" if getattr(phone_detector, 'last_detection_time', 0) > time.time() - 5 else "None",
        "stats": {
            "interruptions_today": stats.times_phone_stopped_day,
            "sessions_today": stats.total_sesh_day,
            "focus_time_today": round(stats.total_focus_time_day),
            "focus_time_hours": round(stats.total_focus_time_day / 3600, 2)
        }
    }
    
    print(f"Sending response: {response_data}")
    return JsonResponse(response_data)

def reset_if_needed(stats: FocusStats):

    today = timezone.now().date()
    

    if stats.last_day_reset != today:
        logger.info(f"Resetting daily stats for user {stats.user_id}")
        stats.total_sesh_day = 0
        stats.total_focus_time_day = 0
        stats.times_phone_stopped_day = 0
        stats.last_day_reset = today
        stats.save()
    

    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)
    
    if stats.last_week_reset < monday:
        logger.info(f"Resetting weekly stats for user {stats.user_id}")
        stats.total_sesh_week = 0
        stats.total_focus_time_week = 0
        stats.times_phone_stopped_week = 0
        stats.last_week_reset = today
        stats.save()



def to_do(request):

    if 'user_id' not in request.session:
        request.session['user_id'] = str(uuid.uuid4())
    
    user_id = request.session['user_id']
    tasks = Tasks.objects.filter(user_id=user_id).order_by('-created_at')
    
    return render(request, "to-do.html", {'tasks': tasks})

@csrf_exempt
@require_http_methods(["POST"])
def add_task(request):

    try:
        data = json.loads(request.body)
        task_text = data.get('task', '').strip()
        
        if not task_text:
            return JsonResponse({'error': 'Task cannot be empty'}, status=400)
        
        if 'user_id' not in request.session:
            request.session['user_id'] = str(uuid.uuid4())
        
        user_id = request.session['user_id']
        
        task = Tasks.objects.create(
            user_id=user_id,
            task=task_text
        )
        
        return JsonResponse({
            'success': True,
            'task': {
                'id': task.id,
                'task': task.task,
                'completed': task.completed,
                'created_at': task.created_at.isoformat()
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def toggle_task(request, task_id):

    try:
        if 'user_id' not in request.session:
            return JsonResponse({'error': 'User not found'}, status=404)
        
        user_id = request.session['user_id']
        task = get_object_or_404(Tasks, id=task_id, user_id=user_id)
        
        task.completed = not task.completed
        task.save()
        
        return JsonResponse({
            'success': True,
            'completed': task.completed
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["PUT"])
def edit_task(request, task_id):

    try:
        data = json.loads(request.body)
        new_task_text = data.get('task', '').strip()
        
        if not new_task_text:
            return JsonResponse({'error': 'Task cannot be empty'}, status=400)
        
        if 'user_id' not in request.session:
            return JsonResponse({'error': 'User not found'}, status=404)
        
        user_id = request.session['user_id']
        task = get_object_or_404(Tasks, id=task_id, user_id=user_id)
        
        task.task = new_task_text
        task.save()
        
        return JsonResponse({
            'success': True,
            'task': task.task
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_task(request, task_id):

    try:
        if 'user_id' not in request.session:
            return JsonResponse({'error': 'User not found'}, status=404)
        
        user_id = request.session['user_id']
        task = get_object_or_404(Tasks, id=task_id, user_id=user_id)
        
        task.delete()
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_completed_tasks(request):

    try:
        if 'user_id' not in request.session:
            return JsonResponse({'error': 'User not found'}, status=404)
        
        user_id = request.session['user_id']
        

        completed_tasks = Tasks.objects.filter(user_id=user_id, completed=True)
        count = completed_tasks.count()
        
        if count == 0:
            return JsonResponse({'message': 'No completed tasks to delete', 'count': 0})
        

        completed_tasks.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Deleted {count} completed task{"s" if count != 1 else ""}',
            'count': count
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)