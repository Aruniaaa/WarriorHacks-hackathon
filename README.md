# FocusAI 
> **Breaking down barriers to focused learning through AI-powered accountability**

## The Problem
Phone addiction is destroying student productivity. Research shows the average person checks their phone 96 times per day, with students being particularly vulnerable to social media distractions during study sessions. Traditional focus apps fail because they don't address the root issue: unconscious phone usage that breaks concentration before students even realize it's happening.

## The Solution
FocusAI is an AI accountability partner that creates an inclusive learning environment by removing digital distractions and providing comprehensive study tools in one platform. Using computer vision and intelligent tracking, it makes focused learning accessible to anyone struggling with phone addiction.

## How It Breaks Down Learning Barriers

- **Real-time Distraction Detection**: AI-powered computer vision identifies when phones enter the study space, helping students build awareness of unconscious habits
- **Inclusive Design**: Works with any webcam-equipped device, making it accessible regardless of technical background or expensive hardware
- **Comprehensive Learning Suite**: Combines focus tracking, task management, and AI-powered text summarization to eliminate the need for multiple apps that themselves become distractions
- **Personalized Insights**: Weekly statistics help students understand their patterns and celebrate progress, making self-improvement more achievable

## Key Features

### Smart Phone Detection
- **Custom AI Model**: Trained using Google's Teachable Machine for accurate phone detection
- **Intelligent Thresholds**: Requires 2 consecutive high-confidence detections over 2 seconds to avoid false positives
- **Non-Intrusive Tracking**: Counts interruptions without being overly aggressive or punitive

### Focus Timer
- Clean, distraction-free interface
- Automatic session tracking
- Seamless integration with phone detection system

### Task Management
- Intuitive to-do list with full CRUD operations
- Persistent storage across sessions
- Quick task completion tracking

### AI Text Summarization
- Powered by DistilBART model for efficient processing
- Intelligent text chunking for long documents
- Helps students quickly extract key information from study materials

### Aesthetic Analytics
- Weekly focus time breakdowns
- Phone interruption tracking
- Streak counting and goal completion metrics
- Modern, glassmorphism design with animated charts

## Installation & Setup

### Prerequisites
- Python 3.10
- Webcam (built-in or external)
- ~2GB free space for dependencies

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aruniaaa/WarriorHacks-hackathon
   cd WarriorHacks-hackathon
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

4. **Start the development server**
   ```bash
   python manage.py runserver
   ```

5. **Open your browser**

### Dependencies
```
django==4.2.0
opencv-python==4.8.1.78
tensorflow==2.13.0
keras==2.13.1
transformers==4.30.0
torch==2.0.0
pandas==2.2.2
numpy<=1.24.3
altair==5.3.0
```

## Usage

1. **Start a Focus Session**: Click the timer to begin your study session
2. **AI Monitoring**: The computer vision system automatically monitors for phone presence
3. **Stay Accountable**: View real-time interruption counts and focus statistics
4. **Manage Tasks**: Add and track your study tasks in the integrated to-do list
5. **Summarize Content**: Use the AI summarization tool to quickly process study materials
6. **Track Progress**: Review weekly statistics to understand and improve your focus habits

## Important Notes

**Local Execution Required**: Due to camera hardware requirements and AI model dependencies, this application is designed for local development. The computer vision features require direct webcam access that isn't available in containerized deployment environments.

**Demo Video**: A comprehensive demo video will be included in the final submission showing all features in action.

## Privacy & Security

- All computer vision processing happens locally
- No camera feed is stored or transmitted
- User data remains on local device
- Session-based user identification (no personal information required)

## Educational Impact

FocusAI addresses the growing crisis of digital distraction in education by:

- **Democratizing Focus**: Making advanced productivity tools accessible without expensive apps or subscriptions
- **Building Self-Awareness**: Helping students recognize unconscious phone usage patterns
- **Comprehensive Learning Support**: Providing study tools, focus tracking, and content processing in one platform
- **Promoting Healthy Habits**: Encouraging sustainable study practices rather than punitive restriction

## Architecture Highlights

- **Modular Design**: Separate concerns for CV processing, web interface, and data management
- **Scalable AI Pipeline**: Efficient model loading and inference optimization
- **Responsive UI**: Modern web design with real-time updates and smooth animations
- **Robust Error Handling**: Graceful degradation when hardware requirements aren't met

## Future Enhancements

- Mobile companion app for cross-device synchronization
- Advanced analytics with ML-powered study recommendations
- Integration with popular learning management systems
- Collaborative focus sessions for study groups

---

**Built for WarriorHacks 2025**  
*Theme: "Build a tool that breaks down barriers to learning, making education more inclusive, accessible, and impactful."*

**Technologies**: Python, Django, TensorFlow, OpenCV, HuggingFace Transformers, HTML/CSS/JavaScript
