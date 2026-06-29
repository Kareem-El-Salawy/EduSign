# EduSign

EduSign is a local web-based platform designed to support communication and learning through American Sign Language (ASL).

The system combines sign recognition, text-to-sign translation, and an educational sign-video library in one FastAPI application.

This project was developed as a graduation project at **Misr International University (MIU)**.

---

## Main Features

EduSign contains three main subsystems:

### 1. Sign-to-Text

The Sign-to-Text module uses a webcam to recognize sign-language gestures and convert them into written text.

Main technologies:

* OpenCV for webcam capture
* MediaPipe for hand landmark detection
* TensorFlow/Keras for sign classification
* Bidirectional LSTM-based recognition model
* Offline text-to-speech output
* Live browser camera preview

The recognition model processes sequences of hand landmarks rather than individual images.

---

### 2. Text-to-Sign

The Text-to-Sign module converts written English input into a sequence of sign-language video clips.

The workflow includes:

1. Receiving a sentence from the user
2. Cleaning and processing the text
3. Converting the sentence into supported sign glosses
4. Matching glosses with available ASL videos
5. Playing the matched clips as a sign sequence
6. Reporting unsupported or missing words

The module supports:

* Direct database matching
* Rule-based fallback processing
* Optional Gemini-assisted text-to-gloss conversion
* WLASL video integration
* Missing-word reporting
* Sequential sign-video playback

The Gemini service is optional. If it is unavailable or its quota is exceeded, the system continues using the local rule-based fallback.

---

### 3. ASL E-Library

The E-Library allows users to search for individual ASL signs and view matching videos.

Features include:

* Searching by English word or gloss
* Fuzzy search support
* WLASL dataset integration
* Multiple video results where available
* Direct video playback in the browser

---

## System Architecture

EduSign uses a modular FastAPI architecture.

```text
User Interface
      │
      ▼
FastAPI Web Application
      │
      ├── Sign-to-Text Router
      │       ├── Webcam capture
      │       ├── MediaPipe landmarks
      │       └── TensorFlow model
      │
      ├── Text-to-Sign Router
      │       ├── Text processing
      │       ├── Gloss matching
      │       ├── Gemini integration
      │       └── Video sequencing
      │
      └── Video Lookup Router
              ├── WLASL index
              └── Video search
```

---

## Project Structure

```text
EduSign/
│
├── app/
│   ├── main.py
│   │
│   ├── core/
│   │   ├── runtime.py
│   │   ├── sign_features.py
│   │   └── video_index.py
│   │
│   └── routers/
│       ├── pages.py
│       ├── sign_to_text.py
│       ├── text_to_sign.py
│       └── video_lookup.py
│
├── static/
│   ├── sign_to_text.html
│   ├── text_to_sign.html
│   ├── video_lookup.html
│   │
│   ├── css/
│   │   ├── app.css
│   │   ├── sign_to_text.css
│   │   ├── text_to_sign.css
│   │   └── video_lookup.css
│   │
│   └── videos/
│
├── data/
│   ├── gloss_db.json
│   └── wlasl_tts_map.json
│
├── archive/
│   ├── WLASL_v0.3.json
│   └── videos/
│
├── scripts/
│
├── model.keras
├── hand_landmarker.task
├── label_map.json
├── gloss_db.txt
├── config.json
├── reqs.txt
└── README.md
```

Some file names may vary depending on the installed project version.

---

## Requirements

Recommended environment:

* Windows 10 or Windows 11
* Python 3.9 or a compatible version
* Webcam
* 8 GB RAM or more
* Modern web browser
* Internet connection only when using Gemini

The main application can operate locally without Gemini by using the rule-based fallback.

---

## Installation

### 1. Clone the repository

Using Git:

```bash
git clone https://github.com/YOUR-USERNAME/EduSign.git
cd EduSign
```

Alternatively, download the project as a ZIP file and extract it.

---

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### 3. Install the dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r reqs.txt
```

---

## Dataset and Video Files

The WLASL video folder may not be included in the GitHub repository because of its large size.

Place the required videos inside:

```text
archive/videos/
```

The WLASL metadata file should be placed at:

```text
archive/WLASL_v0.3.json
```

Text-to-Sign mappings are stored in:

```text
data/wlasl_tts_map.json
```

Without the video files, the website may load normally, but sign videos will not play.

---

## Gemini Configuration

Gemini is used as an optional text-to-gloss conversion method.

Create a local `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not upload the `.env` file or API key to GitHub.

The application will use its local rule-based fallback if:

* No Gemini key is configured
* The Gemini request fails
* The API quota is exceeded
* The internet connection is unavailable

---

## Running the Application

Activate the virtual environment, then run:

```powershell
python -m uvicorn app.main:app --reload
```

Open the following address in your browser:

```text
http://127.0.0.1:8000
```

To stop the server, press:

```text
Ctrl + C
```

---

## Main Routes

Typical application routes include:

```text
/                       Home page
/sign-to-text           Sign-to-Text module
/text-to-sign           Text-to-Sign module
/video-lookup           ASL E-Library
```

API route names may vary depending on the project version.

---

## Sign-to-Text Model

The Sign-to-Text subsystem uses hand-landmark sequences extracted through MediaPipe.

General processing workflow:

```text
Webcam frame
    ↓
MediaPipe hand detection
    ↓
21 landmarks per hand
    ↓
Sequence construction
    ↓
Feature normalization
    ↓
TensorFlow classification model
    ↓
Predicted sign gloss
    ↓
Sentence output
```

The system supports up to two hands.

Each hand contains 21 landmarks with three coordinates:

```text
x, y, z
```

For two hands, one frame can contain:

```text
2 × 21 × 3 = 126 features
```

The recognition model processes multiple consecutive frames to recognize dynamic signs.

---

## Text-to-Sign Workflow

```text
User sentence
    ↓
Text cleaning
    ↓
Local phrase and gloss matching
    ↓
Optional Gemini conversion
    ↓
Rule-based fallback
    ↓
Video availability check
    ↓
Ordered ASL video sequence
```

The output displays:

* Matched glosses
* Sign videos
* Current playing sign
* Missing or unsupported words
* Translation method used

---

## Technologies Used

* Python
* FastAPI
* Uvicorn
* HTML
* CSS
* JavaScript
* OpenCV
* MediaPipe
* TensorFlow
* Keras
* NumPy
* RapidFuzz
* NLTK
* Gemini API
* WLASL dataset

---

## Known Limitations

* Recognition accuracy depends on lighting, camera position, and signing speed.
* The recognition model supports only signs included in its trained classes.
* Text-to-Sign output is limited to available sign videos.
* Some English words may not have a matching video.
* Concatenated sign clips do not fully reproduce natural ASL grammar.
* Different source videos may contain different signers or recording conditions.
* Gemini requests depend on internet availability and API quota.
* The current system is intended as an educational prototype.

---

## Future Improvements

Possible future developments include:

* Larger sign-recognition dataset
* Improved continuous sign-language recognition
* Sentence-level ASL grammar processing
* More consistent Text-to-Sign video signers
* Sign-language avatar generation
* Facial-expression and body-pose analysis
* Additional sign languages
* Improved mobile responsiveness
* User accounts and learning progress tracking
* Cloud deployment

---

## Privacy

The application is designed to run locally.

Webcam frames are processed by the local application and are not intentionally stored unless a screenshot or recording feature is explicitly used.

When Gemini is enabled, submitted text may be sent to the configured Gemini API for processing.

---

## Academic Purpose

EduSign was developed as an academic graduation project.

The system is a prototype and should not be treated as a certified accessibility, medical, educational, or professional interpretation service.

---

## Team

Developed by the EduSign graduation project team at:

**Misr International University — Faculty of Engineering**

Team members:

*Ziad Anwar
*Kareem Moomen
*Ahamd Reda

Supervisor:

*Prof. Medhat Awadallah 

---

## License

This project is currently provided for academic and educational purposes.

A formal open-source license may be added later.
