<!-- Badges Section -->
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/unity-2022%2B-black" alt="Unity Version">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status">
  <img src="https://img.shields.io/badge/AI-Google%20Gemini-orange" alt="Google Gemini">
</p>


- Google drive link with unity prototype game files   - https://drive.google.com/file/d/17cXMqM8OWzklj_dTKQJ-P-MVApiHzuqm/view?usp=sharing
- Google drive link of prototype demo video -  https://drive.google.com/file/d/1IA-tzZjuHzG1E7uOvja-BRXAyR-h_TKh/view?usp=sharing
  
## Table of Contents
- [About the Project](#about-the-project)
- [Who Is It For](#who-is-it-for)
- [What Makes It Different](#what-makes-it-different)
- [Highlights](#highlights)
- [How It Works](#how-it-works)
- [User Experience](#user-experience)
- [Demo Game Flow](#demo-game-flow)
- [Feasibility](#feasibility)
- [Challenges & Risks](#challenges--risks)
- [How to Run Our Game](#how-to-run-our-game)
- [File Structure](#file-structure)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)


---

## About the Project
Public Speaking game is a **gamified training program** that blends **AI-powered feedback, immersive environments, and mini-games** to help individuals improve their public speaking and communication skills through high pressure test scenarios based on real life along with mini games and fun activities.  

---

## Who Is It For
This project is designed for:  
- Young professionals entering the job market  
- Students preparing for interviews and networking  
- Anyone looking to build confidence in high-pressure communication  

---

## What Makes It Different
Unlike traditional workshops and online courses, this project:  
- Adapts to individual needs  
- Provides **real-time AI feedback**  
- Gamifies the learning process with engaging scenarios

---

## Highlights
- **LangGraph GenAI Agents** create adaptable and dynamic high pressure scenarios to test users
- **AI evaluation, scoring and feedback** provide measurable results and areas of improvement to users.
- Evaluates **Voice clarity,tone,Posture and body language** for comprehensive improvement.
- Measurable results. Based on local tests conducted with a focus group of 10, an average self-assessed **15% improvement** was seen.

---
## User Experience
- Built in **Unity Game Engine**  
- Immersive environments: **Office, Presentation Hall, Outdoor**  
- Interactive NPCs with unique personalities & memory  
- Mini-games to improve English skills (e.g., Hangman)


---
## How It Works
1. Audio input from user → transcribed with **OpenAI Whisper**  
2. Camera & posture analysis with **MediaPipe**  
3. Text + body data → processed by **Google Gemini (LLM)**  
4. LLM replies in character → audio generated via **Coqui-TTS**  
5. Evaluation node checks responses → updates pass/fail meter  
6. Conversation continues until time limit → final evaluation  

---


## Prototype Game Flow

```
Start 
   ↓
Posture Game 
   ↓
Office Games (Pronunciation, Convincing, Hangman, NPC Interactions) 
   ↓
Evaluation 
   ↓
Convince the Boss 
   ↓
Final Presentation 
   ↓
End
```
---

## Feasibility
- **Technically Feasible**: Uses mature open-source AI tools  
- **Cost-effective**: Free tiers & open-source options  
- **Market Ready**: Aligned with EdTech gamification demand  

---

## Challenges & Risks
- Privacy concerns with voice/video data  
- AI accuracy for diverse accents  
- High dev time for branching storylines  
- Dependency on large local STT and TTS models 

---

## Scalability
- Currently for he prototype backend is running locally, which can be shifted to a cloud service for easy adoptability for low end systems
- The Game can be expanded by adding more unique and diverse minigames
- More powerful Speech-to-Text and Text-to-Speech  systems can be used for tolerance with wider range of accents and multiple voices.
  
---

## How to Run Our Prototype

1. **Download the Code**
   - Go to the GitHub repository and download the project as a `.zip` file.
   - Unzip the file and place it in a folder of your choice.

2. **Create a Virtual Environment**
   - On **Windows**:
     ```bash
     python -m venv .venv
     ```
   - On **Mac/Linux**:
     ```bash
     python3 -m venv .venv
     ```

3. **Activate the Virtual Environment**
   - On **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - On **Mac/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**
   - Run the following command:
     ```bash
     pip install -r requirements.txt
     ```

5. **Get a Google Gemini API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/) and click **Get API Key**.
   - Follow the instructions to generate a free API key.

6. **Create a `.env` File**
   - Inside the game folder, create a new file named `.env`.
   - Paste your API key in the following format:
     ```
     GOOGLE_API_KEY="YOUR_API_KEY_HERE"
     ```

7. **Install OpenAI Whisper**
   - Download and install from the official repository:  
     [OpenAI Whisper GitHub Repo](https://github.com/openai/whisper)

8. **Start the Backend Server**
   - On **Windows/Mac/Linux**:
     ```bash
     uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
     ```

9. **Run the Prototype Logic**
   - On **Windows/Mac/Linux**:
     ```bash
     python nre.py
     ```

10. **Launch the Game Prototype**
    - On **Windows/Mac/Linux**:
      - Navigate to the `Unity Build` folder.
      - Double-click the game executable (`.exe` on Windows, `.app` on Mac, or Linux build file).
      - Enjoy your fun learning experience!

---

## File Structure
```
├── .gitignore
├── audio_utils.py
├── backend.py
├── camera_utils.py
├── conv_agent.py
├── convince_boss.py
├── nre.py
├── public_speaking.py
└── requirements.txt
```

---

## Tech Stack
- **Game Engine**: Unity  
- **LLM**: Google Gemini  
- **Speech-to-Text**: OpenAI Whisper  
- **TTS**: Coqui-TTS  
- **Video Analysis**: MediaPipe  
- **Backend**: FastAPI (Uvicorn)  
- **Language**: Python




