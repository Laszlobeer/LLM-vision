


---
# 🧠📷 Ollama Vision Camera

[![Ollama](https://img.shields.io/badge/Ollama-AI-blue?logo=ollama)](https://ollama.com/)
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green)](https://docs.python.org/3/library/tkinter.html)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-yellow)](https://python-pillow.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)


A real-time camera application that uses **Ollama vision models** (like **Qwen3-VL-2B-Instruct**, **LLaVA**, **Moondream**, etc.) to analyze webcam frames and generate concise visual descriptions — all through a clean **Tkinter GUI**.

---

## ✨ Features

* 🎥 Live webcam feed
* 🤖 Real-time image analysis using Ollama vision models
* 🔁 Automatic model detection from local Ollama installation
* 🧠 Model-specific prompts for better descriptions
* 🪟 Simple Tkinter GUI with:

  * Live video preview
  * Model selector
  * Timestamped analysis output
* ⚡ Background processing thread (non-blocking UI)
* ⏱️ Frame analysis every 3 seconds (configurable)

---

## 📸 Supported Vision Models

The app automatically detects installed Ollama vision models, including:

* **Qwen3-VL-2B-Instruct**
* **LLaVA**
* **BakLLaVA**
* **Moondream**
* **CogVLM**
* **MiniCPM-V**
* **BLIP / InstructBLIP**
* Other models containing vision-related keywords

---

## 🧩 Requirements

### System

* Python **3.9+**
* A working webcam
* Ollama running locally

### Python Packages

Install dependencies with:

```bash
pip install opencv-python pillow requests
```

### Ollama

Make sure Ollama is installed and running:

```bash
ollama serve
```

Pull at least one vision model, for example:

```bash
ollama pull qwen3-vl-2b-instruct
```

or

```bash
ollama pull llava
```

---

## 🚀 How to Run

1. Clone or download this repository
2. Start Ollama:

   ```bash
   ollama serve
   ```
3. Run the program:

   ```bash
   python main.py
   ```
4. Select a vision model from the dropdown
5. Watch the live camera feed and real-time descriptions

---

## 🖥️ User Interface Overview

* **Left panel:** Live webcam feed
* **Right panel:**

  * Vision model selector
  * Scrollable real-time analysis output
  * Stop / Restart controls
* **Status indicator:** Shows running state
* **Processing interval:** One analysis every 3 seconds

---

## ⚙️ How It Works

1. Captures webcam frames using OpenCV
2. Resizes frames to **448×448** for optimal vision model performance
3. Encodes frames as Base64 JPEGs
4. Sends frames to the Ollama `/api/generate` endpoint
5. Displays model responses in real time

---

## 🛠️ Configuration Notes

* **Processing interval:**
  Adjust in `process_frame_worker()`:

  ```python
  processing_interval = 3
  ```

* **Camera index:**
  Automatically detected (fallbacks included)

* **Temperature & generation limits:**
  Tuned per model for concise, stable output

---

## 🧯 Troubleshooting

**No camera detected**

* Make sure no other app is using the webcam
* Try changing the camera index

**No vision models found**

* Run:

  ```bash
  ollama list
  ```
* Pull a vision model if none are installed

**Connection error**

* Ensure Ollama is running on `localhost:11434`

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

## ❤️ Credits

Built with:

* OpenCV
* Tkinter
* Pillow
* Ollama Vision Models

---

If you want, I can also:

* Add **screenshots**
* Write a **short project description for GitHub**
* Create a **requirements.txt**
* Refactor this into a **pip-installable package**

Just say the word 🚀
