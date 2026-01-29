import cv2
import requests
import json
import base64
import time
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk
import sys
import queue

class OllamaVisionCamera:
    def __init__(self):
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.vision_models = []
        
        # Camera configuration
        self.camera_index = 0
        self.camera = None
        
        # GUI variables
        self.root = None
        self.video_label = None
        self.text_display = None
        self.model_combobox = None
        self.running = False
        
        # Processing thread
        self.processing_queue = queue.Queue(maxsize=1)
        self.latest_description = "Initializing..."
        self.selected_model = ""
        
        # Find available vision models
        self.find_vision_models()
    
    def find_vision_models(self):
        """Scan for available vision models in Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                available_models = models_data.get("models", [])
                
                # List of known vision models (including Qwen3-VL-2B-Instruct)
                vision_model_keywords = [
                    'vision', 
                    'llava', 
                    'bakllava', 
                    'moondream', 
                    'cogvlm',
                    'qwen3-vl',  # For Qwen3-VL-2B-Instruct
                    'qwen-vl',   # For other Qwen vision models
                    'minicpm-v', # For MiniCPM vision models
                    'idefics',   # For Idefics models
                    'instructblip',  # For InstructBLIP models
                    'blip'       # For BLIP models
                ]
                
                for model in available_models:
                    model_name = model.get("name", "").lower()
                    # Check if model name contains vision-related keywords
                    if any(keyword in model_name for keyword in vision_model_keywords):
                        self.vision_models.append(model.get("name"))
                
                if not self.vision_models:
                    print("No vision models found. Please install a vision model like:")
                    print(" - llava")
                    print(" - bakllava")
                    print(" - moondream")
                    print(" - qwen3-vl-2b-instruct (for Qwen3-VL-2B-Instruct)")
                    print("\nExample commands:")
                    print("  ollama pull llava")
                    print("  ollama pull qwen3-vl-2b-instruct")
                    return False
                
                print(f"Found vision models: {self.vision_models}")
                self.selected_model = self.vision_models[0] if self.vision_models else ""
                return True
            else:
                print("Failed to connect to Ollama. Make sure it's running.")
                return False
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Make sure it's running on localhost:11434")
            return False
        except Exception as e:
            print(f"Error finding models: {e}")
            return False
    
    def find_camera(self):
        """Find available camera devices"""
        available_cameras = []
        
        # Test camera indices 0-10
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:  # Check if we can actually read a frame
                    available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            print(f"Available cameras: {available_cameras}")
            self.camera_index = available_cameras[0]
            return True
        else:
            print("No cameras found!")
            # Try to open a test camera anyway (might work on some systems)
            self.camera_index = 0
            return True
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
    
    def encode_frame_to_base64(self, frame):
        """Convert frame to base64 for Ollama API"""
        if frame is None:
            return None
        
        # Resize frame to reduce data size and processing time
        # Qwen3-VL-2B-Instruct works well with 448x448 resolution
        frame_resized = cv2.resize(frame, (448, 448))
        
        # Encode to JPEG then base64
        _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def get_model_prompt(self, model_name):
        """Get appropriate prompt for different vision models"""
        model_name_lower = model_name.lower()
        
        # Different prompts for different models
        if 'qwen' in model_name_lower:
            # Prompt optimized for Qwen vision models
            return "Describe what you see in this image in one concise sentence. Focus on objects, people, and actions."
        elif 'llava' in model_name_lower:
            return "Describe what you see in this image concisely in one sentence."
        elif 'moondream' in model_name_lower:
            return "What's in this image? Be brief."
        elif 'bakllava' in model_name_lower:
            return "Describe the visual content of this image in one sentence."
        elif 'cogvlm' in model_name_lower:
            return "Briefly describe what is shown in this image."
        else:
            # Generic prompt for other vision models
            return "Describe what you see in this image concisely in one sentence."
    
    def analyze_frame_with_ollama(self, image_base64):
        """Send frame to Ollama vision model for analysis"""
        if not self.selected_model and self.vision_models:
            self.selected_model = self.vision_models[0]
        
        if not self.selected_model:
            return "No vision model selected"
        
        try:
            # Get appropriate prompt for the selected model
            prompt = self.get_model_prompt(self.selected_model)
            
            # Prepare the payload
            payload = {
                "model": self.selected_model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent output
                    "num_predict": 100   # Limit response length
                }
            }
            
            # For Qwen3-VL-2B-Instruct, we might need different parameters
            if 'qwen3-vl' in self.selected_model.lower():
                payload["options"]["temperature"] = 0.2
                payload["options"]["top_p"] = 0.8
            
            response = requests.post(
                self.ollama_url, 
                json=payload, 
                timeout=10  # Reduced timeout for faster response
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No description generated")
                # Clean up the response
                response_text = response_text.strip()
                return response_text
            else:
                return f"Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Request timeout - try a smaller image or different model"
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama"
        except Exception as e:
            return f"Analysis error: {str(e)[:50]}"
    
    def process_frame_worker(self):
        """Worker thread for processing frames with Ollama"""
        processing_interval = 3  # Process every 3 seconds
        last_processed = 0
        
        while self.running:
            try:
                current_time = time.time()
                if current_time - last_processed >= processing_interval:
                    if not self.processing_queue.empty():
                        # Get the latest frame
                        frame = self.processing_queue.get_nowait()
                        
                        # Encode and analyze
                        image_base64 = self.encode_frame_to_base64(frame)
                        if image_base64 and self.selected_model:
                            description = self.analyze_frame_with_ollama(image_base64)
                            self.latest_description = description
                            
                            # Update GUI from main thread
                            if self.root:
                                self.root.after(0, self.update_description_display)
                        
                        last_processed = current_time
                
                time.sleep(0.1)  # Small sleep to prevent CPU hogging
                
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1)
    
    def update_description_display(self):
        """Update the text display with latest description"""
        if self.text_display:
            # Clear and insert new text
            self.text_display.delete(1.0, tk.END)
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            self.text_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
            
            # Add description
            self.text_display.insert(tk.END, f"{self.latest_description}\n\n", "description")
            
            # Auto-scroll to bottom
            self.text_display.see(tk.END)
    
    def update_model_selection(self, event=None):
        """Update the selected model when combobox changes"""
        self.selected_model = self.model_combobox.get()
        print(f"Model changed to: {self.selected_model}")
        
        # Clear previous descriptions
        if self.text_display:
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, f"Switched to: {self.selected_model}\n")
            self.text_display.insert(tk.END, "Processing new frame...\n")
    
    def update_video_display(self):
        """Update the video display in the GUI"""
        if not self.running:
            return
        
        # Capture frame
        frame = self.capture_frame()
        if frame is not None:
            # Add frame to processing queue (only keep latest)
            if self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.processing_queue.put_nowait(frame)
            except queue.Full:
                pass
            
            # Convert frame for Tkinter
            img = Image.fromarray(frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update video label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            # If frame capture fails, show error on video feed
            error_img = Image.new('RGB', (640, 480), color='black')
            imgtk = ImageTk.PhotoImage(image=error_img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule next update
        self.root.after(50, self.update_video_display)  # ~20 FPS
    
    def create_gui(self):
        """Create the Tkinter GUI with camera view and sidebar"""
        self.root = tk.Tk()
        self.root.title("Ollama Vision Camera with Qwen3-VL-2B-Instruct")
        self.root.geometry("1200x600")
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Video frame (left side)
        video_frame = tk.Frame(self.root, bg='black')
        video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(expand=True, fill='both')
        
        # Sidebar frame (right side)
        sidebar_frame = tk.Frame(self.root, bg='#f0f0f0')
        sidebar_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Title
        title_label = tk.Label(
            sidebar_frame, 
            text="🎯 Vision Description", 
            font=("Arial", 14, "bold"),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)
        
        # Model selection
        model_frame = tk.Frame(sidebar_frame, bg='#f0f0f0')
        model_frame.pack(pady=5, fill=tk.X, padx=10)
        
        model_label = tk.Label(
            model_frame,
            text="Select Model:",
            font=("Arial", 10),
            bg='#f0f0f0'
        )
        model_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Model dropdown
        self.model_combobox = ttk.Combobox(
            model_frame,
            values=self.vision_models,
            state="readonly",
            width=25
        )
        if self.vision_models:
            self.model_combobox.set(self.vision_models[0])
        self.model_combobox.pack(side=tk.LEFT)
        self.model_combobox.bind("<<ComboboxSelected>>", self.update_model_selection)
        
        # Description display
        desc_frame = tk.Frame(sidebar_frame, bg='#f0f0f0')
        desc_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Add a label for the description box
        desc_label = tk.Label(
            desc_frame,
            text="Real-time Analysis:",
            font=("Arial", 10, "bold"),
            bg='#f0f0f0'
        )
        desc_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text_display = scrolledtext.ScrolledText(
            desc_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            height=20,
            width=30,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.text_display.pack(expand=True, fill='both')
        
        # Configure text tags for formatting
        self.text_display.tag_config("timestamp", foreground="gray", font=("Arial", 9))
        self.text_display.tag_config("description", foreground="black", font=("Arial", 10))
        
        self.text_display.insert(tk.END, f"Model: {self.selected_model}\n")
        self.text_display.insert(tk.END, "="*30 + "\n")
        self.text_display.insert(tk.END, "Waiting for camera feed...\n")
        
        # Controls frame
        controls_frame = tk.Frame(sidebar_frame, bg='#f0f0f0')
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Button frame
        button_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        button_frame.pack(pady=5)
        
        # Stop button
        stop_button = tk.Button(
            button_frame,
            text="🛑 Stop",
            command=self.stop,
            bg='#ff6b6b',
            fg='white',
            font=("Arial", 10, "bold"),
            width=10,
            relief=tk.RAISED,
            padx=10
        )
        stop_button.pack(side=tk.LEFT, padx=5)
        
        # Restart button
        restart_button = tk.Button(
            button_frame,
            text="🔄 Restart",
            command=self.restart,
            bg='#4ecdc4',
            fg='white',
            font=("Arial", 10),
            width=10,
            relief=tk.RAISED,
            padx=10
        )
        restart_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            controls_frame,
            text="✓ Running",
            bg='#f0f0f0',
            font=("Arial", 9),
            fg='green'
        )
        self.status_label.pack(pady=5)
        
        # Instructions
        instructions = tk.Label(
            controls_frame,
            text="Model analyzes frames every 3 seconds",
            bg='#f0f0f0',
            font=("Arial", 8),
            fg='gray'
        )
        instructions.pack(pady=2)
    
    def restart(self):
        """Restart the camera"""
        if self.camera is not None:
            self.camera.release()
            time.sleep(0.5)
        
        self.camera = cv2.VideoCapture(self.camera_index)
        if self.camera.isOpened():
            print("Camera restarted")
            if self.status_label:
                self.status_label.config(text="✓ Running", fg='green')
    
    def start(self):
        """Start the camera and processing"""
        # First, check for vision models
        if not self.vision_models:
            print("Please install a vision model first.")
            print("\nRecommended commands:")
            print("  ollama pull qwen3-vl-2b-instruct")
            print("  ollama pull llava")
            return
        
        # Find and open camera
        if not self.find_camera():
            print("Warning: No camera detected, trying default...")
        
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            # Try a few more indices
            for i in range(1, 5):
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    self.camera_index = i
                    print(f"Found camera at index {i}")
                    break
        
        if not self.camera.isOpened():
            print("Cannot open any camera. Using test pattern.")
            # Create a dummy camera for testing
            self.camera = None
        
        # Create GUI
        self.create_gui()
        
        # Start processing thread
        self.running = True
        processing_thread = threading.Thread(target=self.process_frame_worker, daemon=True)
        processing_thread.start()
        
        # Start video update
        self.update_video_display()
        
        # Start GUI main loop
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the camera and clean up"""
        print("Stopping application...")
        self.running = False
        
        if self.camera is not None:
            self.camera.release()
        
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
        
        cv2.destroyAllWindows()
        print("Application stopped.")

def main():
    print("=" * 60)
    print("Ollama Vision Camera with Qwen3-VL-2B-Instruct Support")
    print("=" * 60)
    print("\nRequirements:")
    print("1. Ollama installed and running (ollama serve)")
    print("2. At least one vision model installed")
    print("3. Python packages: pip install opencv-python pillow requests")
    print("\nVision models recognized:")
    print("  - qwen3-vl-2b-instruct (Qwen3 Vision Language)")
    print("  - llava (LLaVA)")
    print("  - bakllava")
    print("  - moondream")
    print("  - cogvlm")
    print("  - and other vision models")
    print("=" * 60)
    
    app = OllamaVisionCamera()
    
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        app.stop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        app.stop()

if __name__ == "__main__":
    main()