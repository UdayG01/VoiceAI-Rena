import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
import time
from pathlib import Path
from loguru import logger

# Import the recorder class
try:
    from realtime_meeting_recorder import RealtimeMeetingRecorder
except ImportError:
    print("Error: Make sure 'realtime_meeting_recorder.py' is in the same directory!")
    import sys
    sys.exit(1)

logger.remove()


class MeetingRecorderGUI:
    """
    GUI application for real-time meeting recording and note generation.
    Simple start/stop interface - no file uploads needed!
    """
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🎤 Real-Time Meeting Notes Generator")
        self.window.geometry("600x500")
        self.window.configure(bg="#f0f2f5")
        
        # Recorder instance
        self.recorder = RealtimeMeetingRecorder()
        self.is_recording = False
        self.recording_thread = None
        self.start_time = None
        
        self.setup_ui()
        self.update_timer()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Title
        title_frame = tk.Frame(self.window, bg="#667eea", height=80)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎤 Meeting Notes Generator",
            font=("Arial", 24, "bold"),
            bg="#667eea",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.window, bg="#f0f2f5")
        main_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Status indicator
        self.status_frame = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
        self.status_frame.pack(fill="x", pady=(0, 20))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="⚪ Ready to record",
            font=("Arial", 14),
            bg="white",
            fg="#666",
            pady=15
        )
        self.status_label.pack()
        
        # Timer display
        self.timer_label = tk.Label(
            self.status_frame,
            text="00:00:00",
            font=("Arial", 32, "bold"),
            bg="white",
            fg="#667eea"
        )
        self.timer_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="1. Click START before joining your meeting\n"
                 "2. Join Google Meet normally\n"
                 "3. Click STOP when meeting ends\n"
                 "4. Get instant PDF notes!",
            font=("Arial", 11),
            bg="#f0f2f5",
            fg="#555",
            justify="left"
        )
        instructions.pack(pady=(0, 20))
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg="#f0f2f5")
        buttons_frame.pack(pady=10)
        
        # Start button
        self.start_button = tk.Button(
            buttons_frame,
            text="▶ START RECORDING",
            font=("Arial", 14, "bold"),
            bg="#28a745",
            fg="white",
            activebackground="#218838",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            width=20,
            height=2,
            command=self.start_recording
        )
        self.start_button.pack(side="left", padx=10)
        
        # Stop button
        self.stop_button = tk.Button(
            buttons_frame,
            text="⏹ STOP & GENERATE",
            font=("Arial", 14, "bold"),
            bg="#dc3545",
            fg="white",
            activebackground="#c82333",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            width=20,
            height=2,
            command=self.stop_recording,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10)
        
        # Device selection
        device_frame = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
        device_frame.pack(fill="x", pady=(20, 0))
        
        tk.Label(
            device_frame,
            text="Audio Device:",
            font=("Arial", 10),
            bg="white"
        ).pack(side="left", padx=10, pady=10)
        
        self.device_var = tk.StringVar(value="Default")
        self.device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.device_var,
            state="readonly",
            width=40
        )
        self.device_combo.pack(side="left", padx=10, pady=10)
        
        # Load devices
        self.load_devices()
        
        # Refresh button
        refresh_btn = tk.Button(
            device_frame,
            text="🔄",
            command=self.load_devices,
            relief="flat",
            cursor="hand2",
            bg="white"
        )
        refresh_btn.pack(side="left", padx=5)
        
        # Output info
        self.output_label = tk.Label(
            main_frame,
            text="📁 Notes will be saved to: meeting_outputs/",
            font=("Arial", 9),
            bg="#f0f2f5",
            fg="#888"
        )
        self.output_label.pack(side="bottom", pady=10)
    
    def load_devices(self):
        """Load available audio devices"""
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            devices = ["Default (System Audio)"]
            self.device_indices = [None]
            
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append(f"[{i}] {info['name']}")
                    self.device_indices.append(i)
            
            audio.terminate()
            self.device_combo['values'] = devices
            
        except Exception as e:
            self.device_combo['values'] = ["Error loading devices"]
            print(f"Error loading devices: {e}")
    
    def get_selected_device_index(self):
        """Get the selected device index"""
        selection = self.device_combo.current()
        if selection >= 0 and selection < len(self.device_indices):
            return self.device_indices[selection]
        return None
    
    def start_recording(self):
        """Start recording the meeting"""
        if self.is_recording:
            return
        
        device_index = self.get_selected_device_index()
        
        try:
            # Start recorder in background thread
            self.recording_thread = threading.Thread(
                target=self._start_recording_thread,
                args=(device_index,),
                daemon=True
            )
            self.recording_thread.start()
            
            # Update UI
            self.is_recording = True
            self.start_time = time.time()
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.device_combo.config(state="disabled")
            self.status_label.config(text="🔴 Recording in progress...", fg="#dc3545")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording:\n{e}")
    
    def _start_recording_thread(self, device_index):
        """Background thread for recording"""
        try:
            self.recorder.start_recording(device_index)
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.window.after(0, self.reset_ui)
    
    def stop_recording(self):
        """Stop recording and generate notes"""
        if not self.is_recording:
            return
        
        self.status_label.config(text="⏳ Generating notes...", fg="#ffc107")
        self.stop_button.config(state="disabled")
        
        # Stop in background thread
        threading.Thread(target=self._stop_recording_thread, daemon=True).start()
    
    def _stop_recording_thread(self):
        """Background thread for stopping and processing"""
        try:
            result = self.recorder.stop_recording()
            
            if result:
                # Show success message
                self.window.after(0, lambda: self.show_success(result))
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.window.after(0, self.reset_ui)
    
    def show_success(self, result):
        """Show success message with results"""
        message = f"""✅ Meeting notes generated successfully!

📄 PDF saved to: {result['pdf_path']}

📝 Summary:
{result['summary'][:200]}...

Would you like to open the output folder?"""
        
        if messagebox.askyesno("Success!", message):
            import os
            import platform
            output_dir = str(Path(result['pdf_path']).parent)
            
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open '{output_dir}'")
            else:  # Linux
                os.system(f"xdg-open '{output_dir}'")
    
    def reset_ui(self):
        """Reset UI to initial state"""
        self.is_recording = False
        self.start_time = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.device_combo.config(state="readonly")
        self.status_label.config(text="⚪ Ready to record", fg="#666")
        self.timer_label.config(text="00:00:00")
    
    def update_timer(self):
        """Update the recording timer"""
        if self.is_recording and self.start_time:
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Schedule next update
        self.window.after(1000, self.update_timer)
    
    def run(self):
        """Start the GUI application"""
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_recording:
            if messagebox.askokcancel("Quit", "Recording in progress. Stop and quit?"):
                self.recorder.stop_recording()
                self.window.destroy()
        else:
            self.window.destroy()


if __name__ == "__main__":
    print("🚀 Starting Meeting Notes Generator GUI...")
    print("💡 Tip: Make sure Ollama is running (ollama serve)")
    
    app = MeetingRecorderGUI()
    app.run()
