import pyaudio
import wave
import threading
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
from meeting_notes_generator import MeetingNotesGenerator

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


class RealtimeMeetingRecorder:
    """
    Records meeting audio in real-time and auto-generates notes when finished.
    No need to upload files - just start, meet, and stop!
    """
    
    def __init__(self):
        self.is_recording = False
        self.audio_frames = []
        self.stream = None
        self.audio = None
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2  # Stereo (captures system audio better)
        self.RATE = 44100
        
        # Recording file
        self.temp_dir = Path("temp_recordings")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Generator
        self.generator = MeetingNotesGenerator(whisper_model_size="base")
        
        logger.info("✅ Real-time Meeting Recorder initialized")
    
    def list_audio_devices(self):
        """List all available audio input devices"""
        audio = pyaudio.PyAudio()
        logger.info("\n🎤 Available Audio Devices:")
        logger.info("="*60)
        
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Only input devices
                logger.info(f"[{i}] {info['name']}")
                logger.info(f"    Channels: {info['maxInputChannels']}, Rate: {int(info['defaultSampleRate'])} Hz")
        
        logger.info("="*60)
        audio.terminate()
    
    def start_recording(self, device_index=None):
        """Start recording meeting audio"""
        if self.is_recording:
            logger.warning("⚠️  Already recording!")
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Get device info
            if device_index is not None:
                device_info = self.audio.get_device_info_by_index(device_index)
                logger.info(f"🎤 Using device: {device_info['name']}")
            
            # Open stream
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.audio_frames = []
            self.start_time = time.time()
            
            logger.success("🔴 Recording started! Join your Google Meet now.")
            logger.info("💡 Press Ctrl+C or call stop_recording() when meeting ends")
            
            self.stream.start_stream()
            
        except Exception as e:
            logger.error(f"❌ Error starting recording: {e}")
            logger.info("💡 Tip: Try listing devices with list_audio_devices()")
            self.cleanup()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function to capture audio frames"""
        if self.is_recording:
            self.audio_frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop recording and process meeting notes"""
        if not self.is_recording:
            logger.warning("⚠️  Not currently recording!")
            return None
        
        logger.info("⏹️  Stopping recording...")
        self.is_recording = False
        
        # Stop stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Calculate duration
        duration = time.time() - self.start_time
        logger.info(f"⏱️  Recording duration: {int(duration)} seconds")
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"meeting_{timestamp}.wav"
        audio_path = self.temp_dir / audio_filename
        
        logger.info(f"💾 Saving recording: {audio_path}")
        
        # Write WAV file
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_frames))
        
        self.cleanup()
        
        logger.success("✅ Recording saved! Now processing meeting notes...")
        
        # Process immediately
        result = self.generator.process_meeting(str(audio_path))
        
        logger.success("🎉 Meeting notes generated!")
        logger.info(f"📄 PDF: {result['pdf_path']}")
        logger.info(f"📋 JSON: {result['json_path']}")
        
        # Print summary
        print("\n" + "="*60)
        print("MEETING SUMMARY")
        print("="*60)
        print(result["summary"])
        print("\n" + "="*60)
        
        return result
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio:
            self.audio.terminate()
            self.audio = None
    
    def record_meeting_interactive(self, device_index=None):
        """
        Interactive mode: Start recording and wait for user to press Enter to stop
        """
        self.start_recording(device_index)
        
        try:
            input("\n⏸️  Press ENTER when meeting ends to generate notes...\n")
        except KeyboardInterrupt:
            pass
        
        return self.stop_recording()


# CLI Usage
if __name__ == "__main__":
    import sys
    
    recorder = RealtimeMeetingRecorder()
    
    # Check if user wants to list devices
    if len(sys.argv) > 1 and sys.argv[1] == "--list-devices":
        recorder.list_audio_devices()
        sys.exit(0)
    
    # Get device index if provided
    device_index = None
    if len(sys.argv) > 1 and sys.argv[1] != "--list-devices":
        try:
            device_index = int(sys.argv[1])
        except ValueError:
            print("Usage: python realtime_meeting_recorder.py [device_index]")
            print("       python realtime_meeting_recorder.py --list-devices")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎤 REAL-TIME MEETING NOTES GENERATOR")
    print("="*60)
    print("1. Start this script BEFORE joining your Google Meet")
    print("2. Join your meeting - audio will be captured automatically")
    print("3. Press ENTER when meeting ends")
    print("4. Get instant PDF notes! ✨")
    print("="*60 + "\n")
    
    if device_index is None:
        print("💡 Tip: Run with --list-devices to see available audio devices")
        print("💡 Tip: Use 'device_index' argument to select specific device\n")
    
    # Start interactive recording
    try:
        result = recorder.record_meeting_interactive(device_index)
        if result:
            print("\n✅ All done! Your meeting notes are ready.")
    except KeyboardInterrupt:
        print("\n\n⏹️  Recording interrupted")
        recorder.stop_recording()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        recorder.cleanup()
