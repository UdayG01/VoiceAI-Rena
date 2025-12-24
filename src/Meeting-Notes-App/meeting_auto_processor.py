import time
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger
from meeting_notes_generator import MeetingNotesGenerator

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


class MeetingRecordingHandler(FileSystemEventHandler):
    """Automatically processes new meeting recordings"""
    
    def __init__(self, watch_folder: str):
        self.watch_folder = Path(watch_folder)
        self.generator = MeetingNotesGenerator(whisper_model_size="base")
        self.processed_files = set()
        
        # Supported audio/video formats
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.mp4', '.webm', '.ogg', '.flac'}
        
        logger.info(f"👀 Watching folder: {self.watch_folder}")
        logger.info(f"📁 Supported formats: {', '.join(self.supported_formats)}")
    
    def on_created(self, event):
        """Called when a new file is created"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's a supported audio/video file
        if file_path.suffix.lower() in self.supported_formats:
            # Wait a bit to ensure file is fully written
            time.sleep(2)
            self.process_file(file_path)
    
    def on_modified(self, event):
        """Called when a file is modified (useful for large files being downloaded)"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process if not already processed and file size is stable
        if (file_path.suffix.lower() in self.supported_formats and 
            str(file_path) not in self.processed_files):
            
            # Check if file size is stable (download complete)
            if self.is_file_stable(file_path):
                self.process_file(file_path)
    
    def is_file_stable(self, file_path: Path, wait_time: int = 5) -> bool:
        """Check if file size is stable (download/write complete)"""
        try:
            size1 = file_path.stat().st_size
            time.sleep(wait_time)
            size2 = file_path.stat().st_size
            return size1 == size2 and size1 > 0
        except:
            return False
    
    def process_file(self, file_path: Path):
        """Process the meeting recording"""
        file_str = str(file_path)
        
        # Skip if already processed
        if file_str in self.processed_files:
            return
        
        logger.info(f"🆕 New meeting recording detected: {file_path.name}")
        
        try:
            # Mark as being processed
            self.processed_files.add(file_str)
            
            # Process the meeting
            result = self.generator.process_meeting(file_str)
            
            logger.success(f"✅ Processing complete!")
            logger.success(f"📄 PDF: {result['pdf_path']}")
            logger.success(f"📋 JSON: {result['json_path']}")
            
            # Optional: Print summary
            print("\n" + "="*60)
            print("MEETING SUMMARY")
            print("="*60)
            print(result["summary"])
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            # Remove from processed set so it can be retried
            self.processed_files.discard(file_str)


def main():
    """Main function to start the auto-processor"""
    import sys
    
    # Default watch folder (user's Downloads)
    default_watch_folder = str(Path.home() / "Downloads")
    
    if len(sys.argv) > 1:
        watch_folder = sys.argv[1]
    else:
        watch_folder = default_watch_folder
    
    # Create folder if it doesn't exist
    Path(watch_folder).mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Meeting Notes Auto-Processor Starting...")
    logger.info(f"📂 Watching: {watch_folder}")
    logger.info("💡 Drop your meeting recordings here and they'll be auto-processed!")
    logger.info("⏸️  Press Ctrl+C to stop\n")
    
    # Set up the file watcher
    event_handler = MeetingRecordingHandler(watch_folder)
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Stopping auto-processor...")
        observer.stop()
    
    observer.join()
    logger.info("👋 Auto-processor stopped")


if __name__ == "__main__":
    main()
