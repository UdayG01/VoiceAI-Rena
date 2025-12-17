import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from loguru import logger

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


class MeetingNotesGenerator:
    """
    Generates meeting summaries, MOM, and action items from audio recordings.
    Supports Hindi, English, and Hinglish.
    """
    
    def __init__(self, whisper_model_size: str = "base", ollama_model: str = "qwen3:1.7b"):
        """
        Initialize the meeting notes generator.
        
        Args:
            whisper_model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            ollama_model: Ollama model for summarization
        """
        logger.info(f"🔧 Initializing Meeting Notes Generator...")
        
        # Initialize Whisper model (local STT)
        logger.info(f"⏳ Loading Whisper model: {whisper_model_size}")
        self.whisper = WhisperModel(
            whisper_model_size, 
            device="cpu", 
            compute_type="int8"
        )
        
        # Initialize Ollama LLM
        logger.info(f"⏳ Connecting to Ollama model: {ollama_model}")
        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0.3,
            max_tokens=1024,
        )
        
        # Output directory
        self.output_dir = Path("meeting_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Meeting Notes Generator ready!")
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language hint (en, hi, or None for auto-detect)
        
        Returns:
            Dict with transcript, detected_language, and segments
        """
        logger.info(f"🎙️ Transcribing: {audio_path}")
        
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            task="transcribe"  # Use 'translate' to convert to English
        )
        
        # Collect full transcript
        full_transcript = []
        segment_list = []
        
        for segment in segments:
            text = segment.text.strip()
            full_transcript.append(text)
            segment_list.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })
        
        result = {
            "transcript": " ".join(full_transcript),
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 2),
            "segments": segment_list
        }
        
        logger.info(f"✅ Transcription complete | Language: {info.language} ({info.language_probability:.2%})")
        return result
    
    def generate_summary_and_mom(self, transcript: str, language: str = "en") -> Dict:
        """
        Generate meeting summary, MOM, and action items using LLM.
        
        Args:
            transcript: Full meeting transcript
            language: Language for output (en/hi)
        
        Returns:
            Dict with summary, mom, and action_items
        """
        logger.info("🧠 Generating meeting notes...")
        
        prompt = f"""You are an expert meeting notes assistant. Analyze the following meeting transcript and provide:

1. **Summary**: A concise 3-4 sentence overview of the meeting
2. **Minutes of Meeting (MOM)**: Key discussion points in bullet format
3. **Action Items**: Specific tasks with responsible persons (if mentioned)

TRANSCRIPT:
{transcript}

IMPORTANT: 
- Output ONLY valid JSON
- No markdown, no code blocks
- Use this exact structure:

{{
  "summary": "Brief meeting overview here",
  "mom": [
    "First key point discussed",
    "Second key point discussed"
  ],
  "action_items": [
    "Task 1 - Assigned to Person A",
    "Task 2 - Assigned to Person B"
  ]
}}"""

        response = self.llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean response (remove markdown if present)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
            logger.info("✅ Meeting notes generated successfully")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            # Fallback
            return {
                "summary": "Could not generate summary",
                "mom": ["Error processing meeting notes"],
                "action_items": []
            }
    
    def create_pdf(self, meeting_data: Dict, output_filename: str = None) -> str:
        """
        Create PDF report from meeting data.
        
        Args:
            meeting_data: Dict containing transcript, summary, mom, action_items
            output_filename: Custom filename (auto-generated if None)
        
        Returns:
            Path to generated PDF
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"meeting_notes_{timestamp}.pdf"
        
        pdf_path = self.output_dir / output_filename
        
        logger.info(f"📄 Creating PDF: {pdf_path}")
        
        # Create PDF
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor='#2C3E50',
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#34495E',
            spaceAfter=10,
            spaceBefore=10
        )
        
        # Title
        story.append(Paragraph("Meeting Notes", title_style))
        story.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary
        story.append(Paragraph("Summary", heading_style))
        story.append(Paragraph(meeting_data.get("summary", "N/A"), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Minutes of Meeting
        story.append(Paragraph("Minutes of Meeting", heading_style))
        for point in meeting_data.get("mom", []):
            story.append(Paragraph(f"• {point}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        story.append(Spacer(1, 0.2*inch))
        
        # Action Items
        story.append(Paragraph("Action Items", heading_style))
        action_items = meeting_data.get("action_items", [])
        if action_items:
            for item in action_items:
                story.append(Paragraph(f"☑ {item}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("No action items identified", styles['Italic']))
        
        # Build PDF
        doc.build(story)
        logger.info(f"✅ PDF created: {pdf_path}")
        
        return str(pdf_path)
    
    def process_meeting(self, audio_path: str, language: str = None) -> Dict:
        """
        Full pipeline: Transcribe → Summarize → Generate PDF
        
        Args:
            audio_path: Path to meeting audio file
            language: Language hint (optional)
        
        Returns:
            Dict with all meeting data + PDF path
        """
        logger.info(f"🚀 Processing meeting: {audio_path}")
        
        # Step 1: Transcribe
        transcription = self.transcribe_audio(audio_path, language)
        
        # Step 2: Generate notes
        notes = self.generate_summary_and_mom(
            transcription["transcript"],
            transcription["detected_language"]
        )
        
        # Step 3: Combine data
        meeting_data = {
            "transcript": transcription["transcript"],
            "detected_language": transcription["detected_language"],
            "summary": notes["summary"],
            "mom": notes["mom"],
            "action_items": notes["action_items"],
            "segments": transcription["segments"]
        }
        
        # Step 4: Create PDF
        pdf_path = self.create_pdf(meeting_data)
        meeting_data["pdf_path"] = pdf_path
        
        # Step 5: Save JSON
        json_path = pdf_path.replace(".pdf", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meeting_data, f, indent=2, ensure_ascii=False)
        meeting_data["json_path"] = json_path
        
        logger.info("🎉 Meeting processing complete!")
        return meeting_data


# CLI Usage Example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python meeting_notes_generator.py <audio_file_path> [language]")
        print("Example: python meeting_notes_generator.py meeting.mp3 en")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    
    generator = MeetingNotesGenerator(whisper_model_size="base")
    result = generator.process_meeting(audio_file, language=lang)
    
    print("\n" + "="*50)
    print("MEETING SUMMARY")
    print("="*50)
    print(result["summary"])
    print("\n📄 PDF saved at:", result["pdf_path"])
    print("📋 JSON saved at:", result["json_path"])
