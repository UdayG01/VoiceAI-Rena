from flask import Flask, render_template_string, request, send_file, jsonify
import os
from pathlib import Path
from meeting_notes_generator import MeetingNotesGenerator
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Global generator instance
generator = None
processing_status = {}

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'webm', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meeting Notes Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        
        .upload-area.dragover {
            background: #e8ebff;
            border-color: #764ba2;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.3em;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .upload-hint {
            color: #999;
            font-size: 0.9em;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .status.processing {
            background: #fff3cd;
            border: 1px solid #ffc107;
            display: block;
        }
        
        .status.success {
            background: #d4edda;
            border: 1px solid #28a745;
            display: block;
        }
        
        .status.error {
            background: #f8d7da;
            border: 1px solid #dc3545;
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 15px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s;
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        .results {
            margin-top: 30px;
        }
        
        .result-card {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        
        .result-card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .download-btn {
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            text-decoration: none;
            display: inline-block;
            margin-top: 10px;
            transition: background 0.3s;
        }
        
        .download-btn:hover {
            background: #218838;
        }
        
        .file-info {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9ff;
            border-radius: 10px;
            display: none;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Meeting Notes Generator</h1>
        <p class="subtitle">Upload your Google Meet recording to generate automatic notes</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Drop your meeting recording here</div>
                <div class="upload-hint">or click to browse (MP3, MP4, WAV, M4A, WebM)</div>
                <input type="file" id="fileInput" name="file" accept="audio/*,video/*">
            </div>
            
            <div class="file-info" id="fileInfo">
                <strong>Selected:</strong> <span id="fileName"></span>
            </div>
            
            <button type="submit" class="btn" id="uploadBtn">Generate Meeting Notes ✨</button>
        </form>
        
        <div class="status" id="status">
            <div id="statusMessage"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="results" id="results"></div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const uploadBtn = document.getElementById('uploadBtn');
        const status = document.getElementById('status');
        const statusMessage = document.getElementById('statusMessage');
        const progressFill = document.getElementById('progressFill');
        const results = document.getElementById('results');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        
        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
            showFileInfo();
        });
        
        // File selected
        fileInput.addEventListener('change', showFileInfo);
        
        function showFileInfo() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                fileName.textContent = file.name;
                fileInfo.style.display = 'block';
            }
        }
        
        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (fileInput.files.length === 0) {
                alert('Please select a file first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Show processing status
            status.className = 'status processing';
            statusMessage.innerHTML = '<div class="spinner"></div> Processing your meeting... This may take a few minutes.';
            progressFill.style.width = '30%';
            uploadBtn.disabled = true;
            results.innerHTML = '';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                progressFill.style.width = '100%';
                
                if (response.ok) {
                    const data = await response.json();
                    
                    status.className = 'status success';
                    statusMessage.textContent = '✅ Meeting notes generated successfully!';
                    
                    // Display results
                    results.innerHTML = `
                        <div class="result-card">
                            <h3>📝 Summary</h3>
                            <p>${data.summary}</p>
                        </div>
                        
                        <div class="result-card">
                            <h3>📋 Minutes of Meeting</h3>
                            <ul>
                                ${data.mom.map(item => '<li>' + item + '</li>').join('')}
                            </ul>
                        </div>
                        
                        <div class="result-card">
                            <h3>✅ Action Items</h3>
                            ${data.action_items.length > 0 
                                ? '<ul>' + data.action_items.map(item => '<li>' + item + '</li>').join('') + '</ul>'
                                : '<p>No action items identified</p>'}
                        </div>
                        
                        <div class="result-card">
                            <h3>⬇️ Download</h3>
                            <a href="/download/${data.filename}" class="download-btn">📄 Download PDF</a>
                            <a href="/download_json/${data.filename}" class="download-btn">📋 Download JSON</a>
                        </div>
                    `;
                } else {
                    throw new Error('Processing failed');
                }
            } catch (error) {
                status.className = 'status error';
                statusMessage.textContent = '❌ Error processing meeting. Please try again.';
                progressFill.style.width = '0%';
            } finally {
                uploadBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(filepath)
    
    try:
        # Process meeting
        global generator
        if generator is None:
            generator = MeetingNotesGenerator(whisper_model_size="base")
        
        result = generator.process_meeting(str(filepath))
        
        # Extract filename from PDF path
        pdf_filename = Path(result['pdf_path']).name
        
        return jsonify({
            'success': True,
            'summary': result['summary'],
            'mom': result['mom'],
            'action_items': result['action_items'],
            'filename': pdf_filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    pdf_path = Path('meeting_outputs') / filename
    if pdf_path.exists():
        return send_file(pdf_path, as_attachment=True)
    return "File not found", 404


@app.route('/download_json/<filename>')
def download_json(filename):
    json_filename = filename.replace('.pdf', '.json')
    json_path = Path('meeting_outputs') / json_filename
    if json_path.exists():
        return send_file(json_path, as_attachment=True)
    return "File not found", 404


if __name__ == '__main__':
    print("🚀 Meeting Notes Generator Web Interface")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏸️  Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
