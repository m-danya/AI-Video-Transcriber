#!/usr/bin/env python3
"""AI Video Transcriber startup script."""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check whether dependencies are installed."""
    import sys
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn", 
        "yt-dlp": "yt_dlp",
        "faster-whisper": "faster_whisper",
        "openai": "openai"
    }
    
    missing_packages = []
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(display_name)
    
    if missing_packages:
        print("❌ Missing dependency packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nRun this command to install dependencies:")
        print("source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_ffmpeg():
    """Check whether FFmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg was not found")
        print("Please install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: download it from https://ffmpeg.org/download.html")
        return False

def setup_environment():
    """Set environment variables."""
    # Set OpenAI configuration
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY is not set")
        print("Set it with: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print("✅ OpenAI API key is set")
    
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://oneapi.basevec.com/v1"
        print("✅ OpenAI Base URL was set")
    
    # Set other defaults
    if not os.getenv("WHISPER_MODEL_SIZE"):
        os.environ["WHISPER_MODEL_SIZE"] = "base"
    
    print("🔑 OpenAI API is configured; summary generation is available")
    return True

def main():
    """Main entry point."""
    # Check whether production mode is enabled (disables hot reload)
    production_mode = "--prod" in sys.argv or os.getenv("PRODUCTION_MODE") == "true"
    
    print("🚀 AI Video Transcriber startup check")
    if production_mode:
        print("🔒 Production mode - hot reload disabled")
    else:
        print("🔧 Development mode - hot reload enabled")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg is not installed; some video formats may not process correctly")
    
    # Set up environment
    setup_environment()
    
    print("\n🎉 Startup checks complete!")
    print("=" * 50)
    
    # Start server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8099))
    
    print("\n🌐 Starting server...")
    print(f"   URL: http://localhost:{port}")
    print("   Press Ctrl+C to stop the service")
    print("=" * 50)
    
    try:
        # Switch to the backend directory and start the service
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        # Enable hot reload only in development mode
        if not production_mode:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Service stopped")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
