import os
import sys
import subprocess
import webbrowser
import time
from threading import Thread

def install_requirements():
    """Install the required packages."""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Packages installed successfully!")

def setup_kaggle_dataset():
    """Set up the Kaggle dataset."""
    print("Setting up Kaggle dataset...")
    from setup_kaggle_dataset import main as setup_kaggle
    setup_kaggle()
    print("Kaggle dataset set up successfully!")

def start_flask_app():
    """Start the Flask application."""
    print("Starting Flask application...")
    from app import app
    app.run(debug=True)

def open_browser():
    """Open the web browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://127.0.0.1:5000/')

if __name__ == "__main__":
    # Install requirements if needed
    if not os.path.exists('requirements.txt'):
        print("requirements.txt not found!")
        sys.exit(1)
    
    # Setup Kaggle dataset
    if not os.path.exists('datasets/drug_images'):
        setup_kaggle_dataset()
    
    # Start Flask app in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Open web browser
    open_browser()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Application stopped.")