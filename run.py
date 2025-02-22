import uvicorn
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Get static files directory path
    static_dir = Path(project_root) / "src" / "interface" / "static"
    
    # Create necessary directories if they don't exist
    static_js_dir = static_dir / "js"
    static_js_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if webcam-recorder.js exists, copy it if needed
    js_file = static_js_dir / "webcam-recorder.js"
    if not js_file.exists():
        # Look for webcam-recorder.js in common locations
        potential_locations = [
            Path(project_root) / "webcam-recorder.js",
            Path(project_root) / "src" / "webcam-recorder.js",
        ]
        
        for location in potential_locations:
            if location.exists():
                print(f"Found webcam-recorder.js at {location}")
                print(f"Copying to {js_file}")
                with open(location, 'r') as src_file:
                    with open(js_file, 'w') as dst_file:
                        dst_file.write(src_file.read())
                break
        else:
            print("Warning: webcam-recorder.js not found in common locations")
            print(f"Please ensure the file exists at {js_file}")
    
    # Run the application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )