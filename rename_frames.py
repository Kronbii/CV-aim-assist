import os
import json
from pathlib import Path

def rename_frames():
    """
    Rename frames from frame_XXX_angle_YYY.jpg to {id}_{num}.jpg format
    """
    # Load metadata
    metadata_file = Path("character_frames/all_characters_metadata.json")
    
    if not metadata_file.exists():
        print("Metadata file not found!")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("Renaming frames...")
    
    for character_id, data in metadata.items():
        character_dir = Path(data['output_directory'])
        
        if not character_dir.exists():
            print(f"Directory not found for character {character_id}")
            continue
        
        print(f"\nProcessing character {character_id}...")
        
        # Find all frame files
        frame_files = sorted(list(character_dir.glob("frame_*.jpg")))
        
        if not frame_files:
            print(f"  No frames found for character {character_id}")
            continue
        
        # Rename each frame
        for i, frame_file in enumerate(frame_files):
            # New filename: {id}_{num}.jpg where num starts from 1
            new_name = f"{character_id}_{i+1}.jpg"
            new_path = character_dir / new_name
            
            # Rename the file
            frame_file.rename(new_path)
            print(f"  Renamed {frame_file.name} -> {new_name}")
        
        print(f"  Renamed {len(frame_files)} frames for character {character_id}")
    
    print("\n=== RENAMING COMPLETE ===")

if __name__ == "__main__":
    rename_frames()
