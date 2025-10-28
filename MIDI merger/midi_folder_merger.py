#!/usr/bin/env python3
"""
MIDI Folder Merger - Automatically merges all MIDI files in a folder into one multi-track MIDI file
Preserves track names, automation, and all MIDI data. Uses filename as track name if no name exists.
Handles Type 0, Type 1, and Type 2 MIDI files with proper timing conversion.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

def install_mido():
    """Install mido library if not available"""
    try:
        import mido
        return True
    except ImportError:
        print("📦 Installing mido library...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mido"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import mido
            print("✅ mido library installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install mido: {e}")
            return False

def find_midi_files(directory: str) -> List[Tuple[str, float]]:
    """Find all MIDI files directly in the specified directory (no subdirectories), sorted by creation date (oldest first)"""
    midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
    midi_files = []
    
    # Only search in the current directory, not in subdirectories
    for file_path in Path(directory).iterdir():
        # Only process files (not directories) and only MIDI files
        if file_path.is_file() and file_path.suffix in midi_extensions:
            # Get creation time (stat().st_ctime on Windows, st_birthtime on macOS, st_ctime on Linux)
            try:
                stat = file_path.stat()
                # Use creation time if available, otherwise use modification time
                creation_time = getattr(stat, 'st_birthtime', stat.st_ctime)
                midi_files.append((str(file_path), creation_time))
            except Exception:
                # Fallback to modification time if creation time is not available
                midi_files.append((str(file_path), file_path.stat().st_mtime))
    
    # Sort by creation time (oldest first)
    midi_files.sort(key=lambda x: x[1])
    return midi_files

def get_track_name_from_file(filename: str) -> str:
    """Extract track name from filename (remove extension)"""
    return Path(filename).stem

def is_generic_track_name(name: str) -> bool:
    """Check if track name is generic (like 'Track 1', 'Track 2', etc.)"""
    if not name or name.strip() == "":
        return True
    
    # Common generic patterns
    generic_patterns = [
        r'^track\s*\d*$',          # Track, Track 1, Track 2, etc.
        r'^untitled\s*\d*$',       # Untitled, Untitled 1, etc.
        r'^midi\s*track\s*\d*$',   # MIDI Track, MIDI Track 1, etc.
        r'^channel\s*\d+$',        # Channel 1, Channel 2, etc.
        r'^new\s*track\s*\d*$',    # New Track, New Track 1, etc.
        r'^track\s*name$',         # Track Name
        r'^empty$',                # Empty
        r'^no\s*name$',            # No Name
    ]
    
    name_lower = name.lower().strip()
    for pattern in generic_patterns:
        if re.match(pattern, name_lower):
            return True
    
    return False

def extract_track_name_from_track(track) -> Optional[str]:
    """Extract track name from a single MIDI track"""
    try:
        for msg in track:
            if msg.type == 'track_name':
                return msg.name
    except Exception:
        pass
    return None

def should_skip_meta_message(msg, is_first_track: bool) -> bool:
    """
    Determine if a meta message should be skipped to avoid conflicts
    Only keep tempo/time_signature/key_signature from the first file's first track
    """
    # These meta messages should only appear once (in first track)
    global_meta_types = ['set_tempo', 'time_signature', 'key_signature', 'smpte_offset']
    
    if msg.type in global_meta_types:
        return not is_first_track
    
    # Always skip end_of_track as mido handles this automatically
    if msg.type == 'end_of_track':
        return True
    
    return False

def convert_delta_time(delta_time: int, source_tpb: int, target_tpb: int) -> int:
    """
    Convert delta time from source ticks_per_beat to target ticks_per_beat
    """
    if source_tpb == target_tpb:
        return delta_time
    
    # Convert: (delta_time / source_tpb) * target_tpb
    # Use floating point for accuracy, then round
    converted = round((delta_time * target_tpb) / source_tpb)
    return max(0, converted)  # Ensure non-negative

def merge_midi_files(input_files: List[Tuple[str, float]], output_file: str) -> bool:
    """
    Merge multiple MIDI files into one multi-track MIDI file
    Preserves all track names, automation, and MIDI data
    Handles different timing resolutions and MIDI types (Type 0, 1, 2)
    """
    try:
        if not input_files:
            print("❌ No MIDI files found")
            return False
        
        print(f"🎵 Found {len(input_files)} MIDI files to merge...")
        
        # Show processing order with creation times
        print(f"\n📅 Processing order (by creation date):")
        for i, (file_path, creation_time) in enumerate(input_files, 1):
            filename = os.path.basename(file_path)
            from datetime import datetime
            creation_date = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   {i}. {filename} (created: {creation_date})")
        
        # Import mido after installation
        import mido
        from mido import MidiFile, MidiTrack, MetaMessage, Message
        
        # Create new MIDI file (Type 1 - multiple tracks)
        merged_midi = MidiFile(type=1)
        
        # Use the first file's timing as reference
        first_file = MidiFile(input_files[0][0])
        target_tpb = first_file.ticks_per_beat
        merged_midi.ticks_per_beat = target_tpb
        
        print(f"\n⚙️  Target timing: {target_tpb} ticks per beat")
        
        # Track if this is the first track (for global meta messages)
        is_first_track_overall = True
        
        # Process each input file
        for file_idx, (input_file, creation_time) in enumerate(input_files):
            filename = os.path.basename(input_file)
            print(f"\n📁 Processing: {filename}")
            
            try:
                # Load MIDI file
                midi_file = MidiFile(input_file)
                source_tpb = midi_file.ticks_per_beat
                
                # Check for timing conversion
                needs_conversion = (source_tpb != target_tpb)
                if needs_conversion:
                    print(f"   ⚙️  Converting timing: {source_tpb} → {target_tpb} ticks per beat")
                
                if len(midi_file.tracks) == 0:
                    print(f"⚠️  No tracks in {filename}")
                    continue
                
                # Detect MIDI type
                midi_type = getattr(midi_file, 'type', 1)
                if midi_type == 0:
                    print(f"   ℹ️  Type 0 MIDI (single track with all channels)")
                
                # Process each track in the file
                for track_idx, source_track in enumerate(midi_file.tracks):
                    # Skip empty tracks
                    if len(source_track) == 0:
                        continue
                    
                    # Create new track for merged file
                    new_track = MidiTrack()
                    
                    # Extract track name from THIS specific track
                    track_name = extract_track_name_from_track(source_track)
                    filename_stem = get_track_name_from_file(filename)
                    
                    # Decide on final track name
                    if not track_name or is_generic_track_name(track_name):
                        # Use filename as track name
                        track_name = filename_stem
                        if len(midi_file.tracks) > 1:
                            track_name += f" (Track {track_idx + 1})"
                    else:
                        # Track has a meaningful name, but add file context if multiple tracks
                        if len(midi_file.tracks) > 1 and track_idx > 0:
                            track_name = f"{filename_stem} - {track_name}"
                    
                    # Add track name as first message
                    new_track.append(MetaMessage('track_name', name=track_name, time=0))
                    
                    # Copy all messages from source track
                    has_content = False
                    for msg in source_track:
                        # Skip track_name as we already added it
                        if msg.type == 'track_name':
                            continue
                        
                        # Check if we should skip certain meta messages
                        if hasattr(msg, 'type') and msg.type in ['set_tempo', 'time_signature', 
                                                                   'key_signature', 'smpte_offset']:
                            if not is_first_track_overall:
                                continue  # Skip global meta messages in non-first tracks
                        
                        # Skip end_of_track (mido adds it automatically)
                        if hasattr(msg, 'type') and msg.type == 'end_of_track':
                            continue
                        
                        # Convert timing if needed
                        if needs_conversion and hasattr(msg, 'time'):
                            new_time = convert_delta_time(msg.time, source_tpb, target_tpb)
                            msg = msg.copy(time=new_time)
                        
                        new_track.append(msg)
                        
                        # Check if track has actual content (not just meta messages)
                        if hasattr(msg, 'type') and not msg.type.startswith('end_'):
                            if msg.type not in ['track_name', 'text', 'copyright', 'marker', 
                                               'cue_marker', 'instrument_name', 'lyric']:
                                has_content = True
                    
                    # Only add track if it has content or is the first track
                    if has_content or is_first_track_overall:
                        # Add track to merged file
                        merged_midi.tracks.append(new_track)
                        print(f"   ✓ Added track: {track_name}")
                        is_first_track_overall = False
                    else:
                        print(f"   ⊘ Skipped empty track: {track_name}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                print(f"   Details: {traceback.format_exc()}")
                continue
        
        if len(merged_midi.tracks) == 0:
            print("❌ No tracks found to merge")
            return False
        
        # Save merged MIDI file
        merged_midi.save(output_file)
        print(f"\n✅ Successfully created: {output_file}")
        print(f"📊 Total tracks: {len(merged_midi.tracks)}")
        print(f"⚙️  Output type: Type {merged_midi.type} MIDI")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging MIDI files: {e}")
        import traceback
        print(f"Details: {traceback.format_exc()}")
        return False

def main():
    """Main function - automatically merge all MIDI files in script directory"""
    
    print("🎵 MIDI Folder Merger")
    print("=" * 50)
    
    # Install mido if needed
    if not install_mido():
        print("❌ Cannot proceed without mido library")
        return 1
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = script_dir
    print(f"📁 Scanning directory: {current_dir}")
    
    # Find MIDI files (sorted by creation date)
    midi_files = find_midi_files(current_dir)
    
    if not midi_files:
        print("❌ No MIDI files found in current directory")
        print("\n💡 Place MIDI files (.mid, .midi) in this directory and run the script again")
        return 1
    
    # Determine output filename (in the same directory as the script)
    output_file = os.path.join(current_dir, "merged_midi.mid")
    counter = 1
    while os.path.exists(output_file):
        output_file = os.path.join(current_dir, f"merged_midi_{counter}.mid")
        counter += 1
    
    print(f"📤 Output file: {output_file}")
    
    # Merge files
    success = merge_midi_files(midi_files, output_file)
    
    if success:
        print(f"\n🎉 MIDI merge completed successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Merged {len(midi_files)} files into tracks")
        
        # Show track information
        try:
            import mido
            merged = mido.MidiFile(output_file)
            print(f"\n📋 Track information:")
            for i, track in enumerate(merged.tracks):
                track_name = "Unknown"
                for msg in track:
                    if msg.type == 'track_name':
                        track_name = msg.name
                        break
                print(f"   Track {i+1}: {track_name}")
        except Exception:
            pass
        
        return 0
    else:
        print(f"\n❌ MIDI merge failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

