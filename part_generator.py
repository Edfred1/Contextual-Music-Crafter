import os
import yaml
import google.generativeai as genai
import re
from colorama import Fore, Style, init
from midiutil import MIDIFile
import random
import time
from typing import List, Tuple, Dict
import math
import json

# --- ROBUST CONFIG FILE PATH ---
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join the script's directory with the config file name to create an absolute path
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
# --- END ROBUST PATH ---

BEATS_PER_BAR = 4
TICKS_PER_BEAT = 480

# Add new constants at the beginning of the file
AVAILABLE_LENGTHS = [8, 16, 32]
DEFAULT_LENGTH = 16

# Initialize Colorama for console color support
init(autoreset=True)

# Constants
GENERATED_CODE_FILE = os.path.join(script_dir, "generated_code.py")

def load_config(config_file):
    """Loads configuration from YAML file and validates it."""
    print(Fore.CYAN + "Loading configuration..." + Style.RESET_ALL)
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Validate critical fields first
        if not config.get("api_key"):
            raise ValueError("API key is missing in configuration")
            
        if not config.get("model_name"):
            raise ValueError("Model name is missing in configuration")
            
        # Validate API-Key Format (basic check)
        if not isinstance(config["api_key"], str) or len(config["api_key"]) < 10:
            raise ValueError("API key appears to be invalid")

        # Validation messages
        required_fields = [
            "inspiration", 
            "genre", 
            "bpm", 
            "key_scale",  # We use key_scale instead of separate root_note
            "api_key", 
            "model_name", 
            "instruments",
            "time_signature"
        ]
        
        for field in required_fields:
            if field not in config:
                if field == "time_signature":
                    config["time_signature"] = {
                        "beats_per_bar": 4,
                        "beat_value": 4
                    }
                else:
                    raise ValueError(f"Error: Required field '{field}' missing in configuration.")

        # Parse key_scale into root_note and scale_type
        key_parts = config["key_scale"].lower().split()
        note_map = {
            "c": 60, "c#": 61, "db": 61, 
            "d": 62, "d#": 63, "eb": 63,
            "e": 64, "e#": 65, "fb": 64,
            "f": 65, "f#": 66, "gb": 66,
            "g": 67, "g#": 68, "ab": 68,
            "a": 69, "a#": 70, "bb": 70,
            "b": 71, "b#": 72, "cb": 71
        }
        
        if len(key_parts) >= 2:
            root = key_parts[0]
            config["root_note"] = note_map.get(root, 60)  # Default to C4 if not found
            config["scale_type"] = " ".join(key_parts[1:])
        else:
            config["root_note"] = 60  # Default to C4
            config["scale_type"] = "major"  # Default to major scale

        # Validate time_signature Format
        if isinstance(config["time_signature"], dict):
            if "beats_per_bar" not in config["time_signature"] or "beat_value" not in config["time_signature"]:
                config["time_signature"] = {
                    "beats_per_bar": 4,
                    "beat_value": 4
                }
        else:
            # Convert String format (e.g. "4/4") to Dict
            try:
                if isinstance(config["time_signature"], str) and "/" in config["time_signature"]:
                    beats, value = config["time_signature"].split("/")
                    config["time_signature"] = {
                        "beats_per_bar": int(beats),
                        "beat_value": int(value)
                    }
            except:
                config["time_signature"] = {
                    "beats_per_bar": 4,
                    "beat_value": 4
                }

        print(Fore.GREEN + "Configuration loaded successfully." + Style.RESET_ALL)
        return config
    except Exception as e:
        print(Fore.RED + f"Error loading configuration: {str(e)}" + Style.RESET_ALL)
        exit()

def get_scale_notes(root_note, scale_type="minor"):
    """Generates notes for various scale types, including church modes."""
    try:
        # Ensure root_note is a number
        root_note = int(root_note) if isinstance(root_note, (str, tuple)) else root_note
        
        intervals = []
        
        # Basic scales
        # Major/Ionian
        if scale_type == "major" or scale_type == "ionian":
            # Major/Ionian: W-W-H-W-W-W-H
            intervals = [0, 2, 4, 5, 7, 9, 11]
        elif scale_type == "minor" or scale_type == "natural minor" or scale_type == "aeolian":
            # Natural Minor/Aeolian
            intervals = [0, 2, 3, 5, 7, 8, 10]
        elif scale_type == "harmonic minor":
            # Harmonic Minor
            intervals = [0, 2, 3, 5, 7, 8, 11]
        elif scale_type == "melodic minor":
            # Melodic Minor
            intervals = [0, 2, 3, 5, 7, 9, 11]
        
        # Church modes
        elif scale_type == "dorian":
            # Dorian: W-H-W-W-W-H-W
            intervals = [0, 2, 3, 5, 7, 9, 10]
        elif scale_type == "phrygian":
            # Phrygian: H-W-W-W-H-W-W
            intervals = [0, 1, 3, 5, 7, 8, 10]
        elif scale_type == "lydian":
            # Lydian: W-W-W-H-W-W-H
            intervals = [0, 2, 4, 6, 7, 9, 11]
        elif scale_type == "mixolydian":
            # Mixolydian: W-W-H-W-W-H-W
            intervals = [0, 2, 4, 5, 7, 9, 10]
        elif scale_type == "locrian":
            # Locrian: H-W-W-H-W-W-W
            intervals = [0, 1, 3, 5, 6, 8, 10]
        
        # Pentatonic scales
        elif scale_type == "major pentatonic":
            # Major Pentatonic
            intervals = [0, 2, 4, 7, 9]
        elif scale_type == "minor pentatonic":
            # Minor Pentatonic
            intervals = [0, 3, 5, 7, 10]
        
        # Other important scales
        elif scale_type == "chromatic":
            # Chromatic: H-H-H-H-H-H-H-H-H-H-H-H
            intervals = list(range(12))
        elif scale_type == "whole tone":
            # Whole Tone
            intervals = [0, 2, 4, 6, 8, 10]
        elif scale_type == "diminished":
            # Diminished
            intervals = [0, 1, 3, 4, 6, 7, 9, 10]
        elif scale_type == "augmented":
            # Augmented
            intervals = [0, 3, 4, 7, 8, 11]
        elif scale_type == "byzantine":
            # Byzantine: H-W+H-H-W-H-W+H
            intervals = [0, 1, 4, 5, 7, 8, 11]
        elif scale_type == "hungarian minor":
            # Hungarian Minor: W-H-W+H-H-W+H-H
            intervals = [0, 2, 3, 6, 7, 8, 11]
        elif scale_type == "persian":
            # Persian: H-W+H-H-H-W+H-W
            intervals = [0, 1, 4, 5, 6, 8, 11]
        elif scale_type == "arabic":
            # Arabic: W-H-W+H-H-W+H-H
            intervals = [0, 2, 3, 6, 7, 8, 11]
        elif scale_type == "jewish" or scale_type == "ahava raba":
            # Jewish (Ahava Raba): H-W+H-H-W-H-W
            intervals = [0, 1, 4, 5, 7, 8, 10]
        
        # Blues scales
        elif scale_type == "blues":
            # Blues: W+H-W-H-H-W+H
            intervals = [0, 3, 5, 6, 7, 10]
        elif scale_type == "major blues":
            # Major Blues
            intervals = [0, 2, 3, 4, 7, 9]
        
        else:
            # Default to minor if unknown
            print(Fore.YELLOW + f"Warning: Unknown scale type '{scale_type}'. Using minor scale." + Style.RESET_ALL)
            intervals = [0, 2, 3, 5, 7, 8, 10] # Minor as fallback

        # Generate notes in playable range (e.g. +/- 1.5 octaves around root_note)
        min_note = max(0, root_note - 18)
        max_note = min(127, root_note + 18)
        scale = []
        
        # Generate notes across multiple octaves and then filter
        full_scale = []
        start_octave_offset = -24 # Start searching 2 octaves lower
        for octave in range(5): # Search across 5 octaves
             for interval in intervals:
                 note = root_note + start_octave_offset + interval + (octave * 12)
                 if 0 <= note <= 127:
                    full_scale.append(note)
        
        # Filter notes in desired range and remove duplicates
        scale = sorted(list(set([n for n in full_scale if min_note <= n <= max_note])))

        # Fallback if scale is empty
        if not scale:
            print(Fore.YELLOW + f"Warning: Could not generate scale notes in desired range for {scale_type}. Using default notes around root." + Style.RESET_ALL)
            scale = sorted(list(set([root_note + i for i in [-5, -3, 0, 2, 4, 5, 7] if 0 <= root_note + i <= 127])))
            if not scale: # Final fallback
                 scale = [60, 62, 64, 65, 67, 69, 71, 72]

        return scale
    except Exception as e:
        print(Fore.RED + f"Error in get_scale_notes: {str(e)}" + Style.RESET_ALL)
        # Return fallback scale
        return [60, 62, 64, 65, 67, 69, 71, 72] # C major as absolute fallback

def create_single_track_prompt(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], original_track_to_vary: Dict = None):
    """Creates a focused prompt for generating or varying a single instrument track."""
    total_beats = length * config["time_signature"]["beats_per_bar"]

    # --- Context for other tracks ---
    context_prompt_part = ""
    if context_tracks:
        context_json = json.dumps(context_tracks, indent=2)
        context_prompt_part = (
            "**Existing Tracks (Context):**\n"
            "The new part you compose MUST complement these already-written tracks.\n"
            f"```json\n{context_json}\n```\n"
        )
    
    # --- Context for the original track to be varied ---
    variation_prompt_part = ""
    if original_track_to_vary:
        original_track_json = json.dumps(original_track_to_vary, indent=2)
        variation_prompt_part = (
            f"Your primary task is to create a **variation** of the following original '{instrument_name}' track. Modify its rhythm, melody, or harmony, but the core musical idea should remain recognizable.\n"
            "**Original Track to Vary:**\n"
            f"```json\n{original_track_json}\n```\n"
        )
    else:
        # Instruction for creating an original track
        variation_prompt_part = (
            'Your primary goal is to create a new, original track.'
        )

    prompt = (
        f'You are an expert music producer specializing in {config["genre"]}.\n'
        f'The overall song context is: {config["bpm"]} BPM, in the key of {config["key_scale"]}, inspired by {config["inspiration"]}.\n\n'

        f'**Your Task:**\n{variation_prompt_part}\n\n'
        
        f'**Core Composition Principle:**\n'
        f'Compose a musically compelling part for the "{instrument_name}" that intelligently interacts with the existing tracks (if any) provided in the context. Follow these rules:\n'
        f'1. **Musical Conversation:** Analyze the phrases of other instruments. Create **call-and-response** patterns. Your part should either answer a phrase or pose a musical "question".\n'
        f'2. **Use of Silence (Rests):** Do not fill every single beat. Effective use of rests is critical for creating dynamics and tension. Let the track breathe.\n'
        f'3. **Complementary Parts:** Find ways to complement the existing rhythms and melodies. If one part is busy, create a simpler counter-melody, or vice-versa, so the parts lock into a cohesive whole.\n'
        f'4. **Harmonic Cohesion:** Ensure your notes are harmonically consistent with the key and the other instruments.\n\n'

        f'{context_prompt_part}'
        'Your output MUST be a single, valid JSON object. Do not include any other text or markdown formatting.\n\n'

        '**JSON Structure Specification:**\n'
        'The root object must contain:\n'
        '- `name`: (string) The name of the instrument track (e.g., "lead", "bass").\n'
        '- `program_num`: (int) The MIDI program number for the instrument.\n'
        '- `notes`: (array) An array of note objects.\n\n'
        
        'Each object inside the `notes` array MUST contain:\n'
        '- `pitch`: (int) The MIDI note number (0-127).\n'
        '- `time`: (float) The start time of the note in beats, from the beginning of the part.\n'
        '- `duration`: (float) The duration of the note in beats.\n'
        '- `volume`: (int) The velocity of the note (0-127).\n\n'
        
        f'**IMPORTANT CONSTRAINTS:**\n'
        f'- The musical part MUST be exactly {length} bars long.\n'
        f'- With a time signature of {config["time_signature"]["beats_per_bar"]}/4, this means the total duration is {total_beats} beats.\n'
        f'- All note `time` values must be within the range [0.0, {total_beats}).\n'
        f'- Ensure every note object has all four keys: `pitch`, `time`, `duration`, and `volume`.\n\n'

        'Begin the JSON output now.'
    )
    return prompt

def generate_instrument_track_data(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], original_track_to_vary: Dict = None) -> Dict:
    """Generates JSON data for a single instrument track, potentially as a variation."""
    max_attempts = 3
    attempt = 0
    
    prompt = create_single_track_prompt(config, length, instrument_name, program_num, context_tracks, original_track_to_vary)
    
    model = genai.GenerativeModel(
        model_name=config["model_name"],
        generation_config={
            "temperature": 0.8,
            "max_output_tokens": 65536,
            "response_mime_type": "application/json"
        }
    )

    while attempt < max_attempts:
        attempt += 1
        print(f"  Attempt {attempt}/{max_attempts} for {instrument_name}...")
        
        if attempt > 1:
            time.sleep(2)

        try:
            response = model.generate_content(
                contents=prompt,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            )

            if not response or not response.candidates:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    print(Fore.YELLOW + f"  Prompt for {instrument_name} was blocked. Reason: {response.prompt_feedback.block_reason.name}" + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + f"  Invalid or empty response for {instrument_name}." + Style.RESET_ALL)
                continue

            candidate = response.candidates[0]
            
            if candidate.finish_reason.name != "STOP":
                 print(Fore.YELLOW + f"  Generation stopped for {instrument_name}. Reason: {candidate.finish_reason.name}" + Style.RESET_ALL)
                 continue
            
            if not candidate.content or not candidate.content.parts:
                print(Fore.YELLOW + f"  Model returned a STOP signal but no content for {instrument_name}." + Style.RESET_ALL)
                continue

            generated_text = candidate.content.parts[0].text
            track_data = json.loads(generated_text)
            
            if "name" in track_data and "program_num" in track_data and "notes" in track_data:
                print(Fore.GREEN + f"  Successfully generated data for {instrument_name}." + Style.RESET_ALL)
                return track_data
            else:
                print(Fore.YELLOW + "  Generated JSON is missing required keys. Retrying..." + Style.RESET_ALL)
                continue

        except json.JSONDecodeError:
            print(Fore.RED + f"  Error: Model did not return valid JSON for {instrument_name}." + Style.RESET_ALL)
            continue
        except Exception as e:
            print(Fore.RED + f"  An unexpected error occurred for {instrument_name}: {str(e)}" + Style.RESET_ALL)
            continue
            
    return None

def generate_complete_track(config, length: int, is_variation: bool = False, original_song_data: Dict = None) -> Tuple[bool, Dict]:
    """Generates a complete track, handling original and variation logic."""
    try:
        genai.configure(api_key=config["api_key"])
        
        newly_generated_tracks = []
        config_instruments = config.get("instruments", {})
        
        # Determine generation order
        if "generation_order" in config and isinstance(config["generation_order"], list):
            instrument_order = config["generation_order"]
        else:
            instrument_order = list(config_instruments.keys())
        
        instruments_to_generate = []
        for name in instrument_order:
            if name in config_instruments:
                instrument_config = config_instruments[name]
                program_num = 0 # Default
                
                # Handle both simple (e.g., lead: 81) and complex (e.g., lead: {program_num: 81, ...}) configs
                if isinstance(instrument_config, dict):
                    program_num = instrument_config.get("program_num", 0)
                elif isinstance(instrument_config, int):
                    program_num = instrument_config
                
                if program_num > 0: # Add only if a valid program number is found
                    instruments_to_generate.append((name, program_num))

        print(Fore.BLUE + f"Starting {'variation' if is_variation else 'original'} track generation..." + Style.RESET_ALL)

        for instrument_name, program_num in instruments_to_generate:
            print(Fore.CYAN + f"\n--- Generating track for: {instrument_name.upper()} ---" + Style.RESET_ALL)
            
            original_track_to_vary = None
            if is_variation and original_song_data:
                # Find the original track to provide for variation context
                for track in original_song_data.get("tracks", []):
                    if track.get("name") == instrument_name:
                        original_track_to_vary = track
                        break

            # The context is always the set of tracks generated *in this run*.
            track_data = generate_instrument_track_data(config, length, instrument_name, program_num, newly_generated_tracks, original_track_to_vary)
            
            if not track_data:
                print(Fore.RED + f"Failed to generate data for {instrument_name} after multiple attempts. Aborting." + Style.RESET_ALL)
                return False, None
                
            newly_generated_tracks.append(track_data)
            
        final_song_data = {"tracks": newly_generated_tracks}
        
        print(Fore.BLUE + "\nAll track data generated. Creating final MIDI file..." + Style.RESET_ALL)
        if create_midi_from_json(final_song_data, config, f"complete_track_{length}bars.mid", is_variation):
            return True, final_song_data
        else:
            print(Fore.RED + "Failed to create the final MIDI file from combined data." + Style.RESET_ALL)
            return False, None

    except Exception as e:
        print(Fore.RED + f"Critical error in generate_complete_track: {str(e)}" + Style.RESET_ALL)
        return False, None

def create_midi_from_json(song_data: Dict, config: Dict, output_file: str, is_variation: bool = False) -> bool:
    """Creates a valid MIDI file from a JSON song structure."""
    try:
        tracks = song_data.get("tracks")
        if not tracks or not isinstance(tracks, list):
            print(Fore.RED + "Error: JSON must contain a list of 'tracks'." + Style.RESET_ALL)
            return False

        num_tracks = len(tracks)
        midi = MIDIFile(num_tracks)
        
        bpm = config["bpm"]
        
        for i, track_data in enumerate(tracks):
            # Use track index as channel, but reserve channel 9 (MIDI index) for drums
            # In midiutil, channels are 0-15
            is_drum_track = "drum" in track_data.get("name", "").lower()
            channel = 9 if is_drum_track else i if i < 9 else i + 1
            if channel > 15: channel = i % 9 # Fallback to avoid going over 15

            time = 0
            midi.addTrackName(i, time, track_data.get("name", f"Track {i+1}"))
            midi.addTempo(i, time, bpm)
            
            # For non-drum tracks, add the program change
            if not is_drum_track:
                program = track_data.get("program_num", 0)
                midi.addProgramChange(i, channel, time, program)

            notes = track_data.get("notes", [])
            if not notes:
                print(Fore.YELLOW + f"Warning: Track '{track_data.get('name')}' has no notes." + Style.RESET_ALL)
            
            for note in notes:
                # Flexible and robust parsing of note data
                pitch = note.get("pitch")
                if pitch is None:
                    pitch = note.get("midi_note")

                time_val = note.get("time")
                if time_val is None:
                    time_val = note.get("start_time")
                
                duration = note.get("duration")

                volume = note.get("volume")
                if volume is None:
                    volume = note.get("velocity")
                if volume is None:
                    volume = 100 # Default volume

                # Validate that we found all essential values after flexible parsing
                if not all(val is not None for val in [pitch, time_val, duration, volume]):
                    print(Fore.YELLOW + f"Skipping invalid note object (missing essential keys after parsing): {note}" + Style.RESET_ALL)
                    continue
                
                try:
                    midi.addNote(
                        track=i,
                        channel=channel,
                        pitch=int(pitch),
                        time=float(time_val),
                        duration=float(duration),
                        volume=int(volume)
                    )
                except (ValueError, TypeError) as e:
                    print(Fore.YELLOW + f"Skipping note with invalid data type in {note}: {e}" + Style.RESET_ALL)
                    continue

        final_filename = generate_filename(config, output_file, is_variation)
        with open(final_filename, 'wb') as outputfile_handle:
            midi.writeFile(outputfile_handle)

        if os.path.getsize(final_filename) > 0:
            print(Fore.GREEN + f"MIDI file '{final_filename}' successfully created." + Style.RESET_ALL)
            return True
        else:
            print(Fore.RED + "Generated MIDI file is empty (0 bytes)." + Style.RESET_ALL)
            try: os.remove(final_filename)
            except OSError: pass
            return False

    except Exception as e:
        print(Fore.RED + f"Critical error in create_midi_from_json: {str(e)}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return False

def generate_filename(config, base_filename: str, is_variation: bool = False) -> str:
    """Generates a unique, descriptive filename in the script's directory."""
    try:
        # Extract base name without .mid extension
        base_name = base_filename.rsplit('.mid', 1)[0]
        
        # Add timestamp for uniqueness
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive name
        parts = [
            base_name,
            config["genre"],
            f"{config['bpm']}bpm",
            config["key_scale"].replace(" ", "_"),
            timestamp
        ]
        
        # Add variation marker if needed
        if is_variation:
            parts.insert(1, "variation")
            
        # Combine everything into a filename
        filename_only = "_".join(parts) + ".mid"
        
        # Replace invalid characters
        filename_only = re.sub(r'[<>:"/\\|?*]', '_', filename_only)
        
        # Prepend the script's directory path to ensure it's saved in the right place
        full_path = os.path.join(script_dir, filename_only)
        
        return full_path
        
    except Exception as e:
        print(Fore.YELLOW + f"Warning during filename generation: {str(e)}" + Style.RESET_ALL)
        # Fallback: Use base filename with timestamp in the script's directory
        return os.path.join(script_dir, f"track_{time.strftime('%Y%m%d_%H%M%S')}.mid")

def get_tempo_character(bpm: int) -> str:
    """Characterizes tempo musically."""
    if bpm < 70:
        return "Largo/Adagio - slow and expressive"
    elif bpm < 90:
        return "Andante - walking pace"
    elif bpm < 120:
        return "Moderato - moderate tempo"
    elif bpm < 140:
        return "Allegro - quick and bright"
    elif bpm < 160:
        return "Vivace - lively and fast"
    else:
        return "Presto - very fast"

def main():
    """Main function for track generation."""
    try:
        config = load_config(CONFIG_FILE)
        is_variation = False
        original_song_data = None
        
        while True:
            if not is_variation:
                print("\nAvailable lengths (in bars):")
                for length_opt in AVAILABLE_LENGTHS:
                    print(f"- {length_opt}")
                
                while True:
                    try:
                        length = int(input("\nWhich length would you like to generate (in bars)? "))
                        if length in AVAILABLE_LENGTHS:
                            config['length'] = length
                            break
                        else:
                            print(Fore.RED + f"Please choose from available lengths: {', '.join(map(str, AVAILABLE_LENGTHS))}" + Style.RESET_ALL)
                    except ValueError:
                        print(Fore.RED + "Please enter a valid number." + Style.RESET_ALL)
            
            # Pass config WITH length to generate_complete_track
            success, generated_song_data = generate_complete_track(config, config['length'], is_variation, original_song_data)

            if success:
                # If this was the first, successful, original generation, store its data.
                if not is_variation:
                    original_song_data = generated_song_data
                
                print(Fore.GREEN + f"Track generation for {config['length']} bars completed successfully!" + Style.RESET_ALL)
                
                while True:
                    variation_choice = input("\nWould you like to create a variation of the original track? (y/n): ").lower()
                    if variation_choice in ['y', 'n']:
                        break
                    print(Fore.RED + "Please answer with 'y' for yes or 'n' for no." + Style.RESET_ALL)
                
                if variation_choice == 'n':
                    break
                else:
                    is_variation = True
            else:
                print(Fore.RED + f"Track generation for {config['length']} bars failed." + Style.RESET_ALL)
                break
            
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nProgram terminated by user." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Unexpected error: {str(e)}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 