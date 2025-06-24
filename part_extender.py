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
import mido
import glob
import sys

# --- ROBUST CONFIG FILE PATH ---
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join the script's directory with the config file name to create an absolute path
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
# --- END ROBUST PATH ---

BEATS_PER_BAR = 4
TICKS_PER_BEAT = 480

# Add new constants at the beginning of the file
AVAILABLE_LENGTHS = [8, 16, 32, 64, 128]
DEFAULT_LENGTH = 16

# Initialize Colorama for console color support
init(autoreset=True)

# Constants
GENERATED_CODE_FILE = os.path.join(script_dir, "generated_code.py")

def select_midi_file() -> str:
    """
    Scans for MIDI files in the current directory and subdirectories,
    and prompts the user to select one.
    """
    print(Fore.CYAN + "Searching for MIDI files..." + Style.RESET_ALL)
    
    # Scan for .mid and .midi files in the current directory and all subdirectories
    midi_files = glob.glob('**/*.mid', recursive=True) + glob.glob('**/*.midi', recursive=True)
    
    if not midi_files:
        print(Fore.RED + "No MIDI files found in the current directory or subdirectories. Please add a MIDI file to continue." + Style.RESET_ALL)
        sys.exit()

    print(Fore.GREEN + "Found the following MIDI files:" + Style.RESET_ALL)
    for i, file_path in enumerate(midi_files):
        print(f"  {i + 1}: {file_path}")

    while True:
        try:
            selection = input("Enter the number of the MIDI file you want to extend: ").strip()
            selection_index = int(selection) - 1
            if 0 <= selection_index < len(midi_files):
                selected_file = midi_files[selection_index]
                print(Fore.CYAN + f"You selected: {selected_file}" + Style.RESET_ALL)
                return selected_file
            else:
                print(Fore.YELLOW + "Invalid number. Please try again." + Style.RESET_ALL)
        except ValueError:
            print(Fore.YELLOW + "Invalid input. Please enter a number." + Style.RESET_ALL)
        except (KeyboardInterrupt, EOFError):
            print(Fore.RED + "\nSelection cancelled. Exiting." + Style.RESET_ALL)
            sys.exit()

def analyze_midi_file(file_path: str) -> Tuple[List[Dict], int, int, Dict]:
    """
    Analyzes a MIDI file to extract tracks, notes, tempo, and length.
    Returns a tuple of: (tracks_data, bpm, total_bars, time_signature)
    """
    try:
        midi_file = mido.MidiFile(file_path)
        ticks_per_beat = midi_file.ticks_per_beat or 480
        
        # --- Default values ---
        bpm = 120
        time_signature = {"beats_per_bar": 4, "beat_value": 4}

        # --- Find initial BPM and Time Signature ---
        for msg in midi_file.tracks[0]:
            if msg.is_meta and msg.type == 'set_tempo':
                bpm = mido.tempo2bpm(msg.tempo)
            if msg.is_meta and msg.type == 'time_signature':
                time_signature["beats_per_bar"] = msg.numerator
                time_signature["beat_value"] = msg.denominator
                # We assume the time signature stays constant for simplicity

        print(Fore.CYAN + f"Analyzed MIDI: BPM={bpm:.2f}, Time Signature={time_signature['beats_per_bar']}/{time_signature['beat_value']}" + Style.RESET_ALL)

        tracks_data = []
        total_ticks = 0
        
        for i, track in enumerate(midi_file.tracks):
            notes_on = {}
            current_time_ticks = 0
            track_notes = []
            track_name = f"Track {i + 1}"
            is_drum_track = False
            program = 0 # Default to Acoustic Grand Piano

            for msg in track:
                current_time_ticks += msg.time
                
                # Get Track Name
                if msg.is_meta and msg.type == 'track_name':
                    track_name = msg.name
                
                # Get program number (instrument)
                if msg.type == 'program_change':
                    program = msg.program

                # Check if it's a drum track (usually channel 9)
                if hasattr(msg, 'channel') and msg.channel == 9:
                    is_drum_track = True

                is_note_on = msg.type == 'note_on' and msg.velocity > 0
                is_note_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)

                if is_note_on:
                    notes_on[msg.note] = (current_time_ticks, msg.velocity)
                
                elif is_note_off and msg.note in notes_on:
                    start_ticks, velocity = notes_on.pop(msg.note)
                    duration_ticks = current_time_ticks - start_ticks
                    
                    track_notes.append({
                        "pitch": msg.note,
                        "start_beat": start_ticks / ticks_per_beat,
                        "duration_beats": duration_ticks / ticks_per_beat,
                        "velocity": velocity
                    })
            
            if track_notes:
                # Assign a role to help with MIDI channel assignment later
                role = "drums" if is_drum_track else "context"
                tracks_data.append({
                    "instrument_name": track_name,
                    "program_num": program,
                    "role": role,
                    "notes": track_notes
                })
            
            # Update total length of the piece
            if current_time_ticks > total_ticks:
                total_ticks = current_time_ticks

        if total_ticks == 0:
            print(Fore.YELLOW + "Warning: MIDI file seems to be empty or contains no note events." + Style.RESET_ALL)
            return [], 120, 0, time_signature

        total_beats = total_ticks / ticks_per_beat
        total_bars = math.ceil(total_beats / time_signature["beats_per_bar"])

        return tracks_data, bpm, total_bars, time_signature

    except Exception as e:
        print(Fore.RED + f"Error analyzing MIDI file: {str(e)}" + Style.RESET_ALL)
        return [], 120, 0, {"beats_per_bar": 4, "beat_value": 4}

def load_config(config_file):
    """Loads and validates the configuration from a YAML file."""
    print(Fore.CYAN + "Loading configuration..." + Style.RESET_ALL)
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Validate critical fields first
        if not config.get("api_key"):
            raise ValueError("API key is missing in configuration.")
            
        if not config.get("model_name"):
            raise ValueError("Model name is missing in configuration.")
            
        # Validate API-Key Format (basic check)
        if not isinstance(config["api_key"], str) or len(config["api_key"]) < 10:
            raise ValueError("API key appears to be invalid.")

        # A list of required fields for the configuration to be valid.
        required_fields = [
            "inspiration", 
            "genre", 
            "bpm", 
            "key_scale",
            "api_key", 
            "model_name", 
            "instruments", 
            "time_signature"
        ]
        
        # Add use_call_and_response validation
        if "use_call_and_response" not in config:
            config["use_call_and_response"] = 0 # Default to off
        elif config["use_call_and_response"] not in [0, 1]:
            raise ValueError(f"Invalid use_call_and_response: '{config['use_call_and_response']}'. Must be 0 or 1.")
        
        # Add number_of_iterations validation
        if "number_of_iterations" not in config:
            config["number_of_iterations"] = 1
        else:
            try:
                iterations = int(config["number_of_iterations"])
                if iterations < 1:
                    raise ValueError("number_of_iterations must be at least 1.")
                config["number_of_iterations"] = iterations
            except (ValueError, TypeError):
                 raise ValueError("number_of_iterations must be a valid integer.")

        # Add temperature validation
        if "temperature" not in config:
            config["temperature"] = 1.0 # Default creativity
        else:
            try:
                temp = float(config["temperature"])
                if not 0.0 <= temp <= 2.0:
                    raise ValueError("temperature must be between 0.0 and 2.0.")
                config["temperature"] = temp
            except (ValueError, TypeError):
                raise ValueError("temperature must be a valid number between 0.0 and 2.0.")

        # Check for the presence of all required fields.
        for field in required_fields:
            if field not in config:
                if field == "time_signature":
                    # Provide a default time signature if it's missing.
                    config["time_signature"] = {
                        "beats_per_bar": 4,
                        "beat_value": 4
                    }
                else:
                    raise ValueError(f"Error: Required field '{field}' is missing in the configuration.")

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
                # Provide a default time signature if parts are missing.
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
                # Fallback to a default if parsing fails.
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
    """
    Generates a list of MIDI note numbers for a given root note and scale type.
    This function supports a wide variety of musical scales and modes.
    """
    try:
        # Ensure root_note is a number, converting if necessary.
        root_note = int(root_note) if isinstance(root_note, (str, tuple)) else root_note
        
        # A dictionary mapping scale names to their interval patterns (in semitones).
        scale_intervals = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "ionian": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "natural minor": [0, 2, 3, 5, 7, 8, 10],
            "aeolian": [0, 2, 3, 5, 7, 8, 10],
            "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
            "melodic minor": [0, 2, 3, 5, 7, 9, 11],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "locrian": [0, 1, 3, 5, 6, 8, 10],
            "major pentatonic": [0, 2, 4, 7, 9],
            "minor pentatonic": [0, 3, 5, 7, 10],
            "chromatic": list(range(12)),
            "whole tone": [0, 2, 4, 6, 8, 10],
            "diminished": [0, 1, 3, 4, 6, 7, 9, 10],
            "augmented": [0, 3, 4, 7, 8, 11],
            "byzantine": [0, 1, 4, 5, 7, 8, 11],
            "hungarian minor": [0, 2, 3, 6, 7, 8, 11],
            "persian": [0, 1, 4, 5, 6, 8, 11],
            "arabic": [0, 2, 3, 6, 7, 8, 11],
            "jewish": [0, 1, 4, 5, 7, 8, 10],
            "ahava raba": [0, 1, 4, 5, 7, 8, 10],
            "blues": [0, 3, 5, 6, 7, 10],
            "major blues": [0, 2, 3, 4, 7, 9]
        }
        
        # Get the intervals for the specified scale type.
        intervals = scale_intervals.get(scale_type.lower())

        if intervals is None:
            # Default to minor if unknown
            print(Fore.YELLOW + f"Warning: Unknown scale type '{scale_type}'. Using minor scale." + Style.RESET_ALL)
            intervals = scale_intervals["minor"]

        # Generate notes in playable range (e.g. +/- 1.5 octaves around root_note)
        min_note = max(0, root_note - 18)
        max_note = min(127, root_note + 18)
        
        # Generate notes across multiple octaves and then filter.
        full_scale = []
        start_octave_offset = -24 # Start searching 2 octaves lower
        for octave in range(5): # Search across 5 octaves
            for interval in intervals:
                 note = root_note + start_octave_offset + interval + (octave * 12)
                 if 0 <= note <= 127:
                    full_scale.append(note)
        
        # Filter notes in desired range and remove duplicates.
        scale = sorted(list(set([n for n in full_scale if min_note <= n <= max_note])))

        # Fallback if scale is empty
        if not scale:
            print(Fore.YELLOW + f"Warning: Could not generate scale notes in the desired range for {scale_type}. Using default notes around the root." + Style.RESET_ALL)
            scale = sorted(list(set([root_note + i for i in [-5, -3, 0, 2, 4, 5, 7] if 0 <= root_note + i <= 127])))
            if not scale: # Final fallback
                scale = [60, 62, 64, 65, 67, 69, 71] # C Major as a last resort
        return scale
    except Exception as e:
        print(Fore.RED + f"Error generating scale for {scale_type}: {str(e)}" + Style.RESET_ALL)
        return [60, 62, 64, 65, 67, 69, 71] # Return C Major scale on error

def create_single_track_prompt(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int, dialogue_role: str):
    """
    Creates a universal, music-intelligent prompt that is tailored 
    to the role and position in the arrangement.
    """
    total_beats = length * config["time_signature"]["beats_per_bar"]
    scale_notes = get_scale_notes(config["root_note"], config["scale_type"])
    
    # --- Basic music theory instructions ---
    basic_instructions = (
        f"**Genre:** {config['genre']}\n"
        f"**Tempo:** {config['bpm']} BPM\n"
        f"**Time Signature:** {config['time_signature']['beats_per_bar']}/{config['time_signature']['beat_value']}\n"
        f"**Key/Scale:** {config['key_scale'].title()} (Available notes: {scale_notes})\n"
        f"**Track Length:** {length} bars ({total_beats} beats total)\n"
        f"**Instrument:** {instrument_name} (MIDI Program: {program_num})\n"
    )

    # --- Context for other tracks ---
    context_prompt_part = ""
    if context_tracks:
        context_prompt_part = "**Listen to the existing tracks for context. Here is the actual note data for each part:**\n"
        for track in context_tracks:
            # Convert notes to a compact JSON string for the prompt
            notes_as_str = json.dumps(track['notes'])
            context_prompt_part += f"- **{track['instrument_name']}** (Role: {track['role']}):\n```json\n{notes_as_str}\n```\n"
        context_prompt_part += "\n"

    # --- Call and Response Instruction ---
    call_and_response_instructions = ""
    if dialogue_role == 'call':
        call_and_response_instructions = (
            "**Special Instruction: You are the 'Call' in a Dialogue**\n"
            "You are the first part in a musical conversation. Your primary goal is to create a clear, catchy musical phrase (a 'call') and then **intentionally leave space (rests)** for another instrument to answer. "
            "Think of it as asking a musical question. Don't fill all the space. Your phrasing should invite a response.\n\n"
        )
    elif dialogue_role == 'response':
        call_and_response_instructions = (
            "**Special Instruction: You are the 'Response' in a Dialogue**\n"
            "You are part of a musical conversation. Another instrument has just played a 'call'. Listen carefully to its phrase and rhythm. Your primary goal is to **play your part in the spaces the other instrument left behind**. "
            "Think of it as giving a musical answer. Your phrases should complement and respond directly to the 'call'.\n\n"
        )

    # --- NEW: Simple positional context ---
    positional_context_instruction = f"**Your Position:** You are creating track {current_track_index + 1} of {total_tracks} in this arrangement.\n"

    # --- Role-specific instructions ---
    role_instructions = ""
    if role == "drums":
        role_instructions = (
            "**Your Role: The Rhythmic Foundation**\n"
            "1. **Foundation First:** Establish a strong, clear rhythmic backbone. The kick provides the main pulse, the snare defines the backbeat.\n"
            "2. **Add Complexity:** Use hi-hats or other percussion for faster subdivisions (e.g., 16th notes) to create energy and drive.\n"
            "3. **Groove is Key:** Use subtle shifts in timing (micro-timing), velocity, and strategic rests to make the pattern feel human and groovy, not robotic."
        )
    elif role == "percussion":
        role_instructions = (
            "**Your Role: Rhythmic Spice & Texture**\n"
            "1. **No Kick/Snare:** This part is for secondary percussion (e.g., bongos, congas, shakers, tambourines). Do not use typical kick and snare sounds.\n"
            "2. **Complement the Groove:** Create a syncopated, often looping pattern that weaves in and out of the main drum beat. Your goal is to add complexity and groove.\n"
            "3. **Fill the Gaps:** Listen for empty spaces in the main drum pattern and place your percussive hits there."
        )
    elif role == "sub_bass":
        role_instructions = (
            "**Your Role: The Low-End Foundation**\n"
            "1. **Pure Weight:** Your only job is to provide deep, low-end frequency support. Use very low notes (e.g., below MIDI note 40).\n"
            "2. **Keep it Simple:** The rhythm must be extremely simple, often just following the root note of the chord on each downbeat. This part is meant to be felt more than heard.\n"
            "3. **No Melody:** Avoid any complex melodic movement. Your part must not interfere with the main bassline."
        )
    elif role == "bass":
        role_instructions = (
            "**Your Role: The Rhythmic & Harmonic Anchor**\n"
            "1. **Lock with the Kick:** Your primary goal is to create a rhythmic and harmonic lock with the kick drum. Either reinforce the kick's rhythm or create a complementary syncopated pattern that plays off of it.\n"
            "2. **Harmonic Foundation:** Outline the harmony. Use root notes, fifths, or simple arpeggios that define the chords.\n"
            "3. **Space Creates Groove:** A great bassline uses rests effectively to create a powerful, groovy rhythm. Do not fill every beat."
        )
    elif role == "pads":
        role_instructions = (
            "**Your Role: The Harmonic Glue**\n"
            "1. **Sustained Harmony:** Your primary function is to provide a lush, sustained harmonic foundation. Use long, overlapping chords that hold the track together.\n"
            "2. **Slow & Smooth:** Notes should typically be very long (4+ beats). The harmonic changes should be gentle and evolve over many bars.\n"
            "3. **Fill the Background:** Think of your part as the 'bed' or 'cushion' on which the other instruments sit. You are creating the main emotional tone of the song."
        )
    elif role == "atmosphere":
        role_instructions = (
            "**Your Role: The Environment & Mood**\n"
            "1. **Paint a Picture:** Your goal is to create a specific mood or a sense of place (e.g., 'dark', 'dreamy', 'aquatic', 'industrial'). This is more abstract than simple chords.\n"
            "2. **Evolving Soundscapes:** Use sound that changes and evolves over time. This can be tonal or atonal. Think of long, shifting textures, not a repeating melody or rhythm.\n"
            "3. **Stay Out of the Way:** Your part should sit in the background and create a context for the other instruments without drawing too much attention to itself. It adds depth and a professional sheen."
        )
    elif role == "texture":
        role_instructions = (
            "**Your Role: Sonic Detail & 'Ear Candy'**\n"
            "1. **Add Subtle Details & Interest:** Your purpose is to add subtle sonic details that make the track more interesting. This is not about harmony or melody.\n"
            "2. **Subtle Rhythms or Atonal Accents:** Consider adding a quiet, high-frequency rhythmic element (like a shaker or light clicks) or subtle, atonal percussive sounds. These are often quiet elements.\n"
            "3. **Repetitive & Subtle:** Often, textural parts are short, repetitive loops that are low in the mix. They add a professional polish and complexity without cluttering the main arrangement."
        )
    elif role == "chords":
        role_instructions = (
            "**Your Role: The Harmonic Core**\n"
            "1. **Provide Harmony:** Your main purpose is to establish the song's chord progression. Use full, rich chords that clearly define the harmony.\n"
            "2. **Rhythmic Pulse vs. Sustained Pads:** Decide on your function. You can either create a rhythmic pulse with short, stabbed chords (like in House or Funk) or provide a sustained, atmospheric layer with long, evolving chords (like in Trance or Ambient).\n"
            "3. **Voice Leading:** Pay attention to how the notes within the chords move from one chord to the next. Smooth transitions (good voice leading) make the progression sound more natural and professional."
        )
    elif role == "arp":
        role_instructions = (
            "**Your Role: The Hypnotic Arpeggio**\n"
            "1. **Chord Notes Only:** Create a sequence by playing the notes of the underlying chords one after another.\n"
            "2. **Rhythmic & Repetitive:** Establish a clear, driving, and repetitive rhythmic pattern (e.g., using straight 16th notes). This pattern should create a hypnotic effect.\n"
            "3. **Evolve Slowly:** You can subtly vary the pattern over time by changing the order of notes or adding/removing a note, but the core rhythm should remain consistent."
        )
    elif role == "guitar":
        role_instructions = (
            "**Your Role: The Versatile Riff-Machine**\n"
            "A guitar can have many functions. Choose one that fits the track:\n"
            "1. **Rhythmic Strumming:** Create a strummed chord pattern that provides rhythmic and harmonic texture. This works well to complement the drums.\n"
            "2. **Melodic Riff/Lick:** Compose a short, catchy guitar riff that acts like a secondary melody or a response to the main lead.\n"
            "3. **Power Chords:** For heavier genres, use simple, powerful two-note chords (root and fifth) to create a wall of energy."
        )
    elif role in ["lead", "melody"]:
        role_instructions = (
            "**Your Role: The Main Hook / Storyteller**\n"
            "1. **Create a Memorable Motif:** Compose a short, catchy melodic phrase (a 'hook' or 'motif'). This is the central idea of the part.\n"
            "2. **Develop the Idea:** Don't just repeat the motif. Develop it over the section by changing its rhythm, transposing it, or adding small variations.\n"
            "3. **Clear Phrasing with Rests:** Structure your melody with clear beginnings and endings. Rests between phrases are essential to make the melody understandable and memorable."
        )
    elif role == "vocal":
        role_instructions = (
            "**Your Role: The Vocal Element**\n"
            "A vocal part can serve different functions. Choose one that best fits the existing music:\n"
            "1. **Lead Vocal Melody:** Create a clear, memorable, and singable melody, like a main vocal line. This should function like a 'lead' instrument, with clear phrasing, emotional contour, and rests between phrases.\n"
            "2. **Rhythmic Vocal Chops:** Create short, catchy, rhythmic phrases, like sampled vocal chops. Here, the focus is on syncopation and using short, often repetitive notes to create a percussive, groovy texture."
        )
    elif role == "fx":
        role_instructions = (
            "**Your Role: Sound Effects & Ear Candy**\n"
            "1. **Non-Melodic Focus:** Your goal is to add interest, not melody. Use atonal sounds, short percussive 'blips', or unique sound effects that can be triggered by a single MIDI note.\n"
            "2. **Rhythmic Accents:** Place these sounds sparingly to accent specific moments, like the beginning of a phrase or a transition between sections.\n"
            "3. **Create Surprise:** Use unpredictable timing. These sounds should catch the listener's ear without disrupting the main groove."
        )
    else: # Fallback for "complementary" or unknown roles
        role_instructions = (
            "**Your Role: A Complementary Part**\n"
            "Listen to the other instruments. Find a sonic niche and a rhythmic pattern that is currently unoccupied. Your goal is to add a new layer that enhances the overall composition without making it sound cluttered."
        )

    # --- NEW: Add a specific drum map for drum/percussion roles ---
    drum_map_instructions = ""
    if role in ["drums", "percussion"]:
        drum_map_instructions = (
            "**Drum Map Guidance (Addictive Drums 2 Standard):**\n"
            "You MUST use the following MIDI notes for the corresponding drum sounds. This is not a suggestion, but a requirement for this track.\n"
            "- **Kick:** MIDI Note 36\n"
            "- **Snare (Center Hit):** MIDI Note 38\n"
            "- **Snare (Rimshot):** MIDI Note 40\n"
            "- **Hi-Hat (Closed):** MIDI Note 42\n"
            "- **Hi-Hat (Open):** MIDI Note 46\n"
            "- **Hi-Hat (Pedal Close):** MIDI Note 44\n"
            "- **Crash Cymbal 1:** MIDI Note 49\n"
            "- **Ride Cymbal 1:** MIDI Note 51\n"
            "- **High Tom:** MIDI Note 50\n"
            "- **Mid Tom:** MIDI Note 48\n"
            "- **Low Tom:** MIDI Note 45\n\n"
        )

    # --- The final prompt that puts it all together ---
    # Define roles for different polyphony rules
    POLYPHONIC_ROLES = {"harmony", "chords", "pads", "atmosphere", "texture", "guitar"}
    EXPRESSIVE_MONOPHONIC_ROLES = {"lead", "melody", "vocal"}
    
    if role in POLYPHONIC_ROLES:
        polyphony_rule = "2.  **Polyphonic:** Notes for this track CAN overlap. This is crucial for playing chords and creating rich textures. The JSON array can contain note objects whose start and end times overlap."
    elif role in EXPRESSIVE_MONOPHONIC_ROLES:
        polyphony_rule = "2.  **Expressive Monophonic:** Notes should primarily be played one at a time. However, very short, slight overlaps are PERMITTED to create a natural, legato-style performance. Do not create full, sustained chords."
    else: # Strictly Monophonic roles like bass, arp, drums
        polyphony_rule = "2.  **Strictly Monophonic:** The notes in the JSON array must NOT overlap in time. A new note can only start after the previous one has finished."

    # --- NEW: Conditional "Stay in Key" rule ---
    # For drums, this rule is irrelevant and confusing.
    stay_in_key_rule = f"3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n"
    if role in ["drums", "percussion"]:
        stay_in_key_rule = "3.  **Use Drum Map:** You must adhere to the provided Drum Map for all note pitches.\n"

    prompt = (
        f'You are an expert music producer creating a track inspired by: **{config["inspiration"]}**.\n'
        f"Your task is to compose a single instrument track for this production.\n\n"
        f"**--- MUSICAL CONTEXT ---**\n"
        f"{basic_instructions}\n"
        f"{context_prompt_part}"
        f"**--- YOUR TASK ---**\n"
        f"{positional_context_instruction}"
        f"{call_and_response_instructions}"
        f"{drum_map_instructions}"
        f"{role_instructions}\n\n"
        f"**--- UNIVERSAL PRINCIPLES OF GOOD MUSIC ---**\n"
        f"1. **Structure & Evolution:** Your composition should have a clear structure that makes sense for the specified genre and inspiration. A good musical part tells a story over the full {length} bars. Develop your ideas logically, avoiding mindless, robotic repetition. Let the provided inspiration guide whether the part should be consistent and evolving or have more distinct sections.\n"
        f"2. **Clarity through Space:** Do not create a constant wall of sound. The musical role of a part determines how it should use space and silence. Your role-specific instructions provide guidance on this.\n"
        f"3. **Dynamic Phrasing:** Use a wide range of velocity to create accents and shape the energy of the phrase. A static volume is boring.\n"
        f"4. **Tension & Release:** Build musical tension through dynamics, rhythmic complexity, or harmony, and resolve it at key moments (e.g., at the end of 8 or 16 bar phrases) to create a satisfying arc.\n"
        f"5. **Ensemble Playing:** Think like a member of a band. Your performance must complement the existing parts. Pay attention to the phrasing of other instruments and find pockets of space to add your musical statement without cluttering the arrangement.\n"
        f"6. **Micro-timing for Groove:** To add a human feel, you can subtly shift notes off the strict grid. Slightly anticipating a beat can add urgency, while slightly delaying it can create a more relaxed feel. This is especially effective for non-kick/snare elements.\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON array of objects. Each object represents a note and MUST have these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127, from the provided scale notes).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n\n'
        f"**IMPORTANT RULES:**\n"
        f'1.  **JSON ONLY:** Your entire response MUST be only the raw JSON array. DO NOT include ```json ... ```, explanations, or any other text.\n'
        f"{polyphony_rule}\n"
        f"{stay_in_key_rule}"
        f"4.  **Timing is Absolute:** 'start_beat' is the absolute position from the beginning of the {length}-bar clip.\n"
        f"5.  **Be Creative:** Compose a musically interesting and high-quality part that fits the description.\n"
        f'6.  **Valid JSON Syntax:** The output must be a perfectly valid JSON array. Use double quotes for all keys and string values. Ensure no trailing commas.\n\n'
        f"Now, generate the JSON array for the **{instrument_name}** track.\n"
    )
    return prompt

def generate_instrument_track_data(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int, dialogue_role: str) -> Dict:
    """
    Generates musical data for a single instrument track using the generative AI model.
    It includes a retry mechanism for robustness and expects a JSON response.
    """
    prompt = create_single_track_prompt(config, length, instrument_name, program_num, context_tracks, role, current_track_index, total_tracks, dialogue_role)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(Fore.BLUE + f"Attempt {attempt + 1}/{max_retries}: Generating part for {instrument_name} ({role})..." + Style.RESET_ALL)
            
            generation_config = {
                "temperature": config["temperature"],
                "response_mime_type": "application/json",
                "max_output_tokens": 65536
            }

            model = genai.GenerativeModel(
                model_name=config["model_name"],
                generation_config=generation_config
            )

            # Define safety settings to avoid blocking responses unnecessarily
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = model.generate_content(
                prompt, 
                safety_settings=safety_settings
            )
            
            # --- Robust Response Validation ---
            # 1. Check if the response was blocked or is otherwise invalid before accessing .text
            if not response.candidates or response.candidates[0].finish_reason not in [1, "STOP"]: # 1 is the enum for STOP
                finish_reason_name = "UNKNOWN"
                # Safely get the finish reason name
                if response.candidates:
                    try:
                        finish_reason_name = response.candidates[0].finish_reason.name
                    except AttributeError:
                        finish_reason_name = str(response.candidates[0].finish_reason)

                print(Fore.RED + f"Error on attempt {attempt + 1}: Generation failed or was incomplete." + Style.RESET_ALL)
                print(Fore.YELLOW + f"Reason: {finish_reason_name}" + Style.RESET_ALL)
                
                # Also check for safety blocking information
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     print(Fore.YELLOW + f"Block Reason: {response.prompt_feedback.block_reason.name}" + Style.RESET_ALL)
                
                continue # Skip to the next retry attempt

            # 2. Now that we know the response is valid, safely access the text and parse JSON
            response_text = response.text
            if not response_text.strip():
                print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Model returned an empty response for {instrument_name}." + Style.RESET_ALL)
                continue # Skip to the next retry attempt

            # 3. Parse the JSON response
            notes_list = json.loads(response_text)

            if not isinstance(notes_list, list):
                raise TypeError("The generated data is not a valid list of notes.")

            # --- Data Validation ---
            validated_notes = []
            for note in notes_list:
                if not all(k in note for k in ["pitch", "start_beat", "duration_beats", "velocity"]):
                     print(Fore.YELLOW + f"Warning: Skipping invalid note object: {note}" + Style.RESET_ALL)
                     continue
                validated_notes.append(note)

            print(Fore.GREEN + f"Successfully generated and validated part for {instrument_name}." + Style.RESET_ALL)
            return {
                "instrument_name": instrument_name,
                "program_num": program_num,
                "role": role,
                "notes": validated_notes
            }

        except (json.JSONDecodeError, TypeError) as e:
            print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Data validation failed for {instrument_name}. Reason: {str(e)}" + Style.RESET_ALL)
            # We already checked for blocking, so we just show the text if parsing fails.
            if "response_text" in locals():
                print(Fore.YELLOW + "Model response was:\n" + response_text + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"An unexpected error occurred on attempt {attempt + 1} for {instrument_name}: {str(e)}" + Style.RESET_ALL)

        # If we are not on the last attempt, wait before retrying
        if attempt < max_retries - 1:
            print(Fore.YELLOW + "Waiting for 3 seconds before retrying..." + Style.RESET_ALL)
            time.sleep(3)

    print(Fore.RED + f"Failed to generate a valid part for {instrument_name} after {max_retries} attempts." + Style.RESET_ALL)
    return None

def extend_track(config: Dict, length_bars: int, base_tracks: List[Dict]) -> Tuple[bool, Dict]:
    """
    Extends an existing set of tracks by generating new parts for the instruments defined in the config.
    Returns a tuple of (success_status, song_data).
    """
    print(Fore.CYAN + "\n--- Starting Track Extension ---" + Style.RESET_ALL)
    
    # The initial song data includes the tracks from the original MIDI file.
    song_data = {
        "bpm": config["bpm"],
        "time_signature": config["time_signature"],
        "key_scale": config["key_scale"],
        "tracks": list(base_tracks) # Start with a copy of the base tracks
    }
    
    # The context for the AI is also the original tracks.
    context_tracks = list(base_tracks)
    total_new_tracks = len(config["instruments"])
    call_has_been_made = False
    CALL_AND_RESPONSE_ROLES = {'bass', 'chords', 'arp', 'guitar', 'lead', 'melody', 'vocal'}

    for i, instrument in enumerate(config["instruments"]):
        instrument_name = instrument["name"]
        program_num = instrument["program_num"]
        role = instrument.get("role", "complementary")
        dialogue_role = 'none'

        if config.get("use_call_and_response") == 1 and role in CALL_AND_RESPONSE_ROLES:
            if not call_has_been_made:
                dialogue_role = 'call'
                call_has_been_made = True
            else:
                dialogue_role = 'response'

        print(Fore.MAGENTA + f"\n--- Generating New Track {i + 1}/{total_new_tracks}: {instrument_name} ---" + Style.RESET_ALL)
        
        # The AI gets context from all previously existing AND newly generated tracks.
        track_data = generate_instrument_track_data(config, length_bars, instrument_name, program_num, context_tracks, role, len(context_tracks), len(context_tracks) + total_new_tracks - i, dialogue_role)

        if track_data:
            song_data["tracks"].append(track_data)
            # Add the newly created track to the context for the next generation.
            context_tracks.append(track_data)
            time.sleep(2)
        else:
            print(Fore.RED + f"Failed to generate track for {instrument_name}. Stopping generation." + Style.RESET_ALL)
            return False, None

    if not song_data["tracks"]:
        print(Fore.RED + "No tracks were generated successfully. Aborting." + Style.RESET_ALL)
        return False, None

    print(Fore.GREEN + "\n--- All new tracks generated successfully! ---" + Style.RESET_ALL)
    return True, song_data

def create_midi_from_json(song_data: Dict, config: Dict, output_file: str) -> bool:
    """
    Creates a MIDI file from the generated song data structure.
    """
    try:
        bpm = config["bpm"]
        time_signature_beats = config["time_signature"]["beats_per_bar"]
        
        # Initialize MIDIFile with the number of tracks
        num_tracks = len(song_data["tracks"])
        midi_file = MIDIFile(num_tracks, removeDuplicates=True, deinterleave=False)
        
        # Add tempo and time signature to the first track
        midi_file.addTempo(track=0, time=0, tempo=bpm)
        midi_file.addTimeSignature(track=0, time=0, numerator=time_signature_beats, denominator=4, clocks_per_tick=24)

        # Smart channel assignment
        # Channel 10 (index 9) is for percussion. Non-drum channels will be assigned from 0, skipping 9.
        next_melodic_channel = 0
        for i, track_data in enumerate(song_data["tracks"]):
            track_name = track_data["instrument_name"]
            program_num = track_data["program_num"]
            role = track_data.get("role", "complementary")
            
            # Assign MIDI channel
            if role in ["drums", "percussion"]:
                channel = 9 # MIDI Channel 10 for drums
            else:
                channel = next_melodic_channel
                if channel == 9: # Skip the drum channel
                    next_melodic_channel += 1
                    channel = next_melodic_channel
                next_melodic_channel += 1

            # Fallback if we somehow run out of channels
            if channel > 15: channel = 15
            
            midi_file.addTrackName(i, 0, track_name)
            midi_file.addProgramChange(i, channel, 0, program_num)
            
            for note in track_data["notes"]:
                try:
                    pitch = int(note["pitch"])
                    start_beat = float(note["start_beat"])
                    duration_beats = float(note["duration_beats"])
                    velocity = int(note["velocity"])
                    
                    # Ensure values are within MIDI specs
                    if 0 <= pitch <= 127 and 1 <= velocity <= 127 and duration_beats > 0:
                        midi_file.addNote(
                            track=i,
                            channel=channel,
                            pitch=pitch,
                            time=start_beat,
                            duration=duration_beats,
                            volume=velocity
                        )
                except (ValueError, TypeError) as e:
                    print(Fore.YELLOW + f"Warning: Skipping invalid note data in track '{track_name}': {note}. Reason: {e}" + Style.RESET_ALL)

        # Write the MIDI file
        with open(output_file, "wb") as f:
            midi_file.writeFile(f)
            
        print(Fore.GREEN + f"\nSuccessfully created MIDI file: {output_file}" + Style.RESET_ALL)
        return True

    except Exception as e:
        print(Fore.RED + f"Error creating MIDI file: {str(e)}" + Style.RESET_ALL)
        return False

def generate_filename(input_midi_path: str, iteration: int) -> str:
    """
    Generates a clear filename for an extended MIDI file based on the original name.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        original_base_name = os.path.splitext(os.path.basename(input_midi_path))[0]
        sanitized_base_name = re.sub(r'[\\/*?:"<>|]', "", original_base_name)

        new_name = f"{sanitized_base_name}_ext_{iteration}.mid"

        return os.path.join(script_dir, new_name)
    except Exception as e:
        print(Fore.YELLOW + f"Could not generate dynamic filename. Using default. Reason: {e}" + Style.RESET_ALL)
        return os.path.join(script_dir, f"extended_{iteration}.mid") # Fallback

def main():
    """
    Main function to run the music extension process.
    """
    try:
        config = load_config(CONFIG_FILE)
        
        genai.configure(api_key=config["api_key"])
        
        # --- Initial MIDI File Selection ---
        input_midi_file = select_midi_file()
        if not input_midi_file:
            return

        # --- Analyze the original MIDI file once ---
        original_tracks, original_bpm, length_in_bars, original_ts = analyze_midi_file(input_midi_file)
        
        if not original_tracks:
            print(Fore.RED + f"Could not process MIDI file '{input_midi_file}'. Exiting." + Style.RESET_ALL)
            return
            
        # --- Override config with original MIDI data ---
        config["bpm"] = original_bpm
        config["time_signature"] = original_ts
        print(Fore.CYAN + f"Using BPM and Time Signature from MIDI. Generating {length_in_bars} bars." + Style.RESET_ALL)

        # --- Generation Loop ---
        num_iterations = config.get("number_of_iterations", 1)
        for i in range(num_iterations):
            print(Fore.CYAN + f"\n--- Starting Generation Cycle {i + 1}/{num_iterations} ---" + Style.RESET_ALL)
            
            # --- Extend the track (always based on the original) ---
            success, song_data = extend_track(config, length_in_bars, original_tracks)
            
            if success:
                # Generate a unique filename for each extension
                output_filename = generate_filename(input_midi_file, i + 1)
                
                # Create the MIDI file
                create_midi_from_json(song_data, config, output_filename)

    except (ValueError, FileNotFoundError) as e:
        print(Fore.RED + f"A critical error occurred: {str(e)}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {str(e)}" + Style.RESET_ALL)

if __name__ == "__main__":
    main() 