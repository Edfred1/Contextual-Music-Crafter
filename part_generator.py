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

def create_single_track_prompt(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int):
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
        context_prompt_part = "**Listen to the existing tracks for context:**\n"
        for track in context_tracks:
            context_prompt_part += f"- {track['instrument_name']}: Plays a {track['role']} part.\n"
        context_prompt_part += "\n"

    # --- NEW: Instructions based on position in arrangement ---
    generation_stage_instructions = ""
    # We define "early" tracks as the first half of the arrangement
    if current_track_index < (total_tracks / 2):
        generation_stage_instructions = (
            "**Generation Stage: Early (Foundation)**\n"
            "You are one of the first instruments. Your part should be foundational, establishing the core harmony, rhythm, or melody. Keep it compelling but leave space for others. Don't be too busy."
        )
    else:
        generation_stage_instructions = (
            "**Generation Stage: Late (Details & Overdubs)**\n"
            "The foundation is laid. Your role is to add detail, texture, or counter-melodies. Listen carefully to what's already there and find the gaps to fill. You can be more complex or sparse, as long as you complement the existing parts."
        )

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
    elif role in ["pads", "atmosphere", "texture"]:
        role_instructions = (
            "**Your Role: The Atmosphere & Space**\n"
            "1. **Create Texture:** Your purpose is to fill the sonic space and create an emotional atmosphere. Use long, sustained notes (e.g., 4+ beats) that can overlap to create a smooth, continuous soundscape.\n"
            "2. **Slow Evolution:** Avoid sharp, rhythmic elements. The harmonic changes should be slow and subtle, evolving gradually over the entire section.\n"
            "3. **The Exception to the 'Rests' Rule:** For this role, continuous sound is often desired. 'Space' is created by evolving textures and slow changes, not by frequent silence."
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
            "1. **Non-Melodic Focus:** Your goal is to add interest, not melody. Use atonal sounds, pitch sweeps (risers/downlifters), or short, percussive 'blips'.\n"
            "2. **Rhythmic Accents:** Place these sounds sparingly to accent specific moments, like the beginning of a phrase or a transition between sections.\n"
            "3. **Create Surprise:** Use unpredictable timing. These sounds should catch the listener's ear without disrupting the main groove."
        )
    else: # Fallback for "complementary" or unknown roles
        role_instructions = (
            "**Your Role: A Complementary Part**\n"
            "Listen to the other instruments. Find a sonic niche and a rhythmic pattern that is currently unoccupied. Your goal is to add a new layer that enhances the overall composition without making it sound cluttered."
        )

    # --- The final prompt that puts it all together ---
    # Define which roles should be allowed to play overlapping notes (polyphony)
    POLYPHONIC_ROLES = {"harmony", "chords", "pads", "atmosphere", "texture", "guitar"}
    
    if role in POLYPHONIC_ROLES:
        polyphony_rule = "2.  **Polyphonic:** Notes for this track CAN overlap. This is crucial for playing chords and creating rich textures. The JSON array can contain note objects whose start and end times overlap."
    else:
        polyphony_rule = "2.  **Monophonic:** The notes in the JSON array must not overlap in time. A new note can only start after the previous one has finished."

    prompt = (
        f'You are an expert music producer creating a track inspired by: **{config["inspiration"]}**.\n'
        f"Your task is to compose a single instrument track for this production.\n\n"
        f"**--- MUSICAL CONTEXT ---**\n"
        f"{basic_instructions}\n"
        f"{context_prompt_part}"
        f"**--- YOUR TASK ---**\n"
        f"{generation_stage_instructions}\n\n"
        f"{role_instructions}\n\n"
        f"**--- UNIVERSAL PRINCIPLES OF GOOD MUSIC ---**\n"
        f"1. **Structure & Evolution:** A good musical part tells a story. Structure your composition over the full {length} bars. Avoid mindless, robotic repetition. For example, you could use an A/B structure where the first half establishes a theme (A) and the second half provides a variation or answer (B).\n"
        f"2. **Clarity through Space:** Do not create a constant wall of sound. The musical role of a part determines how it should use space and silence. Your role-specific instructions provide guidance on this.\n"
        f"3. **Dynamic Phrasing:** Use a wide range of velocity to create accents and shape the energy of the phrase. A static volume is boring.\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON array of objects. Each object represents a note and MUST have these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127, from the provided scale notes).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n\n'
        f"**IMPORTANT RULES:**\n"
        f'1.  **JSON ONLY:** Your entire response MUST be only the raw JSON array. DO NOT include ```json ... ```, explanations, or any other text.\n'
        f"{polyphony_rule}\n"
        f"3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n"
        f"4.  **Timing is Absolute:** 'start_beat' is the absolute position from the beginning of the {length}-bar clip.\n"
        f"5.  **Be Creative:** Compose a musically interesting and high-quality part that fits the description.\n"
        f'6.  **Valid JSON Syntax:** The output must be a perfectly valid JSON array. Use double quotes for all keys and string values. Ensure no trailing commas.\n\n'
        f"Now, generate the JSON array for the **{instrument_name}** track.\n"
    )
    return prompt

def generate_instrument_track_data(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int) -> Dict:
    """
    Generates musical data for a single instrument track using the generative AI model.
    It includes a retry mechanism for robustness and expects a JSON response.
    """
    prompt = create_single_track_prompt(config, length, instrument_name, program_num, context_tracks, role, current_track_index, total_tracks)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(Fore.BLUE + f"Attempt {attempt + 1}/{max_retries}: Generating part for {instrument_name} ({role})..." + Style.RESET_ALL)
            
            model = genai.GenerativeModel(
                model_name=config["model_name"],
                generation_config={
                    "temperature": 0.9, # A bit of creativity
                    "response_mime_type": "application/json"
                }
            )

            # Define safety settings to avoid blocking responses unnecessarily
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = model.generate_content(prompt, safety_settings=safety_settings)
            
            # --- Enhanced Response Validation ---
            # 1. Check if the response was stopped due to token limits
            if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
                print(Fore.RED + f"Error on attempt {attempt + 1}: Model stopped generating due to reaching the maximum token limit. The returned JSON is likely incomplete." + Style.RESET_ALL)
                print(Fore.YELLOW + "Consider asking for a shorter length (e.g., 8 or 16 bars) or simplifying the request for this instrument." + Style.RESET_ALL)
                continue # Skip to the next retry attempt

            # 2. Check for an empty response before parsing
            if not response.text.strip():
                print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Model returned an empty response for {instrument_name}." + Style.RESET_ALL)
                continue # Skip to the next retry attempt

            # 3. Parse the JSON response
            notes_list = json.loads(response.text)

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

        except (ValueError, json.JSONDecodeError, TypeError) as e:
            print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Data validation failed for {instrument_name}. Reason: {str(e)}" + Style.RESET_ALL)
            # Add more context to the error message
            if "response" in locals():
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    print(Fore.RED + f"Generation may have been blocked. Reason: {response.prompt_feedback.block_reason.name}")
                if hasattr(response, 'text'):
                    print(Fore.YELLOW + "Model response was:\n" + response.text + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error on attempt {attempt + 1} for {instrument_name}: {str(e)}" + Style.RESET_ALL)
            if "response" in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                print(Fore.RED + f"Generation blocked. Reason: {response.prompt_feedback.block_reason.name}")

        # If we are not on the last attempt, wait before retrying
        if attempt < max_retries - 1:
            print(Fore.YELLOW + "Waiting for 3 seconds before retrying..." + Style.RESET_ALL)
            time.sleep(3)

    print(Fore.RED + f"Failed to generate a valid part for {instrument_name} after {max_retries} attempts." + Style.RESET_ALL)
    return None

def generate_complete_track(config, length: int) -> Tuple[bool, Dict]:
    """
    Generates a complete multi-track song by iteratively creating parts for each instrument.
    Returns a tuple of (success_status, song_data).
    """
    print(Fore.CYAN + "\n--- Starting New Song Generation ---" + Style.RESET_ALL)
    
    song_data = {
        "bpm": config["bpm"],
        "time_signature": config["time_signature"],
        "key_scale": config["key_scale"],
        "tracks": []
    }
    
    context_tracks = []
    total_tracks = len(config["instruments"])

    for i, instrument in enumerate(config["instruments"]):
        instrument_name = instrument["name"]
        program_num = instrument["program_num"]
        role = instrument.get("role", "complementary") # Default role

        print(Fore.MAGENTA + f"\n--- Generating Track {i + 1}/{total_tracks}: {instrument_name} ---" + Style.RESET_ALL)
        
        # Determine which tracks to use for context
        # For "call and response", only use other melodic tracks for context
        relevant_context = []
        if config.get("use_call_and_response") == 1:
            MELODIC_ROLES = {'pads', 'texture', 'atmosphere', 'chords', 'arp', 'guitar', 'lead', 'melody', 'vocal', 'fx'}
            if role in MELODIC_ROLES:
                relevant_context = [t for t in context_tracks if t.get("role") in MELODIC_ROLES]
            else: # Non-melodic tracks get full context
                relevant_context = context_tracks
        else: # If not using C&R, all tracks get full context
            relevant_context = context_tracks

        track_data = generate_instrument_track_data(config, length, instrument_name, program_num, relevant_context, role, i, total_tracks)

        if track_data:
            song_data["tracks"].append(track_data)
            # Add only essential info to context for next prompt
            context_tracks.append({
                "instrument_name": instrument_name,
                "role": role,
                "program_num": program_num,
                "notes": track_data["notes"] # Pass the actual notes for context
            })
            # Add a cool-down period to avoid hitting API rate limits too quickly
            time.sleep(2)
        else:
            print(Fore.RED + f"Failed to generate track for {instrument_name}. Stopping generation." + Style.RESET_ALL)
            return False, None

    if not song_data["tracks"]:
        print(Fore.RED + "No tracks were generated successfully. Aborting." + Style.RESET_ALL)
        return False, None

    print(Fore.GREEN + "\n--- All tracks generated successfully! ---" + Style.RESET_ALL)
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

        for i, track_data in enumerate(song_data["tracks"]):
            track_name = track_data["instrument_name"]
            program_num = track_data["program_num"]
            channel = i % 16  # MIDI channels 0-15
            if channel == 9: # MIDI channel 10 (index 9) is for percussion
                channel += 1
            
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

def generate_filename(config, base_filename: str) -> str:
    """
    Generates a descriptive and valid filename based on the configuration,
    ensuring it does not exceed OS path length limits.
    """
    try:
        # Use a shortened and sanitized version of the inspiration prompt
        inspiration_text = config.get("inspiration", "project")
        # Replace spaces with underscores
        sanitized_inspiration = inspiration_text.replace(" ", "_")
        # Remove all characters that are not alphanumeric or underscores
        sanitized_inspiration = re.sub(r'[^a-zA-Z0-9_]', '', sanitized_inspiration)
        # Truncate to a reasonable length to avoid OS limits
        truncated_inspiration = sanitized_inspiration[:50]

        genre = config.get("genre", "audio").replace(" ", "_")
        key = config.get("key_scale", "").replace(" ", "").replace("/", "")
        bpm = config.get("bpm", "120")

        # Sanitize other parts for filename just in case
        genre = re.sub(r'[\\/*?:"<>|]', "", genre)
        key = re.sub(r'[\\/*?:"<>|]', "", key)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        base_dir = os.path.dirname(base_filename)
        
        # Construct the new, shorter filename
        new_name = f"{genre}_{truncated_inspiration}_{key}_{bpm}bpm_{timestamp}.mid"
        
        return os.path.join(base_dir, new_name)
    except Exception as e:
        print(Fore.YELLOW + f"Could not generate dynamic filename. Using default. Reason: {e}" + Style.RESET_ALL)
        return base_filename

def main():
    """
    Main function to run the music generation process.
    """
    try:
        config = load_config(CONFIG_FILE)
        
        # Configure the generative AI model
        genai.configure(api_key=config["api_key"])
        
        # --- Length Selection ---
        length_input = input(f"Enter the desired length in bars ({'/'.join(map(str, AVAILABLE_LENGTHS))}) or press Enter for default ({DEFAULT_LENGTH}): ").strip()
        if not length_input:
            length = DEFAULT_LENGTH
        else:
            try:
                length = int(length_input)
                if length not in AVAILABLE_LENGTHS:
                    print(Fore.YELLOW + f"Invalid length. Must be one of {AVAILABLE_LENGTHS}. Using default {DEFAULT_LENGTH}." + Style.RESET_ALL)
                    length = DEFAULT_LENGTH
            except ValueError:
                print(Fore.YELLOW + f"Invalid input. Using default length {DEFAULT_LENGTH}." + Style.RESET_ALL)
                length = DEFAULT_LENGTH

        # --- Generation Loop ---
        num_iterations = config.get("number_of_iterations", 1)
        for i in range(num_iterations):
            print(Fore.CYAN + f"\n--- Starting Generation Cycle {i + 1}/{num_iterations} ---" + Style.RESET_ALL)
            
            success, song_data = generate_complete_track(config, length)
            
            if success:
                # Generate a unique, descriptive filename for the MIDI output
                output_filename_base = os.path.join(script_dir, "generated_midi.mid")
                output_filename = generate_filename(config, output_filename_base)
                
                # Create the MIDI file from the song data
                create_midi_from_json(song_data, config, output_filename)

    except (ValueError, FileNotFoundError) as e:
        print(Fore.RED + f"A critical error occurred: {str(e)}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {str(e)}" + Style.RESET_ALL)

if __name__ == "__main__":
    main() 