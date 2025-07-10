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
import sys
import glob
if sys.platform == "win32":
    import msvcrt

# --- CONFIGURATION HELPERS (NEW) ---

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*50)
    print(f"--- {title.upper()} ---")
    print("="*50 + "\n")

# --- ROBUST CONFIG FILE PATH ---
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join the script's directory with the config file name to create an absolute path
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
# --- END ROBUST PATH ---

MAX_CONTEXT_CHARS = 100000  # A safe buffer below the 1M token limit for Gemini
BEATS_PER_BAR = 4
TICKS_PER_BEAT = 480

# Add new constants at the beginning of the file
AVAILABLE_LENGTHS = [4, 8, 16, 32, 64, 128]
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

        # Add context_window_size validation
        if "context_window_size" not in config:
            config["context_window_size"] = -1 # Default to dynamic
        else:
            try:
                size = int(config["context_window_size"])
                if size < -1:
                    raise ValueError("context_window_size must be -1 or greater.")
                config["context_window_size"] = size
            except (ValueError, TypeError):
                raise ValueError("context_window_size must be a valid integer.")

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

def get_dynamic_context(all_past_themes: List[Dict], character_budget: int = MAX_CONTEXT_CHARS) -> Tuple[List[Dict], int]:
    """
    Selects the most recent themes that fit within a character limit for the context.
    Returns the selected themes and the starting index from the original list.
    """
    if not all_past_themes:
        return [], 0

    context_themes = []
    current_chars = 0
    
    # Iterate backwards from the most recent theme
    for i in range(len(all_past_themes) - 1, -1, -1):
        theme = all_past_themes[i]
        try:
            theme_str = json.dumps(theme, separators=(',', ':'))
            if current_chars + len(theme_str) > character_budget:
                # The window starts at the next index
                return list(reversed(context_themes)), i + 1
            
            context_themes.append(theme)
            current_chars += len(theme_str)
        except TypeError:
            continue
            
    # If we get here, all themes fit
    return list(reversed(context_themes)), 0

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

def create_theme_prompt(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int, dialogue_role: str, theme_label: str, theme_description: str, previous_themes_full_history: List[Dict], current_theme_index: int):
    """
    Creates a universal, music-intelligent prompt that is tailored to generating a new theme based on previous ones.
    This version dynamically manages the context size to stay within limits.
    """
    total_beats_per_theme = length * config["time_signature"]["beats_per_bar"]
    scale_notes = get_scale_notes(config["root_note"], config["scale_type"])
    
    # --- Part 1: Assemble the NON-NEGOTIABLE parts of the prompt first ---
    
    basic_instructions = (
        f"**Genre:** {config['genre']}\n"
        f"**Tempo:** {config['bpm']} BPM\n"
        f"**Time Signature:** {config['time_signature']['beats_per_bar']}/{config['time_signature']['beat_value']}\n"
        f"**Key/Scale:** {config['key_scale'].title()} (Available notes: {scale_notes})\n"
        f"**Track Length:** {length} bars ({total_beats_per_theme} beats total)\n"
        f"**Instrument:** {instrument_name} (MIDI Program: {program_num})\n"
    )

    # Context for other tracks *within the current theme*
    context_prompt_part = ""
    if context_tracks:
        context_prompt_part = "**Inside the current theme, you have already written these parts. Compose a new part that fits with them:**\n"
        for track in context_tracks:
            notes_as_str = json.dumps(track['notes'], separators=(',', ':'))
            context_prompt_part += f"- **{track['instrument_name']}** (Role: {track['role']}):\n```json\n{notes_as_str}\n```\n"
        context_prompt_part += "\n"

    # Main Task description based on theme
    theme_task_instruction = ""
    timing_rule = f"4.  **Timing is Absolute:** 'start_beat' is the absolute position from the beginning of the {length}-bar clip.\n"

    if current_theme_index == 0:
        theme_task_instruction = (
            f"**Your Task: Compose the First Musical Theme**\n"
            f"This is the very first section of the song. Your goal is to establish the main musical ideas.\n"
            f"**Theme Name/Label:** {theme_label}\n"
            f"**Creative Direction for this Theme:** {theme_description}\n"
        )
    else:
        total_previous_beats = current_theme_index * total_beats_per_theme
        theme_task_instruction = (
            f"**Your Task: Compose a New, Contrasting Theme starting from beat {total_previous_beats}**\n"
            f"You must create a new musical section that logically follows the previous themes, but has a distinct character. It should feel like a new part of the song (e.g., a chorus following a verse, or a bridge).\n"
            f"**Theme Name/Label for this NEW Theme:** {theme_label}\n"
            f"**Creative Direction for this NEW Theme:** {theme_description}\n"
            "Analyze the previous themes and create something that complements them while bringing a fresh energy or emotion. It must work with the established key, tempo, and overall instrumentation."
        )
        timing_rule = f"4.  **Timing is Absolute:** 'start_beat' is the absolute position from the beginning of the *entire song composition so far*.\n"

    # Dialogue instructions
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

    # Role-specific instructions and rules
    role_instructions = get_role_instructions_for_generation(role, config)
    drum_map_instructions = ""
    if role in ["drums", "percussion", "kick_and_snare"]:
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
    
    # Polyphony and Key rules
    POLYPHONIC_ROLES = {"harmony", "chords", "pads", "atmosphere", "texture", "guitar"}
    EXPRESSIVE_MONOPHONIC_ROLES = {"lead", "melody", "vocal"}
    if role in POLYPHONIC_ROLES:
        polyphony_rule = "2.  **Polyphonic:** Notes for this track CAN overlap."
    elif role in EXPRESSIVE_MONOPHONIC_ROLES:
        polyphony_rule = "2.  **Expressive Monophonic:** Notes should primarily be played one at a time, but short overlaps are permitted."
    else: 
        polyphony_rule = "2.  **Strictly Monophonic:** The notes in the JSON array must NOT overlap in time."

    stay_in_key_rule = f"3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n"
    if role in ["drums", "percussion", "kick_and_snare"]:
        stay_in_key_rule = "3.  **Use Drum Map:** You must adhere to the provided Drum Map for all note pitches.\n"

    # General boilerplate and formatting instructions
    boilerplate_instructions = (
        f'You are an expert music producer composing different sections of a song inspired by: **{config["inspiration"]}**.\n\n'
        f"**--- OVERALL MUSICAL CONTEXT ---**\n"
        f"{basic_instructions}\n"
        f"{context_prompt_part}"
        f"**--- YOUR CURRENT TASK ---**\n"
        f"{theme_task_instruction}\n"
        f"You are composing the part for the **{instrument_name}**.\n"
        f"{call_and_response_instructions}"
        f"{drum_map_instructions}"
        f"{role_instructions}\n\n"
        f"**--- UNIVERSAL PRINCIPLES OF GOOD MUSIC ---**\n"
        f"1. **Structure & Evolution:** Your composition should have a clear structure. A good musical part tells a story over the full {length} bars by introducing a core idea ('motif') and then developing it through variation, repetition, and contrast. Avoid mindless, robotic repetition.\n"
        f"2. **Clarity through Space:** Do not create a constant wall of sound. Use rests effectively. The musical role of a part determines how it should use space. Your role-specific instructions provide guidance on this.\n"
        f"3. **Dynamic Phrasing:** Use a wide range of velocity to create accents and shape the energy of the phrase. A static volume is boring and unnatural.\n"
        f"4. **Tension & Release:** Build musical tension through dynamics, rhythmic complexity, or harmony, and resolve it at key moments (e.g., at the end of 4, 8, or 16 bar phrases) to create a satisfying arc.\n"
        f"5. **Ensemble Playing:** Think like a member of a band. Your performance must complement the other parts. Pay attention to the phrasing of other instruments and find pockets of space to add your musical statement without cluttering the arrangement.\n"
        f"6. **Micro-timing for Groove:** To add a human feel, you can subtly shift notes off the strict grid. Slightly anticipating a beat (pushing) can add urgency, while slightly delaying it (pulling) can create a more relaxed feel. This is especially effective for non-kick/snare elements.\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON array of objects. Each object represents a note and MUST have these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n\n'
        f"**IMPORTANT RULES:**\n"
        f'1.  **JSON ONLY:** Your entire response MUST be only the raw JSON array.\n'
        f"{polyphony_rule}\n"
        f"{stay_in_key_rule}"
        f"{timing_rule}"
        f'5.  **Valid JSON Syntax:** The output must be a perfectly valid JSON array.\n'
        f'6.  **Handling Silence:** If the creative direction explicitly requires this instrument to be silent for the entire section, output this specific JSON array to signify intentional silence: `[{{"pitch": 0, "start_beat": 0, "duration_beats": 0, "velocity": 0}}]`. Do not output an empty array for silence.\n\n'
        f"Now, generate the JSON array for the **{instrument_name}** track for the theme described as '{theme_description}'.\n"
    )

    # --- Part 2: Smart Context Calculation ---
    # Calculate the size of the prompt without the historical context
    base_prompt_size = len(
        boilerplate_instructions + basic_instructions + theme_task_instruction +
        context_prompt_part + call_and_response_instructions + drum_map_instructions +
        role_instructions + polyphony_rule + stay_in_key_rule + timing_rule
    )

    # Determine the remaining character budget for previous themes
    history_budget = MAX_CONTEXT_CHARS - base_prompt_size
    
    # Get only the themes that fit in the remaining budget
    safe_previous_themes, _ = get_dynamic_context(previous_themes_full_history, character_budget=history_budget)

    # --- Part 3: Assemble the FINAL prompt ---
    previous_themes_prompt_part = ""
    if safe_previous_themes:
        previous_themes_prompt_part = "**You have already composed the following themes. Use them as the primary context for what comes next:**\n"
        for i, theme in enumerate(safe_previous_themes):
            theme_name = theme.get("description", f"Theme {chr(65 + i)}")
            previous_themes_prompt_part += f"- **{theme_name}**:\n"
            # Include the full note data for each track in the theme.
            for track in theme['tracks']:
                notes_as_str = json.dumps(track['notes'], separators=(',', ':'))
                previous_themes_prompt_part += f"  - **{track['instrument_name']}** (Role: {track['role']}):\n  ```json\n  {notes_as_str}\n  ```\n"
        previous_themes_prompt_part += "\n"

    # The final prompt putting it all together
    prompt = (
        f'You are an expert music producer composing different sections of a song inspired by: **{config["inspiration"]}**.\n\n'
        f"**--- OVERALL MUSICAL CONTEXT ---**\n"
        f"{basic_instructions}\n"
        f"{previous_themes_prompt_part}"
        f"**--- YOUR CURRENT TASK ---**\n"
        f"{theme_task_instruction}\n"
        f"You are composing the part for the **{instrument_name}**.\n"
        f"{context_prompt_part}"
        f"{call_and_response_instructions}"
        f"{drum_map_instructions}"
        f"{role_instructions}\n\n"
        f"**--- UNIVERSAL PRINCIPLES OF GOOD MUSIC ---**\n"
        f"1. **Structure & Evolution:** Your composition should have a clear structure. A good musical part tells a story over the full {length} bars by introducing a core idea ('motif') and then developing it through variation, repetition, and contrast. Avoid mindless, robotic repetition.\n"
        f"2. **Clarity through Space:** Do not create a constant wall of sound. Use rests effectively. The musical role of a part determines how it should use space. Your role-specific instructions provide guidance on this.\n"
        f"3. **Dynamic Phrasing:** Use a wide range of velocity to create accents and shape the energy of the phrase. A static volume is boring and unnatural.\n"
        f"4. **Tension & Release:** Build musical tension through dynamics, rhythmic complexity, or harmony, and resolve it at key moments (e.g., at the end of 4, 8, or 16 bar phrases) to create a satisfying arc.\n"
        f"5. **Ensemble Playing:** Think like a member of a band. Your performance must complement the other parts. Pay attention to the phrasing of other instruments and find pockets of space to add your musical statement without cluttering the arrangement.\n"
        f"6. **Micro-timing for Groove:** To add a human feel, you can subtly shift notes off the strict grid. Slightly anticipating a beat (pushing) can add urgency, while slightly delaying it (pulling) can create a more relaxed feel. This is especially effective for non-kick/snare elements.\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON array of objects. Each object represents a note and MUST have these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n\n'
        f"**IMPORTANT RULES:**\n"
        f'1.  **JSON ONLY:** Your entire response MUST be only the raw JSON array.\n'
        f"{polyphony_rule}\n"
        f"{stay_in_key_rule}"
        f"{timing_rule}"
        f'5.  **Valid JSON Syntax:** The output must be a perfectly valid JSON array.\n'
        f'6.  **Handling Silence:** If the creative direction explicitly requires this instrument to be silent for the entire section, output this specific JSON array to signify intentional silence: `[{{"pitch": 0, "start_beat": 0, "duration_beats": 0, "velocity": 0}}]`. Do not output an empty array for silence.\n\n'
        f"Now, generate the JSON array for the **{instrument_name}** track for the theme described as '{theme_description}'.\n"
    )
    # Debug print to check prompt length
    # print(f"DEBUG: Prompt length: {len(prompt)} characters.")
    return prompt

def generate_instrument_track_data(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int, dialogue_role: str, theme_label: str, theme_description: str, previous_themes_full_history: List[Dict], current_theme_index: int) -> Dict:
    """
    Generates musical data for a single instrument track using the generative AI model, adapted for themes.
    """
    prompt = create_theme_prompt(config, length, instrument_name, program_num, context_tracks, role, current_track_index, total_tracks, dialogue_role, theme_label, theme_description, previous_themes_full_history, current_theme_index)
    
    while True: # Loop to allow retrying after complete failure
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

                # --- NEW: Check for special silence signal ---
                if isinstance(notes_list, list) and len(notes_list) == 1:
                    note = notes_list[0]
                    if note.get("pitch") == 0 and note.get("start_beat") == 0 and note.get("duration_beats") == 0 and note.get("velocity") == 0:
                        print(Fore.GREEN + f"Recognized intentional silence for {instrument_name}. The track will be empty." + Style.RESET_ALL)
                        return {
                            "instrument_name": instrument_name,
                            "program_num": program_num,
                            "role": role,
                            "notes": [] # Return a valid track with no notes
                        }

                if not isinstance(notes_list, list):
                    raise TypeError("The generated data is not a valid list of notes.")

                # --- Data Validation ---
                validated_notes = []
                for note in notes_list:
                    if not all(k in note for k in ["pitch", "start_beat", "duration_beats", "velocity"]):
                         print(Fore.YELLOW + f"Warning: Skipping invalid note object: {note}" + Style.RESET_ALL)
                         continue
                    validated_notes.append(note)

                print(Fore.GREEN + f"Successfully generated part for {instrument_name}." + Style.RESET_ALL)
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
        
        # This part is reached after all retries fail.
        print(Fore.RED + f"Failed to generate a valid part for {instrument_name} after {max_retries} attempts." + Style.RESET_ALL)
        
        print(Fore.CYAN + "Automatic retry in 60 seconds..." + Style.RESET_ALL)
        print(Fore.YELLOW + "Press 'y' to retry now, 'n' to cancel, or 'w' to pause the timer." + Style.RESET_ALL)

        user_action = None
        # Countdown loop
        for i in range(60, 0, -1):
            if sys.platform == "win32" and msvcrt.kbhit():
                char = msvcrt.getch().decode().lower()
                if char in ['y', 'n', 'w']:
                    user_action = char
                    break # Interrupt countdown
            print(f"  Retrying in {i} seconds...  ", end="\r")
            time.sleep(1)
        
        print("                               ", end="\r") # Clear the countdown line

        # --- Decide what to do based on user_action ---
        
        # Case 1: Countdown finished or user pressed 'y'
        if user_action is None or user_action == 'y':
            print(Fore.CYAN + "Retrying now..." + Style.RESET_ALL)
            continue # Restart the whole `while True` loop

        # Case 2: User pressed 'n'
        elif user_action == 'n':
            print(Fore.RED + f"Aborting generation for '{instrument_name}'." + Style.RESET_ALL)
            return None # Exit function
            
        # Case 3: User pressed 'w' to wait
        elif user_action == 'w':
            print(Fore.YELLOW + "Timer paused. Waiting for manual input." + Style.RESET_ALL)
            while True:
                manual_choice = input(Fore.YELLOW + f"Retry for '{instrument_name}'? (y/n): " + Style.RESET_ALL).strip().lower()
                if manual_choice in ['y', 'yes']:
                    # This will break the inner 'w' loop, and the outer `while True` will continue, causing a retry.
                    break 
                elif manual_choice in ['n', 'no']:
                    print(Fore.RED + f"Aborting generation for '{instrument_name}'." + Style.RESET_ALL)
                    return None
                else:
                    print(Fore.YELLOW + "Invalid input. Please enter 'y' or 'n'." + Style.RESET_ALL)


def create_song_optimization(config: Dict, theme_length: int, themes_to_optimize: List[Dict], script_dir: str, opt_iteration_num: int, run_timestamp: str, user_optimization_prompt: str = "", resume_data=None) -> List[Dict]:
    """
    Creates a new version of the song by optimizing it theme-by-theme and track-by-track.
    It now saves each optimized part immediately after it's created.
    Can resume from previous progress if resume_data is provided.
    """
    print(Fore.CYAN + "\n--- Creating and Saving New Song Optimization by Part ---" + Style.RESET_ALL)
    
    final_optimized_themes = []
    
    # Determine the effective user prompt. For new runs, it's the argument. For resumed runs, it's from the saved data.
    prompt_for_this_run = user_optimization_prompt

    start_theme_index = 0
    start_track_index = 0
    if resume_data:
        final_optimized_themes = resume_data.get('final_optimized_themes', [])
        start_theme_index = resume_data.get('current_theme_index', 0)
        start_track_index = resume_data.get('current_track_index', 0)
        prompt_for_this_run = resume_data.get('user_optimization_prompt', "") # Use saved prompt
        print(Fore.CYAN + f"Resuming optimization from theme {start_theme_index + 1}, track {start_track_index + 1}" + Style.RESET_ALL)
        if prompt_for_this_run:
            print(Fore.CYAN + f"Using saved optimization prompt: '{prompt_for_this_run}'" + Style.RESET_ALL)
    
    try:
        for i in range(start_theme_index, len(themes_to_optimize)):
            theme_to_optimize = themes_to_optimize[i]
            theme_description = theme_to_optimize.get('description', f"Theme {chr(65+i)}")
            print(Fore.CYAN + f"\n--- Optimizing Theme {i+1}/{len(themes_to_optimize)}: '{theme_description}' ---" + Style.RESET_ALL)
            
            # Use a dynamic sliding window for context based on config
            if config.get("context_window_size", -1) == -1:
                context_themes, context_start_index = get_dynamic_context(final_optimized_themes[:i])
            else:
                window_size = config["context_window_size"]
                context_start_index = max(0, i - window_size)
                context_themes = final_optimized_themes[context_start_index:i]

            optimized_theme_tracks = []
            current_section_context_tracks = [dict(t) for t in theme_to_optimize['tracks']]
            
            track_start_index = start_track_index if i == start_theme_index else 0
            
            if i == start_theme_index and resume_data and 'partial_theme_tracks' in resume_data:
                optimized_theme_tracks = resume_data['partial_theme_tracks']
                for j, optimized_track in enumerate(optimized_theme_tracks):
                    if j < len(current_section_context_tracks):
                        current_section_context_tracks[j] = optimized_track
            
            for track_index in range(track_start_index, len(theme_to_optimize['tracks'])):
                track_to_optimize = theme_to_optimize['tracks'][track_index]
                role = track_to_optimize.get("role", "complementary")
                print(Fore.MAGENTA + f"\n--- Optimizing Track: {track_to_optimize['instrument_name']} (Role: {role}) ---" + Style.RESET_ALL)
                print(f"{Fore.CYAN}Goal:{Style.RESET_ALL} {get_optimization_goal_for_role(role)}")
                
                inner_context_for_ai = [t for j, t in enumerate(current_section_context_tracks) if j != track_index]

                new_track_data = generate_optimization_data(
                    config, theme_length, track_to_optimize, role,
                    theme_description, context_themes, inner_context_for_ai, context_start_index,
                    prompt_for_this_run
                )
                
                if new_track_data:
                    if track_index < len(optimized_theme_tracks):
                        optimized_theme_tracks[track_index] = new_track_data
                    else:
                        optimized_theme_tracks.append(new_track_data)
                    current_section_context_tracks[track_index] = new_track_data
                    
                    progress_data = {
                        'type': 'optimization', 'config': config, 'theme_length': theme_length,
                        'themes_to_optimize': themes_to_optimize, 'opt_iteration_num': opt_iteration_num,
                        'final_optimized_themes': final_optimized_themes, 'current_theme_index': i,
                        'current_track_index': track_index + 1, 'partial_theme_tracks': optimized_theme_tracks,
                        'completed_tracks': len(optimized_theme_tracks), 'total_tracks_in_theme': len(theme_to_optimize['tracks']),
                        'timestamp': run_timestamp,
                        'user_optimization_prompt': prompt_for_this_run
                    }
                    save_progress(progress_data, script_dir, run_timestamp)
                else:
                    print(Fore.RED + f"Failed to generate optimization for {track_to_optimize['instrument_name']}. Using original track." + Style.RESET_ALL)
                    if track_index < len(optimized_theme_tracks):
                        optimized_theme_tracks[track_index] = track_to_optimize
                    else:
                        optimized_theme_tracks.append(track_to_optimize)

            original_part_filename = theme_to_optimize.get('original_filename', f"Theme_{chr(65+i)}_{i}.mid")
            part_base_name = os.path.splitext(original_part_filename)[0]
            clean_part_base_name = re.sub(r'_opt_\d+$', '', part_base_name)
            opt_part_filename = os.path.join(script_dir, f"{clean_part_base_name}_opt_{opt_iteration_num}.mid")
            
            # Create a data structure for the single theme and use the correct MIDI saving function.
            # Notes from optimization are already relative, so no time offset is needed for the part file.
            theme_part_data = {"tracks": optimized_theme_tracks}
            create_part_midi_from_theme(theme_part_data, config, opt_part_filename, time_offset_beats=0)
            time.sleep(1)

            final_optimized_themes.append({
                "description": theme_description, "tracks": optimized_theme_tracks,
                "original_filename": original_part_filename
            })
            start_track_index = 0

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n--- Optimization interrupted by user ---" + Style.RESET_ALL)
        progress_data = {
            'type': 'optimization_interrupted', 'config': config, 'theme_length': theme_length,
            'themes_to_optimize': themes_to_optimize, 'opt_iteration_num': opt_iteration_num,
            'final_optimized_themes': final_optimized_themes, 'current_theme_index': len(final_optimized_themes),
            'current_track_index': 0, 'completed_themes': len(final_optimized_themes),
            'total_themes': len(themes_to_optimize), 'timestamp': run_timestamp,
            'user_optimization_prompt': prompt_for_this_run
        }
        save_progress(progress_data, script_dir, run_timestamp)
        return None
    except Exception as e:
        print(Fore.RED + f"Unexpected error during optimization: {e}" + Style.RESET_ALL)
        progress_data = {
            'type': 'optimization_error', 'config': config, 'theme_length': theme_length,
            'themes_to_optimize': themes_to_optimize, 'opt_iteration_num': opt_iteration_num,
            'final_optimized_themes': final_optimized_themes, 'current_theme_index': len(final_optimized_themes),
            'current_track_index': 0, 'completed_themes': len(final_optimized_themes),
            'total_themes': len(themes_to_optimize), 'error': str(e), 'timestamp': run_timestamp,
            'user_optimization_prompt': prompt_for_this_run
        }
        save_progress(progress_data, script_dir, run_timestamp)
        return None

    print(Fore.GREEN + "\n--- Successfully created and saved all optimized parts! ---" + Style.RESET_ALL)
    return final_optimized_themes

def create_optimization_prompt(config: Dict, length: int, track_to_optimize: Dict, role: str, theme_description: str, context_themes: List[Dict], inner_context_tracks: List[Dict], context_start_index: int, user_optimization_prompt: str) -> str:
    """
    Creates a prompt for optimizing a single track within a themed song structure.
    It now normalizes the context themes' timestamps to be relative.
    """
    scale_notes = get_scale_notes(config["root_note"], config["scale_type"])
    
    basic_instructions = (
        f"**Genre:** {config['genre']}\n"
        f"**Tempo:** {config['bpm']} BPM\n"
        f"**Time Signature:** {config['time_signature']['beats_per_bar']}/{config['time_signature']['beat_value']}\n"
        f"**Key/Scale:** {config['key_scale'].title()} (Available notes: {scale_notes})\n"
        f"**Instrument:** {track_to_optimize['instrument_name']} (MIDI Program: {track_to_optimize['program_num']})\n"
    )

    # --- Context from PREVIOUS themes (with normalized timestamps) ---
    previous_themes_prompt_part = ""
    theme_length_beats = length * config["time_signature"]["beats_per_bar"]
    if context_themes:
        previous_themes_prompt_part = "**Context from Previous Song Sections:**\nThe song begins with the following part(s). Their timings are relative to the start of their own section. Use them as a reference for your optimization.\n"
        for i, theme in enumerate(context_themes):
            theme_name = theme.get("description", f"Theme {chr(65 + i)}")
            # Calculate the absolute time offset for this specific context theme
            context_theme_index = context_start_index + i
            time_offset_beats = context_theme_index * theme_length_beats
            
            previous_themes_prompt_part += f"- **{theme_name}**:\n"
            for track in theme['tracks']:
                # Normalize notes to be relative to the start of their own theme
                normalized_notes = []
                for note in track['notes']:
                    new_note = note.copy()
                    new_note['start_beat'] = float(new_note['start_beat']) - time_offset_beats
                    # Clip at 0 to prevent negative start times from floating point errors
                    new_note['start_beat'] = max(0, round(new_note['start_beat'], 4))
                    normalized_notes.append(new_note)

                notes_as_str = json.dumps(normalized_notes, separators=(',', ':'))
                previous_themes_prompt_part += f"  - **{track['instrument_name']}** (Role: {track['role']}):\n  ```json\n  {notes_as_str}\n  ```\n"
        previous_themes_prompt_part += "\n"

    # --- Context from tracks WITHIN the CURRENT theme ---
    inner_context_prompt_part = ""
    if inner_context_tracks:
        inner_context_prompt_part = "**Context from the Current Song Section:**\nWithin this section, you have already optimized these parts. Make your new part fit perfectly with them.\n"
        for track in inner_context_tracks:
            notes_as_str = json.dumps(track['notes'], separators=(',', ':'))
            inner_context_prompt_part += f"- **{track['instrument_name']}** (Role: {track['role']}):\n```json\n{notes_as_str}\n```\n"
        inner_context_prompt_part += "\n"

    original_part_prompt = (
        "**This is the original part you need to optimize:**\n"
        f"```json\n{json.dumps(track_to_optimize['notes'], separators=(',', ':'))}\n```\n"
    )

    optimization_instruction = (
        "**Your Task: Optimize The Provided Part**\n"
        f"You are currently working on a section described as: **'{theme_description}'**. Your main goal is to **improve and refine** the original part provided below. Keep its core identity and instrumentation, but enhance it to better fit the creative direction of this specific section and the overall song.\n"
        f"**IMPORTANT DURATION REQUIREMENT:** You MUST generate a complete musical phrase that spans the entire **{length} bars**. Do not just create a short 4-bar loop. The musical ideas must evolve and be present throughout the full {length}-bar duration. The composition must actively fill the entire **{length} bars**. Intentional silence for musical effect (like rests between phrases or a build-up) is encouraged, but avoid leaving the end of the track empty simply because the generation was incomplete.\n"
        f"**Creative Direction:** {config['inspiration']}\n"
    )
    
    # --- NEW: Add user's specific prompt if provided ---
    user_prompt_section = ""
    if user_optimization_prompt:
        user_prompt_section = (
            f"**--- User's Specific Optimization Instructions ---**\n"
            f"Apply the following user-provided creative direction to your optimization:\n"
            f"'{user_optimization_prompt}'\n\n"
        )

    # Use detailed role instructions for optimization
    role_instructions = get_role_instructions_for_optimization(role, config)

    drum_map_instructions = ""
    if role in ["drums", "percussion", "kick_and_snare"]:
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
    
    POLYPHONIC_ROLES = {"harmony", "chords", "pads", "atmosphere", "texture", "guitar"}
    EXPRESSIVE_MONOPHONIC_ROLES = {"lead", "melody", "vocal"}
    if role in POLYPHONIC_ROLES:
        polyphony_rule = "2.  **Polyphonic:** Notes for this track CAN overlap."
    elif role in EXPRESSIVE_MONOPHONIC_ROLES:
        polyphony_rule = "2.  **Expressive Monophonic:** Notes should primarily be played one at a time, but short overlaps are permitted."
    else: 
        polyphony_rule = "2.  **Strictly Monophonic:** The notes in the JSON array must NOT overlap in time."

    stay_in_key_rule = f"3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n"
    if role in ["drums", "percussion", "kick_and_snare"]:
        stay_in_key_rule = "3.  **Use Drum Map:** You must adhere to the provided Drum Map for all note pitches.\n"

    # The final prompt putting it all together
    prompt = (
        f"You are an expert music producer optimizing a single track within a larger song.\n"
        f"**--- MUSICAL CONTEXT ---**\n"
        f"{basic_instructions}\n"
        f"{previous_themes_prompt_part}"
        f"**--- YOUR TASK ---**\n"
        f"{inner_context_prompt_part}"
        f"{original_part_prompt}"
        f"{optimization_instruction}\n"
        f"{user_prompt_section}"
        f"{role_instructions}\n"
        f"{drum_map_instructions}"
        f"--- Your Producer's Checklist for Optimization ---\n\n"
        f"1.  **Analyze Musical Density:** First, assess the original part. \n"
        f"    - If it's too crowded and sounds cluttered, your main goal is to **simplify**. Create space by adding more rests or removing less important notes.\n"
        f"    - Conversely, if the part is too sparse or overly repetitive, your goal is to **add interest**. Introduce subtle 'ear candy' like short melodic fills, quick arpeggios, or an occasional, unexpected note to keep the listener engaged.\n\n"
        f"2.  **Exaggerate Dynamics:** Don't just copy the velocity of the original. Give the part life by exaggerating its dynamics. Make quiet notes quieter and loud notes louder. Use velocity to create clear accents and a sense of human performance.\n\n"
        f"3.  **Enhance the Groove:** If the original rhythm feels stiff, improve its groove. You can achieve this by adding subtle syncopation or by slightly shifting notes off the strict grid (micro-timing) to make them 'push' or 'pull' against the beat.\n\n"
        f"4.  **Tell a Story Over Time:** A great part evolves. Avoid simply looping a 4-bar phrase for 16 bars. Introduce subtle variations as the part progresses. For example, you could add or alter a note in the second half, or transpose a key phrase up or down an octave to build intensity.\n\n"
        f"5.  **Play with the Ensemble:** Always remember the context of the other instruments. Listen to their phrasing and find pockets of space for your musical statements without cluttering the overall arrangement.\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON array of objects. Each object represents a note and MUST have these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n\n'
        f"**IMPORTANT RULES:**\n"
        f'1. **JSON ONLY:** Your entire response MUST be only the raw JSON array.\n'
        f"{polyphony_rule}\n"
        f"{stay_in_key_rule}"
        f"4.  **Timing is Relative:** All 'start_beat' values must be relative to the beginning of this {length}-bar section, NOT the whole song.\n"
        f"5.  **Be Creative:** Compose a high-quality, optimized part that is musically interesting and follows the creative direction.\n"
        f'6.  **Valid JSON Syntax:** The output must be a perfectly valid JSON array.\n'
        f'7.  **Handling Silence:** If your optimization goal is to make the instrument completely silent, output this specific JSON array: `[{{"pitch": 0, "start_beat": 0, "duration_beats": 0, "velocity": 0}}]`.\n\n'
        f"Now, generate the JSON array for the new, optimized version of the **{track_to_optimize['instrument_name']}** track for the section '{theme_description}'. Remember, the generated part MUST cover the full {length} bars.\n"
    )
    return prompt

def generate_optimization_data(config: Dict, length: int, track_to_optimize: Dict, role: str, theme_description: str, context_themes: List[Dict], inner_context_tracks: List[Dict], context_start_index: int, user_optimization_prompt: str) -> Dict:
    """
    Generates an optimization for a single instrument track.
    """
    prompt = create_optimization_prompt(config, length, track_to_optimize, role, theme_description, context_themes, inner_context_tracks, context_start_index, user_optimization_prompt)
    
    while True: # Loop to allow retrying after complete failure
        max_retries = 3
        for attempt in range(max_retries):
            try:
                instrument_name = track_to_optimize['instrument_name']
                print(Fore.BLUE + f"Attempt {attempt + 1}/{max_retries}: Generating optimization for {instrument_name}..." + Style.RESET_ALL)
                
                generation_config = {
                    "temperature": config["temperature"],
                    "response_mime_type": "application/json",
                    "max_output_tokens": 65536,
                }
                model = genai.GenerativeModel(model_name=config["model_name"], generation_config=generation_config)
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]

                response = model.generate_content(prompt, safety_settings=safety_settings)
                
                if not response.candidates or response.candidates[0].finish_reason not in [1, "STOP"]:
                    print(Fore.RED + f"Error on attempt {attempt + 1}: Generation failed." + Style.RESET_ALL)
                    continue

                notes_list = json.loads(response.text)

                # --- NEW: Check for special silence signal ---
                if isinstance(notes_list, list) and len(notes_list) == 1:
                    note = notes_list[0]
                    if note.get("pitch") == 0 and note.get("start_beat") == 0 and note.get("duration_beats") == 0 and note.get("velocity") == 0:
                        print(Fore.GREEN + f"Recognized intentional silence for {track_to_optimize['instrument_name']}. The optimized track will be empty." + Style.RESET_ALL)
                        return {
                            "instrument_name": track_to_optimize['instrument_name'],
                            "program_num": track_to_optimize.get("program_num", 0),
                            "role": role,
                            "notes": []
                        }

                if not isinstance(notes_list, list):
                    raise TypeError("The generated data is not a valid list of notes.")

                validated_notes = []
                for note in notes_list:
                    if all(k in note for k in ["pitch", "start_beat", "duration_beats", "velocity"]):
                        validated_notes.append(note)
                
                print(Fore.GREEN + f"Successfully generated optimization for {instrument_name}." + Style.RESET_ALL)
                return {
                    "instrument_name": instrument_name,
                    "program_num": track_to_optimize.get("program_num", 0),
                    "role": role,
                    "notes": validated_notes
                }

            except (json.JSONDecodeError, TypeError) as e:
                print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Data validation failed for {track_to_optimize['instrument_name']}. Reason: {str(e)}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"An unexpected error occurred on attempt {attempt + 1}: {str(e)}" + Style.RESET_ALL)

            if attempt < max_retries - 1:
                time.sleep(3)
        
        # This part is reached after all retries fail.
        instrument_name = track_to_optimize['instrument_name']
        print(Fore.RED + f"Failed to generate a valid optimization for {instrument_name} after {max_retries} attempts." + Style.RESET_ALL)
        
        print(Fore.CYAN + "Automatic retry in 60 seconds..." + Style.RESET_ALL)
        print(Fore.YELLOW + "Press 'y' to retry now, 'n' to cancel, or 'w' to pause the timer." + Style.RESET_ALL)

        user_action = None
        # Countdown loop
        for i in range(60, 0, -1):
            if sys.platform == "win32" and msvcrt.kbhit():
                char = msvcrt.getch().decode().lower()
                if char in ['y', 'n', 'w']:
                    user_action = char
                    break # Interrupt countdown
            print(f"  Retrying in {i} seconds...  ", end="\r")
            time.sleep(1)
        
        print("                               ", end="\r") # Clear the countdown line

        # --- Decide what to do based on user_action ---
        
        # Case 1: Countdown finished or user pressed 'y'
        if user_action is None or user_action == 'y':
            print(Fore.CYAN + "Retrying now..." + Style.RESET_ALL)
            continue # Restart the whole `while True` loop

        # Case 2: User pressed 'n'
        elif user_action == 'n':
            print(Fore.RED + f"Aborting optimization for '{instrument_name}'." + Style.RESET_ALL)
            return None # Exit function
            
        # Case 3: User pressed 'w' to wait
        elif user_action == 'w':
            print(Fore.YELLOW + "Timer paused. Waiting for manual input." + Style.RESET_ALL)
            while True:
                manual_choice = input(Fore.YELLOW + f"Retry for '{instrument_name}'? (y/n): " + Style.RESET_ALL).strip().lower()
                if manual_choice in ['y', 'yes']:
                    # This will break the inner 'w' loop, and the outer `while True` will continue, causing a retry.
                    break 
                elif manual_choice in ['n', 'no']:
                    print(Fore.RED + f"Aborting optimization for '{instrument_name}'." + Style.RESET_ALL)
                    return None
                else:
                    print(Fore.YELLOW + "Invalid input. Please enter 'y' or 'n'." + Style.RESET_ALL)

def generate_one_theme(config, length: int, theme_def: dict, previous_themes: List[Dict]) -> Tuple[bool, Dict]:
    """
    Generates one complete musical theme (e.g., a verse or chorus).
    """
    theme_description = theme_def.get('description', 'No description')
    theme_label = theme_def.get('label', 'Untitled Theme')
    
    # The main header for the theme is now printed in generate_all_themes_and_save_parts.
    # We can print the detailed blueprint here.
    print(f"{Style.DIM}{Fore.WHITE}Blueprint: {theme_description}{Style.RESET_ALL}")
    
    song_data = {
        "bpm": config["bpm"],
        "time_signature": config["time_signature"],
        "key_scale": config["key_scale"],
        "tracks": [],
        "description": theme_description # Store the description
    }
    
    context_tracks = []
    total_tracks = len(config["instruments"])
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

        print(
            f"\n{Fore.MAGENTA}--- Generating Track {Style.BRIGHT}{Fore.YELLOW}{i + 1}/{total_tracks}{Style.RESET_ALL}{Fore.MAGENTA}"
            f": {Style.BRIGHT}{Fore.GREEN}{instrument_name}{Style.NORMAL}"
            f" (Role: {role})"
            f"{Style.RESET_ALL}"
        )
        
        # Pass the growing list of *all previously generated themes* to the track generator
        track_data = generate_instrument_track_data(
            config, length, instrument_name, program_num, 
            context_tracks, role, i, total_tracks, dialogue_role, 
            theme_def.get('label', ''), theme_def.get('description', ''), previous_themes,
            current_theme_index=i
        )

        if track_data:
            song_data["tracks"].append(track_data)
            # Add the new track to the context *for the current theme*
            context_tracks.append({
                "instrument_name": instrument_name,
                "role": role,
                "program_num": program_num,
                "notes": track_data["notes"]
            })
            time.sleep(2)
        else:
            print(Fore.RED + f"Failed to generate track for {instrument_name}. Stopping generation for this theme." + Style.RESET_ALL)
            return False, None

    if not song_data["tracks"]:
        print(Fore.RED + f"No tracks were generated for {theme_label}. Aborting." + Style.RESET_ALL)
        return False, None

    print(Fore.GREEN + f"\n--- Theme '{theme_label}' generated successfully! ---" + Style.RESET_ALL)
    return True, song_data

def create_midi_from_json(song_data: Dict, config: Dict, output_file: str, time_offset_beats: float = 0.0) -> bool:
    """
    Creates a MIDI file from the generated song data structure.
    An optional time_offset_beats can be provided to shift the entire MIDI content.
    """
    try:
        bpm = config["bpm"]
        time_signature_beats = config["time_signature"]["beats_per_bar"]
        
        # Initialize MIDIFile with one extra track for tempo/metadata
        num_instrument_tracks = len(song_data["tracks"])
        midi_file = MIDIFile(num_instrument_tracks + 1, removeDuplicates=True, deinterleave=False)
        
        # Add tempo and time signature to the dedicated track 0
        tempo_track = 0
        midi_file.addTempo(track=tempo_track, time=0, tempo=bpm)
        midi_file.addTimeSignature(track=tempo_track, time=0, numerator=time_signature_beats, denominator=4, clocks_per_tick=24)

        # Smart channel assignment
        # Channel 10 (index 9) is for percussion. Non-drum channels will be assigned from 0, skipping 9.
        next_melodic_channel = 0
        for i, track_data in enumerate(song_data["tracks"]):
            track_name = track_data["instrument_name"]
            program_num = track_data["program_num"]
            role = track_data.get("role", "complementary")
            
            # MIDI track number is now offset by 1
            midi_track_num = i + 1

            # Assign MIDI channel
            if role in ["drums", "percussion", "kick_and_snare"]:
                channel = 9 # MIDI Channel 10 for drums
            else:
                channel = next_melodic_channel
                if channel == 9: # Skip the drum channel
                    next_melodic_channel += 1
                    channel = next_melodic_channel
                next_melodic_channel += 1

            # Fallback if we somehow run out of channels
            if channel > 15: channel = 15
            
            midi_file.addTrackName(midi_track_num, 0, track_name)
            midi_file.addProgramChange(midi_track_num, channel, 0, program_num)
            
            for note in track_data["notes"]:
                try:
                    pitch = int(note["pitch"])
                    start_beat = float(note["start_beat"])
                    duration_beats = float(note["duration_beats"])
                    velocity = int(note["velocity"])
                    
                    # Ensure values are within MIDI specs
                    if 0 <= pitch <= 127 and 1 <= velocity <= 127 and duration_beats > 0:
                        midi_file.addNote(
                            track=midi_track_num,
                            channel=channel,
                            pitch=pitch,
                            time=start_beat - time_offset_beats,
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

def create_part_midi_from_theme(theme_data: Dict, config: Dict, output_file: str, time_offset_beats: float = 0.0) -> bool:
    """
    Creates a MIDI file for a single theme (part) from its track data.
    It subtracts the provided time_offset_beats from all notes to normalize them.
    """
    try:
        bpm = config["bpm"]
        time_signature_beats = config["time_signature"]["beats_per_bar"]
        
        num_instrument_tracks = len(theme_data["tracks"])
        midi_file = MIDIFile(num_instrument_tracks + 1, removeDuplicates=True, deinterleave=False)
        
        tempo_track = 0
        midi_file.addTempo(track=tempo_track, time=0, tempo=bpm)
        midi_file.addTimeSignature(track=tempo_track, time=0, numerator=time_signature_beats, denominator=4, clocks_per_tick=24)

        next_melodic_channel = 0
        for i, track_data in enumerate(theme_data["tracks"]):
            track_name = track_data["instrument_name"]
            program_num = track_data["program_num"]
            role = track_data.get("role", "complementary")
            
            midi_track_num = i + 1

            if role in ["drums", "percussion", "kick_and_snare"]:
                channel = 9
            else:
                channel = next_melodic_channel
                if channel == 9:
                    next_melodic_channel += 1
                    channel = next_melodic_channel
                next_melodic_channel += 1
            if channel > 15: channel = 15
            
            midi_file.addTrackName(midi_track_num, 0, track_name)
            midi_file.addProgramChange(midi_track_num, channel, 0, program_num)
            
            for note in track_data["notes"]:
                try:
                    pitch = int(note["pitch"])
                    start_beat = float(note["start_beat"])
                    duration_beats = float(note["duration_beats"])
                    velocity = int(note["velocity"])
                    
                    if 0 <= pitch <= 127 and 1 <= velocity <= 127 and duration_beats > 0:
                        midi_file.addNote(
                            track=midi_track_num, channel=channel, pitch=pitch,
                            time=start_beat - time_offset_beats, # Normalize time
                            duration=duration_beats, volume=velocity
                        )
                except (ValueError, TypeError) as e:
                    print(Fore.YELLOW + f"Warning: Skipping invalid note data in track '{track_name}': {note}. Reason: {e}" + Style.RESET_ALL)

        with open(output_file, "wb") as f:
            midi_file.writeFile(f)
            
        print(Fore.GREEN + f"\nSuccessfully created MIDI part file: {output_file}" + Style.RESET_ALL)
        return True

    except Exception as e:
        print(Fore.RED + f"Error creating MIDI part file: {str(e)}" + Style.RESET_ALL)
        return False

def generate_filename(config: Dict, base_dir: str, length_bars: int, theme_label: str, theme_index: int, timestamp: str) -> str:
    """
    Generates a descriptive and valid filename for a theme.
    """
    try:
        genre = config.get("genre", "audio").replace(" ", "_").replace("/", "-")
        key = config.get("key_scale", "").replace(" ", "").replace("#", "s")
        bpm = round(float(config.get("bpm", 120)))
        theme_char = chr(65 + theme_index) # A, B, C...

        # Sanitize parts for filename
        genre = re.sub(r'[\\*?:"<>|]', "", genre)
        key = re.sub(r'[\\*?:"<>|]', "", key)
        # Sanitize the user-provided label for the filename
        sanitized_label = re.sub(r'[\s/\\:*?"<>|]+', '_', theme_label)

        # Construct the new, descriptive filename
        # Format: ThemeChar_Label_Genre_Key_Length_BPM_Timestamp.mid
        new_name = f"{theme_char}_{sanitized_label}_{genre}_{key}_{length_bars}bars_{bpm}bpm_{timestamp}.mid"
        
        return os.path.join(base_dir, new_name)
    except Exception as e:
        theme_char = chr(65 + theme_index)
        print(Fore.YELLOW + f"Could not generate dynamic filename. Using default. Reason: {e}" + Style.RESET_ALL)
        return os.path.join(base_dir, f"theme_{theme_char}_{timestamp}.mid")

def get_next_available_file_number(base_midi_path: str) -> int:
    """
    Scans the directory for existing optimized files for a given base name
    and determines the next available number to avoid overwriting.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = os.path.splitext(os.path.basename(base_midi_path))[0]
        
        # Clean the base name by removing any existing "_opt_N" suffix before searching
        clean_base_name = re.sub(r'_opt_\d+$', '', base_name)
        sanitized_base_name = re.sub(r'[\\/*?:"<>|]', "", clean_base_name)
        
        # Create a search pattern for files like "basename_opt_*.mid"
        pattern = os.path.join(script_dir, f"{sanitized_base_name}_opt_*.mid")
        existing_files = glob.glob(pattern)
        
        max_num = 0
        if existing_files:
            # Find the highest number from the existing filenames
            for file in existing_files:
                match = re.search(r'_opt_(\d+)\.mid$', os.path.basename(file))
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        
        # The next available number is the highest found number + 1
        return max_num + 1
    except Exception as e:
        print(Fore.YELLOW + f"Could not determine next file number. Starting from 1. Reason: {e}" + Style.RESET_ALL)
        return 1

def main(resume_file_path=None):
    """
    Main function to run the music theme generation process.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    settings_file = os.path.join(script_dir, "song_settings.json")

    # --- State variables for the interactive session ---
    last_generated_themes = []
    last_generated_song_data = None
    final_song_basename = None
    # --- End State variables ---

    # --- NEW: Argument Parsing for run modes ---
    run_automatically = '--run' in sys.argv
    optimize_automatically = '--optimize' in sys.argv
    
    # If resume_file_path is NOT passed as an argument to the function, 
    # then check the command line arguments.
    if resume_file_path is None:
        if '--resume' in sys.argv:
            try:
                resume_index = sys.argv.index('--resume')
                if len(sys.argv) > resume_index + 1:
                    resume_file_path = sys.argv[resume_index + 1]
                else:
                    print(Fore.RED + "Error: --resume flag requires a file path argument." + Style.RESET_ALL)
                    return
            except ValueError:
                pass # Should not happen

    run_automatically = '--run' in sys.argv

    try:
        with open(settings_file, 'r') as f:
            previous_settings = json.load(f)
        print(Fore.CYAN + "Loaded previous song settings from file." + Style.RESET_ALL)
    except (FileNotFoundError, json.JSONDecodeError):
        previous_settings = {}
        if run_automatically:
            print(Fore.RED + "Cannot run automatically: 'song_settings.json' not found or is invalid." + Style.RESET_ALL)
            return
    
    try:
        config = load_config(CONFIG_FILE)
        genai.configure(api_key=config["api_key"])
    except (ValueError, FileNotFoundError) as e:
        print(Fore.RED + f"A critical error occurred on startup: {str(e)}" + Style.RESET_ALL)
        return

    # --- NEW: Handle different run modes ---
    if resume_file_path:
        print(Fore.CYAN + f"\n--- Resuming from progress file: {os.path.basename(resume_file_path)} ---" + Style.RESET_ALL)
        progress_data = load_progress(resume_file_path)
        if not progress_data:
             print(Fore.RED + f"Failed to load progress data from {resume_file_path}" + Style.RESET_ALL)
             return
        
        run_timestamp = progress_data.get('timestamp')
        if not run_timestamp:
            print(Fore.RED + "Timestamp missing in progress file. Cannot resume." + Style.RESET_ALL)
            return
        
        if 'generation' in progress_data.get('type', ''):
            length, defs = progress_data['length'], progress_data['theme_definitions']
            generated_themes = generate_all_themes_and_save_parts(config, length, defs, script_dir, run_timestamp, progress_data)
            if generated_themes:
                time.sleep(2)
                final_song_data, final_song_basename_val = combine_and_save_final_song(config, generated_themes, script_dir, run_timestamp)
                if final_song_data:
                    try:
                        os.remove(resume_file_path)
                        print(Fore.GREEN + "Resumed generation finished. Progress file removed." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
                    
                    # --- NEW: Populate state for the interactive session ---
                    last_generated_themes = generated_themes
                    last_generated_song_data = final_song_data
                    final_song_basename = final_song_basename_val
                    previous_settings = {
                        "length": progress_data.get('length'),
                        "theme_definitions": progress_data.get('theme_definitions')
                    }
        
        elif 'optimization' in progress_data.get('type', ''):
            theme_len, themes_opt, opt_iter = progress_data['theme_length'], progress_data['themes_to_optimize'], progress_data['opt_iteration_num']
            
            # --- NEW: Enhanced Optimization Welcome ---
            print_header("SONG OPTIMIZATION ENGINE (RESUMED)")
            print("Resuming the process to refine the last generated song, track by track.")
            print("The AI will act as a producer, focusing on improving groove, dynamics, and musicality.")
            
            user_opt_prompt = progress_data.get('user_optimization_prompt', "")
            if user_opt_prompt:
                print(Fore.CYAN + f"\nApplying user's creative direction: '{user_opt_prompt}'" + Style.RESET_ALL)
            else:
                print(Fore.CYAN + "\nRunning a general enhancement pass based on best practices." + Style.RESET_ALL)
            # --- End Enhanced Welcome ---

            optimized_themes = create_song_optimization(config, theme_len, themes_opt, script_dir, opt_iter, run_timestamp, user_opt_prompt, progress_data)
            if optimized_themes:
                time.sleep(2)
                base_name = re.sub(r'_opt_\d+$', '', progress_data.get('last_generated_song_basename', '')) or f"Final_Song_resumed_{run_timestamp}"
                opt_fname = os.path.join(script_dir, f"{base_name}_opt_{opt_iter}.mid")
                final_song_data = merge_themes_to_song_data(optimized_themes, config, theme_len)
                create_midi_from_json(final_song_data, config, opt_fname)
                try:
                    os.remove(resume_file_path)
                    print(Fore.GREEN + "Resumed optimization finished. Progress file removed." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
        # --- NEW: Do NOT exit after resume. Fall through to the interactive menu. ---

    if run_automatically:
        if not previous_settings:
            print(Fore.RED + "No settings found in 'song_settings.json' to run automatically." + Style.RESET_ALL)
            return

        print(Fore.CYAN + "\n--- Running in automatic mode ---" + Style.RESET_ALL)
        length, defs = previous_settings['length'], previous_settings['theme_definitions']
        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        generated_themes = generate_all_themes_and_save_parts(config, length, defs, script_dir, run_timestamp)
        if generated_themes:
            time.sleep(2)
            final_song_data, final_song_basename_val = combine_and_save_final_song(config, generated_themes, script_dir, run_timestamp)
            if final_song_data:
                # This state is needed for a potential 'optimize' run later
                last_generated_song_data, last_generated_themes = final_song_data, generated_themes
                
                # NEW: Save details for optimization
                details_for_optimizer = {
                    "generated_themes": generated_themes,
                    "final_song_basename": final_song_basename_val,
                    "previous_settings": previous_settings,
                    "run_timestamp": run_timestamp,
                }
                details_file = os.path.join(script_dir, "last_generation_details.json")
                with open(details_file, 'w') as f:
                    json.dump(details_for_optimizer, f, indent=4)
                print(Fore.CYAN + "Saved generation details for potential optimization." + Style.RESET_ALL)
                
                print(Fore.GREEN + "\n--- Automatic generation complete! ---" + Style.RESET_ALL)
                try:
                    prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                    if os.path.exists(prog_file):
                        os.remove(prog_file)
                        print(Fore.GREEN + "Generation finished. Progress file removed." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
        return # Exit after automatic run

    elif '--optimize' in sys.argv:
        # --- NEW: Enhanced Optimization Welcome ---
        print_header("SONG OPTIMIZATION ENGINE")
        print("This mode will refine the last generated song, track by track.")
        print("The AI will act as a producer, focusing on improving groove, dynamics, and musicality.")
        # --- End Enhanced Welcome ---

        details_file = os.path.join(script_dir, "last_generation_details.json")
        try:
            with open(details_file, 'r') as f:
                details = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(Fore.RED + "Could not find 'last_generation_details.json'. Cannot optimize." + Style.RESET_ALL)
            return

        # Extract optimization prompt from args
        try:
            opt_index = sys.argv.index('--optimize')
            # Handle prompt with spaces
            user_opt_prompt = ' '.join(sys.argv[opt_index + 1:])
        except (ValueError, IndexError):
            user_opt_prompt = ""

        if user_opt_prompt:
             print(Fore.CYAN + f"\nApplying user's creative direction: '{user_opt_prompt}'" + Style.RESET_ALL)
        else:
            print(Fore.CYAN + "\nRunning a general enhancement pass based on best practices." + Style.RESET_ALL)

        # Replicate the logic from the 'o' case in the interactive loop
        themes_to_opt = details['generated_themes']
        final_song_basename = details['final_song_basename']
        previous_settings = details['previous_settings']
        run_timestamp = details['run_timestamp']
        theme_len = previous_settings.get('length', DEFAULT_LENGTH)
        theme_length_beats = theme_len * config["time_signature"]["beats_per_bar"]

        # --- Normalization logic (copied from 'o' branch) ---
        needs_normalization = False
        if len(themes_to_opt) > 1 and themes_to_opt[0]['tracks'] and themes_to_opt[0]['tracks'][0]['notes'] and themes_to_opt[1]['tracks'] and themes_to_opt[1]['tracks'][0]['notes']:
            first_note_of_second_theme = min([n['start_beat'] for n in themes_to_opt[1]['tracks'][0]['notes']] or [0])
            if first_note_of_second_theme >= theme_length_beats:
                needs_normalization = True

        if needs_normalization:
            print(Fore.CYAN + "Normalizing absolute-timed themes to relative time for optimization..." + Style.RESET_ALL)
            normalized_themes = []
            for i, theme in enumerate(themes_to_opt):
                time_offset = i * theme_length_beats
                new_theme = theme.copy()
                new_theme['tracks'] = []
                for track in theme['tracks']:
                    new_track = track.copy()
                    new_track['notes'] = []
                    for note in track['notes']:
                        new_note = note.copy()
                        new_note['start_beat'] = max(0, round(new_note['start_beat'] - time_offset, 4))
                        new_track['notes'].append(new_note)
                    new_theme['tracks'].append(new_track)
                normalized_themes.append(new_theme)
            themes_to_opt = normalized_themes
            print(Fore.GREEN + "Normalization complete." + Style.RESET_ALL)
        # --- End Normalization ---

        start_opt_num = get_next_available_file_number(os.path.join(script_dir, final_song_basename + ".mid"))
        all_ok = True
        for i in range(config.get("number_of_iterations", 1)):
            opt_iter = start_opt_num + i
            print(Fore.CYAN + f"\n--- Opt Cycle {i + 1}/{config.get('number_of_iterations', 1)} (Version {opt_iter}) ---" + Style.RESET_ALL)
                    
            optimized_themes = create_song_optimization(config, theme_len, themes_to_opt, script_dir, opt_iter, run_timestamp, user_optimization_prompt=user_opt_prompt)
            if optimized_themes:
                time.sleep(2)
                base_name = re.sub(r'_opt_\d+$', '', final_song_basename)
                opt_fname = os.path.join(script_dir, f"{base_name}_opt_{opt_iter}.mid")
                
                # Merge the newly optimized themes correctly for the final MIDI file
                final_song_data = merge_themes_to_song_data(optimized_themes, config, theme_len)
                create_midi_from_json(final_song_data, config, opt_fname)
                
                # The newly optimized themes become the basis for the next optimization
                themes_to_opt = optimized_themes 
                last_generated_themes = optimized_themes
            else:
                print(Fore.RED + "Optimization failed. Stopping." + Style.RESET_ALL)
                all_ok = False
                break
        
        if all_ok:
            try:
                prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                if os.path.exists(prog_file): os.remove(prog_file)
                print(Fore.GREEN + "All optimizations finished. Progress file removed." + Style.RESET_ALL)
            except Exception as e: print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)

        # Clean up the details file
        try:
            os.remove(details_file)
            print(Fore.CYAN + "Cleaned up temporary generation details." + Style.RESET_ALL)
        except OSError as e:
            print(Fore.YELLOW + f"Could not remove details file: {e}" + Style.RESET_ALL)
        
        return # Exit after optimizing

    # --- Interactive Mode ---
    print(Fore.BLUE + "="*60)
    print(Style.BRIGHT + "        ⚙️ Song Generator - Interactive Mode ⚙️")
    print(Fore.BLUE + "="*60 + Style.RESET_ALL)
    
    print(f"\n{Style.BRIGHT}{Fore.CYAN}Welcome to the Generation Engine!{Style.RESET_ALL}")
    print("This script reads your configuration and brings your music to life.")
    print("From here, you can:")
    print("  - Generate music using your last defined structure.")
    print("  - Define a new song structure from scratch.")
    print("  - Optimize a previously generated song for better results.\n")
    
    print(f"{Fore.YELLOW}Note: The 'Generate New Song' option will overwrite 'song_settings.json'.{Style.RESET_ALL}\n")
    
    while True:
        print_header("Song Generator Menu")
        
        # Build the menu options dynamically
        menu_options = {}
        next_option = 1

        # Option to generate a new song
        if previous_settings:
            menu_options[str(next_option)] = ('generate_again', "Generate Again (using last settings)")
            next_option += 1
        
        menu_options[str(next_option)] = ('generate_new', "Generate New Song (define new parts)")
        next_option += 1

        # Option to optimize
        if last_generated_themes:
            menu_options[str(next_option)] = ('optimize', "Optimize Last Generated Song")
            next_option += 1

        # Option to resume
        progress_files = find_progress_files(script_dir)
        if progress_files:
            menu_options[str(next_option)] = ('resume', "Resume In-Progress Job")
            next_option += 1
        
        menu_options[str(next_option)] = ('quit', "Quit")

        # Display menu
        for key, (_, text) in menu_options.items():
            print(f"{Fore.YELLOW}{key}.{Style.RESET_ALL} {text}")
        
        user_choice_key = input(f"\n{Style.BRIGHT}{Fore.GREEN}Choose an option: {Style.RESET_ALL}").strip()
        
        action = menu_options.get(user_choice_key, (None, None))[0]

        try:
            if action == 'quit':
                print(Fore.CYAN + "Exiting. Goodbye!" + Style.RESET_ALL)
                clean_old_progress_files(script_dir)
                break
            
            elif action == 'resume':
                print_header("Resume In-Progress Job")
                for i, pfile in enumerate(progress_files[:10]):
                    basename = os.path.basename(pfile)
                    time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(pfile)))
                    info = ""
                    try:
                        with open(pfile, 'r') as f: pdata = json.load(f)
                        ptype = pdata.get('type', 'unknown')
                        if 'generation' in ptype: info = f"Gen: T {pdata.get('current_theme_index',0)+1}, Trk {pdata.get('current_track_index',0)}"
                        elif 'optimization' in ptype: info = f"Opt: T {pdata.get('current_theme_index',0)+1}, Trk {pdata.get('current_track_index',0)}"
                    except: info = "Unknown"
                    print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {basename} ({time_str}) - {info}")
                
                choice_idx = -1
                while not (0 <= choice_idx < len(progress_files[:10])):
                    try: choice_idx = int(input(f"{Fore.GREEN}Choose file (1-{len(progress_files[:10])}): {Style.RESET_ALL}").strip()) - 1
                    except ValueError: pass
                
                selected_progress_file = progress_files[choice_idx]
                # Fallback to main() to handle the resume logic
                # This is a bit of a hack to restart the process with the resume file
                # A cleaner way would be to refactor the resume logic out of the main arg parsing
                main(resume_file_path=selected_progress_file)
                # After resuming, we restart the loop to show the menu again.
                continue

            elif action in ['generate_again', 'generate_new']:
                if action == 'generate_new':
                    print_header("Define New Song Structure")

                    print(f"\n{Style.BRIGHT}Each part of your song (like a verse or chorus) will have a fixed length.{Style.RESET_ALL}")
                    len_in = input(f"{Fore.GREEN}Enter the length for each part in bars ({'/'.join(map(str, AVAILABLE_LENGTHS))}) [default: {DEFAULT_LENGTH}]: {Style.RESET_ALL}").strip()
                    length = int(len_in) if len_in.isdigit() and int(len_in) in AVAILABLE_LENGTHS else DEFAULT_LENGTH

                    print(f"\n{Style.BRIGHT}A song is made of different parts (themes). For example, a simple song might have an Intro, a Verse, and a Chorus (3 parts).{Style.RESET_ALL}")
                    num_themes_in = input(f"{Fore.GREEN}How many parts will your song have? [default: 2]: {Style.RESET_ALL}").strip()
                    num_themes = int(num_themes_in) if num_themes_in.isdigit() and int(num_themes_in) > 0 else 2
                    
                    defs = []
                    for i in range(num_themes):
                        part_char = chr(65 + i)
                        print_header(f"Defining Part {part_char} ({i + 1}/{num_themes})")

                        print(f"\n{Style.BRIGHT}The label is a name for this part (e.g., 'Intro', 'Verse_1'). It's used for the MIDI filename.{Style.RESET_ALL}")
                        print(f"{Style.DIM}Tip: A descriptive label like 'Tense_Bridge_Buildup' can also help guide the AI's composition.{Style.RESET_ALL}")
                        print(f"{Style.DIM}Tip: For parts longer than {length} bars, simply create two consecutive parts (e.g., 'Verse_A', 'Verse_B').{Style.RESET_ALL}")
                        label = input(f"{Fore.GREEN}Label for Part {part_char}: {Style.RESET_ALL}").strip() or f"Part_{part_char}"

                        print(f"\n{Style.BRIGHT}This is the most important step. Describe the musical direction for this part.{Style.RESET_ALL}")
                        print(f"{Style.DIM}You can be very specific, describing each instrument's actions in detail, or use abstract concepts to guide the AI.{Style.RESET_ALL}")
                        print(f"{Style.DIM}  - Specific Example: 'The drums play a four-on-the-floor beat. Bass enters after 4 bars. Lead synth is silent.'{Style.RESET_ALL}")
                        print(f"{Style.DIM}  - Abstract Example: 'A feeling of slowly waking up on a rainy morning. Sounds appear gradually.'{Style.RESET_ALL}")
                        print(f"{Style.DIM}Tip: To make a part shorter than {length} bars, describe the silence. e.g., 'First 4 bars: A fast synth arp. The rest is silent.'{Style.RESET_ALL}")
                        description = input(f"{Fore.CYAN}Creative direction for '{label}': {Style.RESET_ALL}").strip() or "A musical part, creatively composed by an AI."
                        
                        defs.append({'label': label, 'description': description})

                    print(f"\n{Fore.YELLOW}Note: This will save your new song structure to '{os.path.basename(settings_file)}'.{Style.RESET_ALL}")
                    previous_settings = {'length': length, 'theme_definitions': defs}
                    with open(settings_file, 'w') as f: 
                        json.dump(previous_settings, f, indent=4)
                    print(Fore.GREEN + "New song structure saved successfully." + Style.RESET_ALL)

                elif not previous_settings:
                    print(Fore.YELLOW + "No previous settings found. Choose 'Generate New Song' first." + Style.RESET_ALL); continue

                length, defs = previous_settings['length'], previous_settings['theme_definitions']
                run_timestamp = time.strftime("%Y%m%d-%H%M%S")
                
                generated_themes = generate_all_themes_and_save_parts(config, length, defs, script_dir, run_timestamp)
                if generated_themes:
                    time.sleep(2)
                    final_song_data, final_song_basename_val = combine_and_save_final_song(config, generated_themes, script_dir, run_timestamp)
                    if final_song_data:
                        last_generated_song_data = final_song_data
                        last_generated_themes = generated_themes
                        final_song_basename = final_song_basename_val
                        try:
                            prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                            if os.path.exists(prog_file): os.remove(prog_file)
                            print(Fore.GREEN + "Generation finished. Progress file removed." + Style.RESET_ALL)
                        except Exception as e: print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)

            elif action == 'optimize':
                if not last_generated_song_data:
                    print(Fore.YELLOW + "No song has been generated yet in this session to optimize." + Style.RESET_ALL); continue

                print_header("Optimize Song")
                user_opt_prompt = input(f"{Fore.CYAN}\nEnter an optional English prompt for this optimization (or press Enter to skip):\n> {Style.RESET_ALL}").strip()

                match = re.search(r'(\d{8}-\d{6})', final_song_basename)
                if not match:
                    print(Fore.RED + f"Could not get timestamp from '{final_song_basename}'. Cannot link progress." + Style.RESET_ALL); continue
                run_timestamp = match.group(1)
                
                num_opts = config.get("number_of_iterations", 1)
                themes_to_opt = last_generated_themes
                theme_len = previous_settings.get('length', DEFAULT_LENGTH)
                
                # --- Normalize themes to ensure relative timing before optimization ---
                normalized_themes = normalize_themes(themes_to_opt, theme_len, config)
                themes_to_opt = normalized_themes
                # --- End Normalization ---

                start_opt_num = get_next_available_file_number(os.path.join(script_dir, final_song_basename + ".mid"))
                all_ok = True
                for i in range(num_opts):
                    opt_iter = start_opt_num + i
                    print(Fore.CYAN + f"\n--- Opt Cycle {i + 1}/{num_opts} (Version {opt_iter}) ---" + Style.RESET_ALL)
                    
                    optimized_themes = create_song_optimization(config, theme_len, themes_to_opt, script_dir, opt_iter, run_timestamp, user_optimization_prompt=user_opt_prompt)
                    if optimized_themes:
                        time.sleep(2)
                        base_name = re.sub(r'_opt_\d+$', '', final_song_basename)
                        opt_fname = os.path.join(script_dir, f"{base_name}_opt_{opt_iter}.mid")
                        
                        # Merge the newly optimized themes correctly for the final MIDI file
                        final_song_data = merge_themes_to_song_data(optimized_themes, config, theme_len)
                        create_midi_from_json(final_song_data, config, opt_fname)
                        
                        # The newly optimized themes become the basis for the next optimization
                        themes_to_opt = optimized_themes 
                        last_generated_themes = optimized_themes
                        last_generated_song_data = final_song_data
                    else:
                        print(Fore.RED + "Optimization failed. Stopping." + Style.RESET_ALL); all_ok = False; break
                
                if all_ok:
                    try:
                        prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                        if os.path.exists(prog_file): os.remove(prog_file)
                        print(Fore.GREEN + "All optimizations finished. Progress file removed." + Style.RESET_ALL)
                    except Exception as e: print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
            
            else:
                print(Fore.YELLOW + "Invalid choice." + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"An unexpected error occurred in main loop: {e}" + Style.RESET_ALL)
            import traceback
            traceback.print_exc()

def normalize_themes(themes: List[Dict], theme_length_bars: int, config: Dict) -> List[Dict]:
    """
    Checks if themes are using absolute timing and converts them to relative timing if needed.
    """
    if not themes: return []

    theme_length_beats = theme_length_bars * config["time_signature"]["beats_per_bar"]
    
    # Check if normalization is needed by looking at the second theme's notes
    needs_normalization = False
    if len(themes) > 1:
        # Ensure the tracks and notes exist before trying to access them
        if themes[1].get('tracks') and themes[1]['tracks'][0].get('notes'):
            second_theme_notes = [note for track in themes[1]['tracks'] for note in track['notes']]
            if second_theme_notes:
                min_start_beat = min(n['start_beat'] for n in second_theme_notes)
                if min_start_beat >= theme_length_beats:
                    needs_normalization = True

    if not needs_normalization:
        return themes # No change needed

    print(Fore.CYAN + "Normalizing absolute-timed themes to relative time for optimization..." + Style.RESET_ALL)
    normalized_themes = []
    for i, theme in enumerate(themes):
        time_offset = i * theme_length_beats
        new_theme = theme.copy()
        new_theme['tracks'] = []
        for track in theme['tracks']:
            new_track = track.copy()
            new_track['notes'] = []
            for note in track['notes']:
                new_note = note.copy()
                new_note['start_beat'] = max(0, round(new_note['start_beat'] - time_offset, 4))
                new_track['notes'].append(new_note)
            new_theme['tracks'].append(new_track)
        normalized_themes.append(new_theme)
    print(Fore.GREEN + "Normalization complete." + Style.RESET_ALL)
    return normalized_themes

def merge_themes_to_song_data(themes: List[Dict], config: Dict, theme_length_bars: int) -> Dict:
    """
    Merges a list of theme dictionaries into a single song_data dictionary
    ready for MIDI file creation. It correctly offsets the notes of each theme.
    """
    merged_tracks = {}
    instrument_order = [inst['name'] for inst in config['instruments']]
    theme_length_beats = theme_length_bars * config["time_signature"]["beats_per_bar"]

    for i, theme in enumerate(themes):
        time_offset_beats = i * theme_length_beats
        for track in theme['tracks']:
            name = track['instrument_name']
            if name not in merged_tracks:
                merged_tracks[name] = {
                    "instrument_name": name,
                    "program_num": track.get('program_num', 0),
                    "role": track.get('role', 'complementary'),
                    "notes": []
                }
            # Offset the notes before adding them
            offset_notes = []
            for note in track['notes']:
                new_note = note.copy()
                new_note['start_beat'] += time_offset_beats
                offset_notes.append(new_note)
            merged_tracks[name]['notes'].extend(offset_notes)
    
    # Preserve the original instrument order
    final_tracks_sorted = [merged_tracks[name] for name in instrument_order if name in merged_tracks]

    return {
        "bpm": config["bpm"],
        "time_signature": config["time_signature"],
        "key_scale": config["key_scale"],
        "tracks": final_tracks_sorted
    }

def generate_all_themes_and_save_parts(config, length, theme_definitions, script_dir, timestamp, resume_data=None):
    """Generates all themes, saves progress track-by-track, and saves a MIDI file for each completed theme."""
    print(Fore.CYAN + "\n--- Stage 1: Generating all individual song parts... ---" + Style.RESET_ALL)
    
    # --- Resume Logic ---
    all_themes_data = []
    start_theme_index = 0
    start_track_index = 0
    used_labels = {}

    if resume_data:
        all_themes_data = resume_data.get('all_themes_data', [])
        start_theme_index = resume_data.get('current_theme_index', 0)
        start_track_index = resume_data.get('current_track_index', 0)
        used_labels = resume_data.get('used_labels', {})
        print(Fore.CYAN + f"Resuming generation from Theme {start_theme_index + 1}, Track {start_track_index + 1}" + Style.RESET_ALL)

    try:
        # --- Theme Loop ---
        for i in range(start_theme_index, len(theme_definitions)):
            theme_def = theme_definitions[i]
            
            # Ensure a placeholder for the current theme exists in the data structure
            if i >= len(all_themes_data):
                all_themes_data.append({
                    "description": theme_def.get('description', 'No description'),
                    "label": theme_def.get('label', 'Untitled Theme'),
                    "tracks": []
                })

            current_theme_data = all_themes_data[i]
            context_tracks_for_current_theme = current_theme_data['tracks']
            
            # Use a dynamic sliding window to get the maximum safe context.
            if config.get("context_window_size", -1) == -1:
                previous_themes_context, _ = get_dynamic_context(all_themes_data[:i])
            else:
                window_size = config["context_window_size"]
                start_index = max(0, i - window_size)
                previous_themes_context = all_themes_data[start_index:i]

            print(Fore.BLUE + f"\n--- Generating Theme {Style.BRIGHT}{Fore.YELLOW}{i+1}/{len(theme_definitions)}{Style.RESET_ALL}{Fore.BLUE}: '{Style.BRIGHT}{theme_def['label']}{Style.NORMAL}' ---" + Style.RESET_ALL)
            print(f"{Style.DIM}{Fore.WHITE}Blueprint: {theme_def['description']}{Style.RESET_ALL}")

            # --- Track Loop (within the current theme) ---
            track_start_index_for_this_theme = start_track_index if i == start_theme_index else 0
            
            call_has_been_made = False
            CALL_AND_RESPONSE_ROLES = {'bass', 'chords', 'arp', 'guitar', 'lead', 'melody', 'vocal'}
            
            for j in range(track_start_index_for_this_theme, len(config["instruments"])):
                instrument = config["instruments"][j]
                instrument_name, program_num, role = instrument["name"], instrument["program_num"], instrument.get("role", "complementary")
                
                dialogue_role = 'none'
                if config.get("use_call_and_response") == 1 and role in CALL_AND_RESPONSE_ROLES:
                    if not call_has_been_made:
                        dialogue_role = 'call'; call_has_been_made = True
                    else:
                        dialogue_role = 'response'
                
                print(f"\n{Fore.MAGENTA}--- Track {Style.BRIGHT}{Fore.YELLOW}{j + 1}/{len(config['instruments'])}{Style.RESET_ALL}{Fore.MAGENTA}: {Style.BRIGHT}{Fore.GREEN}{instrument_name}{Style.RESET_ALL}")
                
                track_data = generate_instrument_track_data(
                    config, length, instrument_name, program_num, 
                    context_tracks_for_current_theme, role, j, len(config['instruments']), dialogue_role,
                    theme_def['label'], theme_def['description'], previous_themes_context,
                    current_theme_index=i
                )

                if track_data:
                    # Append new track data to the current theme's track list
                    context_tracks_for_current_theme.append(track_data)
                    time.sleep(2)

                    # --- Save progress after EACH track ---
                    progress_data = {
                        'type': 'generation', 'config': config, 'length': length,
                        'theme_definitions': theme_definitions, 'timestamp': timestamp,
                        'all_themes_data': all_themes_data, 'used_labels': used_labels,
                        'current_theme_index': i,
                        'current_track_index': j + 1, # The next track to be generated
                        'total_themes': len(theme_definitions)
                    }
                    save_progress(progress_data, script_dir, timestamp)
                else:
                    print(Fore.RED + f"Failed to generate track for {instrument_name}. Stopping generation." + Style.RESET_ALL)
                    return None
            
            # --- After a theme's tracks are all generated, create its MIDI file ---
            print(Fore.GREEN + f"\n--- Theme '{theme_def['label']}' generated successfully! Saving part file... ---" + Style.RESET_ALL)
            
            # Use the original, user-provided label for the filename generation,
            # the sanitization will happen inside generate_filename.
            output_filename = generate_filename(config, script_dir, length, theme_def['label'], i, timestamp)
            current_theme_data['original_filename'] = os.path.basename(output_filename)
            
            # Create the MIDI part file for the completed theme, passing the correct time offset
            time_offset_for_this_theme = i * length * config["time_signature"]["beats_per_bar"]
            create_part_midi_from_theme(current_theme_data, config, output_filename, time_offset_for_this_theme)
            
            # Reset start_track_index for the next theme
            start_track_index = 0

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n--- Generation interrupted by user. Progress has been saved. ---" + Style.RESET_ALL)
        return None
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred during generation: {e}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None

    return all_themes_data

def combine_and_save_final_song(config, generated_themes, script_dir, timestamp):
    """Merges generated themes into a final song and saves it to a MIDI file."""
    if not generated_themes:
        print(Fore.YELLOW + "No themes were generated, cannot create a final song." + Style.RESET_ALL)
        return None, None

    print(Fore.CYAN + "\n--- Stage 2: Combining all parts into the final song... ---" + Style.RESET_ALL)

    try:
        # Merging logic (from main)
        merged_tracks = {}
        instrument_order = [inst['name'] for inst in config['instruments']]
        
        for theme in generated_themes:
            for track in theme['tracks']:
                name = track['instrument_name']
                if name not in merged_tracks:
                    merged_tracks[name] = {
                        "instrument_name": name,
                        "program_num": track.get('program_num', 0),
                        "role": track.get('role', 'complementary'),
                        "notes": []
                    }
                merged_tracks[name]['notes'].extend(track['notes'])
        
        final_tracks_sorted = [merged_tracks[name] for name in instrument_order if name in merged_tracks]

        final_song_data = {
            "bpm": config["bpm"],
            "time_signature": config["time_signature"],
            "key_scale": config["key_scale"],
            "tracks": final_tracks_sorted
        }
        
        # Filename logic (from main)
        genre = config.get("genre", "audio").replace(" ", "_").replace("/", "-")
        key = config.get("key_scale", "").replace(" ", "").replace("#", "s")
        bpm = round(float(config.get("bpm", 120)))
        
        final_filename_str = f"Final_Song_{genre}_{key}_{bpm}bpm_{timestamp}.mid"
        final_filename = os.path.join(script_dir, final_filename_str)

        create_midi_from_json(final_song_data, config, final_filename)
        
        final_song_basename = os.path.splitext(final_filename_str)[0]
        
        # Return the data needed for the optimization step
        return final_song_data, final_song_basename

    except Exception as e:
        print(Fore.RED + f"Failed to create the final combined MIDI file. Reason: {e}" + Style.RESET_ALL)
        return None, None



def get_role_instructions_for_generation(role: str, config: Dict) -> str:
    """
    Returns simplified role instructions for initial theme generation.
    """
    if role == "drums":
        return (
            "**Your Role: The Rhythmic Foundation**\n"
            "Create a strong, clear rhythmic backbone with kick, snare, and hi-hats that fits the genre."
        )
    elif role == "kick_and_snare":
        return (
            "**Your Role: The Core Beat**\n"
            f"Create the main kick and snare pattern for {config['genre']}. Focus only on kick and snare sounds."
        )
    elif role == "percussion":
        return (
            "**Your Role: Rhythmic Texture**\n"
            "Add secondary percussion (bongos, shakers, etc.) that complements the main drums."
        )
    elif role == "bass":
        return (
            "**Your Role: The Groove Foundation**\n"
            "Create a rhythmic bassline that locks with the kick and provides harmonic foundation."
        )
    elif role == "pads":
        return (
            "**Your Role: Harmonic Atmosphere**\n"
            "Provide lush harmonic foundation with sustained chords and atmospheric textures."
        )
    elif role == "lead":
        return (
            "**Your Role: The Main Hook (Lead)**\n"
            "Create the primary, most memorable, and catchy melodic hook of the song. This is the central focus for the listener and should be instantly recognizable."
        )
    elif role == "melody":
        return (
            "**Your Role: The Supporting Melody**\n"
            "Create a secondary or counter-melody that complements the lead or fills the space between other musical phrases. It should be supportive and less dominant than the lead part."
        )
    elif role in ["lead", "melody"]:
        return (
            "**Your Role: The Main Melody**\n"
            "Create the main melodic hook with clear phrasing and memorable musical ideas."
        )
    elif role == "chords":
        return (
            "**Your Role: Harmonic Structure**\n"
            "Define the chord progression with clear harmonic movement."
        )
    elif role == "arp":
        return (
            "**Your Role: Rhythmic Harmony**\n"
            "Create a hypnotic arpeggio pattern using chord notes in a repetitive rhythm."
        )
    else:
        return (
            f"**Your Role: {role.title()}**\n"
            "Create a complementary part that enhances the overall composition."
        )

def get_role_instructions_for_optimization(role: str, config: Dict) -> str:
    """
    Returns detailed role instructions for optimization with advanced techniques.
    """
    if role == "drums":
        return (
            "**Your Role: The Rhythmic Foundation**\n"
            "1. **Foundation First:** Establish a strong, clear rhythmic backbone. The kick provides the main pulse, the snare defines the backbeat.\n"
            "2. **Add Complexity:** Use hi-hats or other percussion for faster subdivisions (e.g., 16th notes) to create energy and drive.\n"
            "3. **Groove is Key:** Use subtle shifts in timing (micro-timing), velocity, and strategic rests to make the pattern feel human and groovy, not robotic."
        )
    elif role == "kick_and_snare":
        return (
            "**Your Role: The Core Beat (Kick & Snare)**\n"
            "1. **Focus on Kick and Snare:** Your ONLY job is to create the main rhythmic backbone using ONLY kick and snare sounds.\n"
            f"2. **Genre is Key:** Create a kick and snare pattern characteristic for {config['genre']}.\n"
            "3. **No Extra Percussion:** Do NOT add hi-hats, cymbals, toms, or any other percussive elements."
        )
    elif role == "bass":
        return (
            "**Your Role: The Rhythmic and Harmonic Anchor**\n"
            "1. **Don't just follow the kick, challenge it.** Create a powerful groove *with* the kick using **syncopation**. Find the rhythmic pockets the kick leaves empty.\n"
            "2. **You are the harmonic foundation.** Your notes must provide an unambiguous harmonic anchor. If there are no pads/chords, *you* define the harmony.\n"
            "3. **A great bassline is defined by its rests.** Use silence strategically to make the notes you *do* play more impactful and groovy."
        )
    elif role == "pads":
        return (
            "**Your Role: The Emotional Core and Harmonic Glue**\n"
            "1. **Don't just play static chords, create movement.** Use **smooth voice leading** and subtle **filter modulation** to make the sound evolve.\n"
            "2. **Support the lead melody.** Your pad should be the harmonic 'cushion' they sit on. Avoid playing in the same octave to prevent clashing.\n"
            "3. **You are texture, not a wall of sound.** Use long, overlapping notes that create rich, transparent texture."
        )
    elif role == "lead":
        return (
            "**Your Role: The Storyteller and Main Hook**\n"
            "1. **Think in musical sentences (phrasing).** Use rests to separate phrases and create 'question and answer' structures.\n"
            "2. **Find the pockets.** Place your phrases in the gaps left by rhythm, bass, or chords to make them stand out.\n"
            "3. **Develop the Idea:** Don't just repeat the motif. Develop it through rhythm changes, transposition, or variations.\n"
            "4. **Clear Phrasing with Rests:** Structure your melody with clear beginnings and endings."
        )
    elif role == "melody":
        return (
            "**Your Role: The Supportive Counter-Melody**\n"
            "1. **Complement, Don't Compete:** Your primary goal is to enhance the arrangement. Analyze the lead melody and find empty spaces to add your musical statement without creating clutter.\n"
            "2. **Create Dialogue:** Think of your part as an answer to the lead's call. Create short, tasteful phrases that respond to the main melody.\n"
            "3. **Subtlety is Key:** Your part should add texture and interest without stealing the spotlight. Consider using a slightly softer dynamic range than the lead."
        )
    elif role in ["lead", "melody"]:
        return (
            "**Your Role: The Storyteller and Main Hook**\n"
            "1. **Think in musical sentences (phrasing).** Use rests to separate phrases and create 'question and answer' structures.\n"
            "2. **Find the pockets.** Place your phrases in the gaps left by rhythm, bass, or chords to make them stand out.\n"
            "3. **Develop the Idea:** Don't just repeat the motif. Develop it through rhythm changes, transposition, or variations.\n"
            "4. **Clear Phrasing with Rests:** Structure your melody with clear beginnings and endings."
        )
    elif role == "chords":
        return (
            "**Your Role: The Harmonic Core**\n"
            "1. **Provide Harmony:** Establish the song's chord progression with full, rich chords.\n"
            "2. **Rhythmic vs. Sustained:** Choose between rhythmic stabbed chords or sustained atmospheric layers.\n"
            "3. **Voice Leading:** Pay attention to how notes within chords move smoothly from one chord to the next."
        )
    elif role == "arp":
        return (
            "**Your Role: The Hypnotic Arpeggio**\n"
            "1. **Chord Notes Only:** Play the notes of underlying chords one after another.\n"
            "2. **Rhythmic & Repetitive:** Establish a clear, driving pattern (e.g., 16th notes) for hypnotic effect.\n"
            "3. **Evolve Slowly:** Subtly vary the pattern over time while keeping the core rhythm consistent."
        )
    else:
        return (
            f"**Your Role: {role.title()}**\n"
            "Listen to other instruments. Find a sonic niche and rhythmic pattern that enhances the composition without cluttering."
        )

def get_progress_filename(config: Dict, run_timestamp: str) -> str:
    """Constructs a descriptive progress filename."""
    genre = config.get("genre", "audio").replace(" ", "_").replace("/", "-")
    bpm = round(float(config.get("bpm", 120)))
    genre = re.sub(r'[\\*?:"<>|]', "", genre) # Sanitize
    return f"progress_run_{genre}_{bpm}bpm_{run_timestamp}.json"

def save_progress(data: Dict, script_dir: str, run_timestamp: str) -> str:
    """Saves progress to a single, run-specific, overwritable JSON file."""
    progress_filename = get_progress_filename(data.get('config', {}), run_timestamp)
    progress_path = os.path.join(script_dir, progress_filename)
    try:
        progress_type = data.get('type', 'unknown')
        with open(progress_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(Fore.GREEN + f"Progress ({progress_type}) for run {run_timestamp} saved to {os.path.basename(progress_path)}" + Style.RESET_ALL)
        return progress_path
    except Exception as e:
        print(Fore.RED + f"Failed to save progress for run {run_timestamp}: {e}" + Style.RESET_ALL)
        return None

def load_progress(progress_path: str) -> Dict:
    """Loads progress from a JSON file."""
    try:
        with open(progress_path, 'r') as f: data = json.load(f)
        print(Fore.GREEN + f"Progress loaded from: {os.path.basename(progress_path)}" + Style.RESET_ALL)
        return data
    except Exception as e:
        print(Fore.RED + f"Failed to load progress: {e}" + Style.RESET_ALL)
        return None

def find_progress_files(script_dir: str) -> List[str]:
    """Finds all run-specific progress files in the script directory."""
    pattern = os.path.join(script_dir, "progress_run_*.json")
    progress_files = glob.glob(pattern)
    return sorted(progress_files, key=os.path.getmtime, reverse=True)

def clean_old_progress_files(script_dir: str, keep_count: int = 5):
    """
    Keeps only the most recent progress files and deletes older ones.
    """
    progress_files = find_progress_files(script_dir)
    if len(progress_files) > keep_count:
        for old_file in progress_files[keep_count:]:
            try:
                os.remove(old_file)
                print(Fore.YELLOW + f"Removed old progress file: {os.path.basename(old_file)}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.YELLOW + f"Could not remove {old_file}: {e}" + Style.RESET_ALL)

def get_optimization_goal_for_role(role: str) -> str:
    """Returns a short, user-friendly description of the optimization goal for a given role."""
    goals = {
        "drums": "Improving groove, adding fills, and enhancing dynamics.",
        "kick_and_snare": "Strengthening the core beat and ensuring it fits the genre.",
        "percussion": "Adding rhythmic complexity and texture that complements the main drums.",
        "sub_bass": "Ensuring a clean and powerful low-end foundation.",
        "bass": "Enhancing groove and syncopation to lock in with the drums.",
        "pads": "Creating smoother chord transitions and more atmospheric evolution.",
        "atmosphere": "Making the soundscape more immersive and evolving.",
        "texture": "Adding subtle 'ear candy' and refining sonic details.",
        "chords": "Refining voice leading and rhythmic interest.",
        "harmony": "Ensuring the harmonic structure strongly supports the melodies.",
        "arp": "Making the pattern more hypnotic and ensuring it evolves over time.",
        "guitar": "Refining the riff or strumming pattern to better fit the song's energy.",
        "lead": "Improving melodic phrasing, memorability, and dynamic expression.",
        "melody": "Ensuring it complements the lead and fills space effectively.",
        "vocal": "Refining the vocal-like phrasing for more emotional impact or rhythmic catchiness.",
        "fx": "Ensuring effects are impactful and create smooth transitions.",
        "complementary": "Enhancing the part to better support the overall composition."
    }
    return goals.get(role, goals["complementary"]) # Default to complementary

if __name__ == "__main__":
    main() 