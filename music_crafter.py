import os
import yaml
import google.generativeai as genai
import re
from colorama import Fore, Style, init
import json
import subprocess
import sys
import time
import glob
from ruamel.yaml import YAML
if sys.platform == "win32":
    import msvcrt

# --- CONFIGURATION ---
init(autoreset=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
SONG_SETTINGS_FILE = os.path.join(script_dir, "song_settings.json")
PART_GENERATOR_SCRIPT = os.path.join(script_dir, "part_generator.py")
SONG_GENERATOR_SCRIPT = os.path.join(script_dir, "song_generator.py")

# --- HELPER FUNCTIONS ---

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*50)
    print(f"--- {title.upper()} ---")
    print("="*50 + "\n")

def get_user_input(prompt, default=None):
    """Gets user input with a default value."""
    response = input(f"{Fore.GREEN}{prompt}{Style.RESET_ALL} ").strip()
    return response or default

def load_config():
    """Loads the base config file while preserving comments."""
    try:
        yaml = YAML()
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.load(f)
    except FileNotFoundError:
        print(Fore.RED + f"Error: The configuration file '{CONFIG_FILE}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"Error loading configuration: {str(e)}")
        sys.exit(1)

def save_config(config_data):
    """Saves the updated config data to the YAML file while preserving comments."""
    try:
        yaml = YAML()
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        print(Fore.GREEN + "Configuration file updated successfully.")
    except Exception as e:
        print(Fore.RED + f"Error saving configuration: {str(e)}")

def save_song_settings(settings_data):
    """Saves the generated song structure to the JSON file."""
    try:
        with open(SONG_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_data, f, indent=4)
        print(Fore.GREEN + "Song settings file updated successfully.")
    except Exception as e:
        print(Fore.RED + f"Error saving song settings: {str(e)}")

def extract_config_details(file_path):
    """Extracts available roles, scales, and MIDI programs from config comments."""
    details = {"roles": [], "scales": [], "midi_programs": []}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract Roles
        roles_section_match = re.search(r"# Available Roles:(.*?)(?=\n\n)", content, re.DOTALL)
        if roles_section_match:
            roles_text = roles_section_match.group(1)
            details["roles"] = re.findall(r'#   - "(.*?)"', roles_text)

        # Extract Scales
        scales_match = re.search(r'# Available scales: (.*)', content)
        if scales_match:
            scales_text = scales_match.group(1)
            details["scales"] = [s.strip().replace('"', '') for s in scales_text.split(',')]

        # Extract MIDI Programs
        midi_section_match = re.search(r"# ALL MIDI Program Numbers\n(.*)", content, re.DOTALL)
        if midi_section_match:
            midi_text = midi_section_match.group(1)
            details["midi_programs"] = re.findall(r"# (\d+\..*)", midi_text)

    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not parse details from config.yaml: {e}")
    
    return details

def call_generative_model(prompt_text, config):
    """A centralized function to call the generative model with robust retries."""
    max_retries = 3
    while True: # Loop for user-prompted retries
        for attempt in range(max_retries):
            try:
                # Give a slightly more descriptive message depending on the prompt content
                task_description = "Expanding inspiration" if "expand this" in prompt_text else "Generating content"
                print(Fore.BLUE + f"Attempt {attempt + 1}/{max_retries}: Calling generative AI ({task_description})..." + Style.RESET_ALL)
                
                model = genai.GenerativeModel(model_name=config["model_name"])
                
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]

                response = model.generate_content(
                    prompt_text,
                    safety_settings=safety_settings
                )

                if not response.candidates or response.candidates[0].finish_reason not in [1, "STOP"]:
                    finish_reason_name = "UNKNOWN"
                    if response.candidates:
                        try:
                            finish_reason_name = response.candidates[0].finish_reason.name
                        except AttributeError:
                            finish_reason_name = str(response.candidates[0].finish_reason)
                    print(Fore.RED + f"Error on attempt {attempt + 1}: Generation failed or was incomplete." + Style.RESET_ALL)
                    print(Fore.YELLOW + f"Reason: {finish_reason_name}" + Style.RESET_ALL)
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                         print(Fore.YELLOW + f"Block Reason: {response.prompt_feedback.block_reason.name}" + Style.RESET_ALL)
                    continue

                print(Fore.GREEN + "AI call successful." + Style.RESET_ALL)
                return response.text

            except Exception as e:
                if "429" in str(e) and "quota" in str(e).lower():
                    print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: API quota exceeded." + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"An unexpected error occurred on attempt {attempt + 1}: {str(e)}" + Style.RESET_ALL)
            
                if attempt < max_retries - 1:
                    wait_time = 5
                    print(Fore.YELLOW + f"Waiting for {wait_time} seconds before retrying..." + Style.RESET_ALL)
                    time.sleep(wait_time)

        print(Fore.RED + f"Failed to get a valid response from the AI after {max_retries} attempts." + Style.RESET_ALL)
        
        print(Fore.CYAN + "Automatic retry in 60 seconds..." + Style.RESET_ALL)
        print(Fore.YELLOW + "Press 'y' to retry now, 'n' to cancel the setup." + Style.RESET_ALL)

        user_action = None
        for i in range(60, 0, -1):
            if sys.platform == "win32" and msvcrt.kbhit():
                char = msvcrt.getch().decode().lower()
                if char in ['y', 'n']:
                    user_action = char
                    break
            print(f"  Retrying in {i} seconds...  ", end="\r")
            time.sleep(1)
        
        print("                               ", end="\r")

        if user_action == 'n':
            print(Fore.RED + "Setup cancelled by user." + Style.RESET_ALL)
            sys.exit(0)
        
        print(Fore.CYAN + "Retrying now..." + Style.RESET_ALL)

def generate_instrument_list_with_ai(genre, inspiration, num_instruments, config, config_details):
    """Uses AI to generate a list of instruments."""
    
    roles_list_str = ", ".join([f'"{r}"' for r in config_details.get("roles", [])])
    roles_prompt_part = ""
    if roles_list_str:
        roles_prompt_part = f"You MUST choose a role from this specific list: {roles_list_str}."

    midi_programs_list_str = "\n".join(config_details.get("midi_programs", []))
    midi_programs_prompt_part = ""
    if midi_programs_list_str:
        midi_programs_prompt_part = f"""
    **Reference MIDI Programs:**
    Use this list to select an appropriate `program_num`.
    ```
    {midi_programs_list_str}
    ```
    """

    prompt = f"""
    Based on the following musical direction, generate a list of exactly {num_instruments} instruments suitable for a music production.

    **Genre:** {genre}
    **Inspiration/Style:** {inspiration}
    {midi_programs_prompt_part}

    Your response MUST be a valid JSON array of objects. Each object must have the following keys:
    - "name": A creative, descriptive name for the instrument (e.g., "Rolling Bass", "Ethereal Pad").
    - "program_num": An appropriate General MIDI program number (integer between 1 and 128).
    - "role": The musical role of the instrument. {roles_prompt_part}

    Example for 2 instruments:
    [
      {{
        "name": "House Kick & Snare",
        "program_num": 10,
        "role": "kick_and_snare"
      }},
      {{
        "name": "Funky Bass",
        "program_num": 34,
        "role": "bass"
      }}
    ]

    Now, generate the JSON for {num_instruments} instruments. Your entire response must be ONLY the raw JSON array.
    """
    response_text = call_generative_model(prompt, config)
    if response_text:
        try:
            # Clean up the response to ensure it's valid JSON
            json_text = response_text.strip().replace("```json", "").replace("```", "")
            instrument_list = json.loads(json_text)
            if isinstance(instrument_list, list) and len(instrument_list) == num_instruments:
                return instrument_list
            else:
                print(Fore.YELLOW + "Warning: AI did not return the expected number of instruments. Using a default list.")
                return None
        except json.JSONDecodeError:
            print(Fore.YELLOW + "Warning: Failed to decode JSON from AI response for instruments. Using a default list.")
            return None
    return None

def generate_song_structure_with_ai(genre, inspiration, instruments, num_parts, part_length, config):
    """Uses AI to generate theme definitions for a full song."""
    instrument_list_str = "\n".join([f"- {i['name']} (Role: {i['role']})" for i in instruments])
    
    prompt = f"""
    You are a creative music producer. Your task is to design the complete structure for a new song.
    The song should have exactly {num_parts} distinct parts, and each part will be {part_length} bars long.

    **Musical Context:**
    - **Genre:** {genre}
    - **Inspiration/Style:** {inspiration}
    - **Available Instruments:**
    {instrument_list_str}

    **Your Task:**
    Generate a JSON array of {num_parts} theme objects. Each object needs a "label" and a "description".

    **CRITICAL INSTRUCTIONS FOR DESCRIPTIONS:**
    1.  **MIDI-Focused:** The descriptions MUST be concrete and 100% translatable to MIDI. Focus exclusively on pitch, rhythm, velocity, timing, and note duration.
    2.  **NO Sound Design Terms:** Do NOT use words related to sound design, synthesis, or audio effects (e.g., "reverb", "delay", "filter sweep", "warm", "punchy", "crystal-clear").
    3.  **Be Specific:** Instead of "energetic drums," write "kick on every beat with high velocity (120-127), snare on beats 2 and 4."
    4.  **Instrument by Instrument:** For each part, describe what EACH of the available instruments is doing, or state if it is "completely silent".

    **Output Format:**
    Your response MUST be a valid JSON array of objects.

    Example for 2 parts:
    [
      {{
        "label": "Intro_Groove",
        "description": "Progressive Kick & Snare: Kick on beats 1 and 3, velocity 90. Snare is completely silent. Rolling Bass: Plays a simple quarter-note pattern on the root note, velocity 80. Floating Lead: Completely silent. Ethereal Pad: Sustains a single minor chord for all {part_length} bars, velocity 60."
      }},
      {{
        "label": "Main_Section",
        "description": "Progressive Kick & Snare: Kick on all four beats, velocity 120. Snare on beats 2 and 4, velocity 110. Rolling Bass: Plays a syncopated 16th-note pattern using root and fifth notes, velocity 100. Floating Lead: Plays a high-register melody with long, sustained notes. Ethereal Pad: Plays a rhythmic chord stab pattern on the off-beats."
      }}
    ]

    Now, generate the JSON for {num_parts} parts, following all instructions precisely.
    """
    response_text = call_generative_model(prompt, config)
    if response_text:
        try:
            json_text = response_text.strip().replace("```json", "").replace("```", "")
            themes = json.loads(json_text)
            if isinstance(themes, list) and len(themes) == num_parts:
                return themes
            else:
                 print(Fore.YELLOW + "Warning: AI did not return the expected number of song parts. Using placeholders.")
                 return None
        except json.JSONDecodeError:
            print(Fore.YELLOW + "Warning: Failed to decode JSON from AI response for song structure. Using placeholders.")
            return None
    return None

def expand_inspiration_with_ai(genre, inspiration, config):
    """Uses AI to expand a short user inspiration into a detailed one, while preserving the original context."""
    prompt = f"""
    A user has provided a brief musical direction. Your most important task is to expand this idea into a detailed, MIDI-focused paragraph for a music generation AI, while ensuring the user's original concept remains the central theme.

    **User's Input:**
    - **Genre:** {genre}
    - **Inspiration/Style:** "{inspiration}"

    **Your Goal:**
    Write a single-paragraph description that first states the user's core idea and then expands on it with concrete musical elements.

    **CRITICAL INSTRUCTIONS:**
    1.  **Incorporate, Don't Replace:** You MUST begin your response by incorporating the user's original inspiration. For example, if the user's input is "a sad song about a rainy day", your response should start with "Generate a track that captures the feeling of a sad song about a rainy day..." before detailing the musical elements. This ensures the original creative spark is not lost.
    2.  **Translate to MIDI:** After stating the core concept, translate it into concrete MIDI terms. Describe rhythm (e.g., "slow, sparse kick drum"), melody (e.g., "descending melodic phrases"), harmony (e.g., "sustained minor chords"), and structure.
    3.  **Preserve Specifics:** If the user mentions specific artists, songs, or any other details, these MUST be included in your expanded description.
    4.  **Avoid Sound Design:** Do NOT use words about synthesis or audio effects (like 'filter sweep', 'reverb', 'warmth').

    **Example 1 (Artist Style):**
    - **User Input:** "Progressive Psytrance in the style of Phaxe, songs like Bloom, Paraphonic."
    - **Your Expanded Output:** "Create a progressive psytrance track in a style reminiscent of Phaxe's work (e.g., Bloom, Paraphonic), anchored by a clean, driving four-on-the-floor kick and a deep, rolling 16th-note bassline. The harmony should be built on sustained, atmospheric pads playing an emotive minor key progression. Weave in a hypnotic, bouncing 16th-note arpeggio, and top it with a simple, soaring lead melody made of long, legato notes."

    **Example 2 (Abstract Concept):**
    - **User Input:** "A song that sounds like a car crash."
    - **Your Expanded Output:** "Generate a track that musically interprets a car crash. It should start with a build-up of tension, represented by a fast, chaotic 16th-note arpeggio that rapidly increases in pitch. The moment of impact must be a sudden, dense, and dissonant chord cluster using a wide range of notes played with maximum velocity (127). The aftermath should be represented by a long period of silence, followed by a single, low, sustained drone note."

    Now, take the user's input "{inspiration}" and expand it into a single, detailed, MIDI-focused paragraph. Remember to incorporate the original idea directly into your response.
    """
    response_text = call_generative_model(prompt, config)
    if response_text:
        # Remove potential markdown and quotes
        return response_text.strip().replace("```", "").replace('"', '')
    return inspiration # Fallback to original inspiration

def confirm_and_execute(target_script, config, settings=None):
    """Shows a summary and asks for confirmation before saving and executing."""
    print_header("Configuration Summary")
    
    # Use brighter colors for key values
    print(f"{Fore.CYAN}Genre:{Style.RESET_ALL} {Style.BRIGHT}{config.get('genre', 'N/A')}{Style.RESET_ALL}")
    # Use dim for the long inspiration string
    inspiration_text = config.get('inspiration', 'N/A')
    inspiration_preview = (inspiration_text[:120] + '...') if len(inspiration_text) > 123 else inspiration_text
    print(f"{Fore.CYAN}Inspiration:{Style.RESET_ALL} {Style.DIM}{inspiration_preview}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}BPM:{Style.RESET_ALL} {Style.BRIGHT}{config.get('bpm', 'N/A')}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Key:{Style.RESET_ALL} {Style.BRIGHT}{config.get('key_scale', 'N/A')}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}The settings above will be saved to 'config.yaml'.{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Instruments ({len(config.get('instruments', []))}):{Style.RESET_ALL}")
    for i, inst in enumerate(config.get('instruments', [])):
        print(f"  {i+1}. {Fore.GREEN}{inst['name']}{Style.RESET_ALL} (Role: {Fore.YELLOW}{inst['role']}{Style.RESET_ALL}, MIDI Program: {Style.BRIGHT}{inst['program_num']}{Style.RESET_ALL})")

    if settings:
        print(f"\n{Fore.CYAN}Song Structure ({len(settings.get('theme_definitions', []))} Parts):{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Length per Part:{Style.RESET_ALL} {Style.BRIGHT}{settings.get('length', 'N/A')}{Style.RESET_ALL} bars")
        for i, theme in enumerate(settings.get('theme_definitions', [])):
            description_preview = theme['description']
            if len(description_preview) > 75:
                description_preview = description_preview[:72] + "..."
            print(f"  Part {i+1}: {Fore.GREEN}{theme['label']}{Style.RESET_ALL} - {Style.DIM}{description_preview}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}The detailed descriptions for each part will be saved to 'song_settings.json'.{Style.RESET_ALL}")

    confirmation = get_user_input("\nProceed to save these settings and start generation? (y/n):", "y").lower()
    
    if confirmation == 'y':
        print("\nSaving configuration to 'config.yaml' and 'song_settings.json'...")
        save_config(config)
        if settings:
            save_song_settings(settings)
        
        print_header("Starting Generation")
        try:
            # We use sys.executable to ensure we're using the same python interpreter
            # that is running the setup script.
            subprocess.run([sys.executable, target_script, '--run'], check=True)
            print(Fore.GREEN + "\nGeneration script finished successfully!")
            return True # Indicate that generation was started
        except subprocess.CalledProcessError as e:
            print(Fore.RED + f"\nError during generation script execution: {e}")
        except FileNotFoundError:
            print(Fore.RED + f"\nError: Could not find the script '{target_script}'. Make sure it's in the same directory.")
    else:
        print(Fore.YELLOW + "Generation cancelled by user. Restarting setup...")
    
    return False # Indicate that generation was cancelled

def find_progress_files(script_dir: str) -> list:
    """Finds all run-specific progress files in the script directory."""
    pattern = os.path.join(script_dir, "progress_run_*.json")
    progress_files = glob.glob(pattern)
    return sorted(progress_files, key=os.path.getmtime, reverse=True)

def get_musical_parameters_with_ai(genre, inspiration, config, config_details):
    """Uses AI to suggest BPM and Key/Scale based on genre and inspiration."""
    
    scales_list_str = ", ".join(config_details.get("scales", []))
    scales_prompt_part = ""
    if scales_list_str:
        scales_prompt_part = f"\\n\\n**Available Scales:**\\nYou MUST choose a scale from this list: {scales_list_str}."

    prompt = f"""
    You are an expert music producer. Based on the user's creative direction, suggest the most appropriate BPM and Key/Scale.

    **User's Input:**
    - **Genre:** {genre}
    - **Inspiration/Style:** {inspiration}

    **Your Task:**
    Analyze the user's input and determine an optimal BPM and a fitting Key/Scale. The key should include a scale type (e.g., 'minor', 'major', 'dorian').{scales_prompt_part}

    **Output Format:**
    Your response MUST be a single, valid JSON object with two keys: "bpm" (integer) and "key_scale" (string).

    **Example 1:**
    - **User Input:** "Progressive Psytrance like Astrix"
    - **Your Output:** {{"bpm": 140, "key_scale": "F# minor"}}

    **Example 2:**
    - **User Input:** "A slow, melancholic ambient track that sounds like a rainy day."
    - **Your Output:** {{"bpm": 75, "key_scale": "C# minor"}}

    Now, provide the JSON for the given user input.
    """
    response_text = call_generative_model(prompt, config)
    if response_text:
        try:
            json_text = response_text.strip().replace("```json", "").replace("```", "")
            params = json.loads(json_text)
            if isinstance(params, dict) and "bpm" in params and "key_scale" in params:
                return params
        except json.JSONDecodeError:
            print(Fore.YELLOW + "Warning: Failed to decode JSON from AI response for musical parameters.")
    return None

# --- MAIN LOGIC ---

def main():
    print(Fore.MAGENTA + "="*60)
    print(Style.BRIGHT + "        🎵 Welcome to the Contextual Music Crafter 🎵")
    print(Fore.MAGENTA + "="*60 + Style.RESET_ALL)
    
    print(f"\n{Style.BRIGHT}{Fore.CYAN}This is your creative partner for making music.{Style.RESET_ALL}")
    print("This script will guide you through a few simple steps:")
    print("  1. Define a musical idea (genre, style, concept).")
    print("  2. The AI will expand on your idea and create a full song plan.")
    print("  3. The plan is used to generate the final MIDI music files.\n")

    print(f"{Fore.YELLOW}Important: The 'New Song' and 'Single Part' modes will overwrite 'config.yaml' and 'song_settings.json' with your new ideas.{Style.RESET_ALL}\n")
    
    print(f"{Style.BRIGHT}Let's get started!{Style.RESET_ALL}")
    
    while True: # Loop to allow restarting the whole process
        config = load_config()
        config_details = extract_config_details(CONFIG_FILE) # Extract details from comments
        try:
            genai.configure(api_key=config["api_key"])
        except Exception as e:
            print(Fore.RED + f"API Key configuration error: {e}. Please check your config.yaml.")
            break # Exit if API key is not configured

        print_header("MAIN MENU")
    
        # --- NEW: Menu with Resume option ---
        progress_files = find_progress_files(script_dir)
        
        prompt_lines = ["1. New Full Song"]
        if progress_files:
            prompt_lines.append("2. Resume In-Progress Song")
            prompt_lines.append("3. Single Part Generator")
        else:
            prompt_lines.append("2. Single Part Generator")

        mode = get_user_input("Choose an option:\n" + "\n".join(prompt_lines) + "\n> ", "1")

        # Map input to a consistent action
        action = None
        if mode == '1':
            action = 'new_song'
        elif mode == '2' and progress_files:
            action = 'resume'
        elif (mode == '2' and not progress_files) or (mode == '3' and progress_files):
            action = 'part_generator'
        else:
            print(Fore.YELLOW + "Invalid choice. Restarting.")
            continue

        if action == 'part_generator':
            print_header("SINGLE PART GENERATOR")
            print("This mode lets you quickly generate a single musical part (like a loop or a specific section).")
            print(f"{Fore.YELLOW}This process will create a new 'config.yaml' and 'song_settings.json' based on your input.{Style.RESET_ALL}")
            confirm_overwrite = get_user_input("\nDo you want to continue? (y/n):", "y").lower()
            if confirm_overwrite != 'y':
                print(Fore.YELLOW + "Setup cancelled. Returning to main menu.")
                continue

            # --- PHASE 1: GATHER ALL USER INPUTS ---
            print_header("STEP 1: DEFINE YOUR MUSICAL IDEA")
            print("First, provide the general musical context for your part. This helps the AI choose the right BPM, key, and instruments.")
            
            genre = get_user_input("\nEnter the genre (e.g., 'Progressive Psytrance', 'House'):")
            inspiration = get_user_input("Describe the overall style or a creative concept (e.g., 'like the artist Com Truise', 'a soundtrack for a rainy day'):")
            num_instruments_str = get_user_input("How many instruments should be used? (default: 4):", "4")
            
            print_header("STEP 2: DEFINE THE PART'S DETAILS")
            print("Now, let's get specific about this single part.")

            part_length_str = get_user_input("\nHow many bars long should the part be? (e.g., 8, 16, 32, default: 16):", "16")
            part_label = get_user_input("Enter a label for this part (e.g., 'Funky_Bass_Loop', used for the filename):", "AI_Generated_Part")

            print(f"\n{Style.BRIGHT}This is the most important step. Describe the musical direction for this specific part.{Style.RESET_ALL}")
            print(f"{Style.DIM}You can be very specific (e.g., 'A fast, aggressive synth arpeggio that builds tension') or abstract ('a feeling of waking up').{Style.RESET_ALL}")
            part_description = get_user_input(f"{Fore.CYAN}Creative direction for '{part_label}':{Style.RESET_ALL}").strip()
            if not part_description:
                part_description = "A musical part, creatively generated by an AI." # Default description

            try:
                num_instruments = int(num_instruments_str)
                part_length = int(part_length_str)
            except ValueError:
                print(Fore.RED + "Invalid number entered. Please use integers only. Restarting setup.")
                continue

            # 3. Process with AI
            print_header("PROCESSING WITH AI - PLEASE WAIT")
        
            print("\nDetermining musical parameters (BPM, Key)...")
            musical_params = get_musical_parameters_with_ai(genre, inspiration, config, config_details)
            if musical_params:
                bpm = musical_params.get('bpm', 120)
                key_scale = musical_params.get('key_scale', 'C minor')
                print(f"{Fore.CYAN}AI Suggestion: {bpm} BPM, {key_scale}{Style.RESET_ALL}")
            else:
                print(Fore.YELLOW + "AI failed to suggest musical parameters. Using defaults from config.yaml.")
                bpm = config.get('bpm', 120)
                key_scale = config.get('key_scale', 'C minor')

            print("\nExpanding creative direction...")
            expanded_inspiration = expand_inspiration_with_ai(genre, inspiration, config)
            if not expanded_inspiration: continue

            print("\nGenerating instrument list...")
            instruments = generate_instrument_list_with_ai(genre, expanded_inspiration, num_instruments, config, config_details)
            if not instruments:
                print(Fore.YELLOW + "AI failed to generate instruments. Using default list from config.yaml.")
                instruments = config.get('instruments', [])
                if not instruments:
                    print(Fore.RED + "No instruments found in config.yaml. Cannot proceed.")
                    continue

            # 4. Create a single "theme" definition
            theme_definitions = [{"label": part_label, "description": part_description}]
            
            # 5. Finalize and Execute
            print("\nFinalizing configuration...")
        
            config['genre'] = genre
            config['inspiration'] = expanded_inspiration
            config['bpm'] = bpm
            config['key_scale'] = key_scale
            config['instruments'] = instruments

            song_settings = {
                'length': part_length,
                'theme_definitions': theme_definitions
            }
            
            # Save and execute
            if confirm_and_execute(PART_GENERATOR_SCRIPT, config, song_settings):
                 print(Fore.GREEN + "\nSingle Part Generator finished.")
            
            continue # Go back to mode selection

        if action == 'resume':
            print_header("Resume In-Progress Song")
            for i, pfile in enumerate(progress_files[:10]): # Show top 10 recent
                basename = os.path.basename(pfile)
                time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(pfile)))
                info = ""
                try:
                    with open(pfile, 'r') as f:
                        pdata = json.load(f)
                    ptype = pdata.get('type', 'unknown')
                    if 'generation' in ptype:
                        info = f"Gen: T {pdata.get('current_theme_index', 0) + 1}, Trk {pdata.get('current_track_index', 0)}"
                    elif 'optimization' in ptype:
                        info = f"Opt: T {pdata.get('current_theme_index',0)+1}, Trk {pdata.get('current_track_index',0)}"
                except:
                    info = "Unknown"
                print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {basename} ({time_str}) - {info}")

            choice_idx = -1
            while not (0 <= choice_idx < len(progress_files[:10])):
                try:
                    choice_str = get_user_input(f"Choose file to resume (1-{len(progress_files[:10])}):", "1")
                    choice_idx = int(choice_str) - 1
                except ValueError:
                    pass
            
            selected_file = progress_files[choice_idx]

            try:
                print_header("Resuming Process")
                subprocess.run([sys.executable, SONG_GENERATOR_SCRIPT, '--resume', selected_file], check=True)
                print(Fore.GREEN + "\nResumed process finished successfully!")
            except subprocess.CalledProcessError as e:
                print(Fore.RED + f"\nError during resumed execution: {e}")
            except FileNotFoundError:
                print(Fore.RED + f"\nError: Could not find the script '{SONG_GENERATOR_SCRIPT}'.")
    
            # After resuming, we can either exit or show the menu again. Let's show the menu.
            continue

        # --- Full Song Mode ('new_song' action) ---
        if action == 'new_song':
            print_header("NEW FULL SONG SETUP")
            print("You are about to define a new song from scratch.")
            print(f"{Fore.YELLOW}This process will generate new settings and overwrite your 'config.yaml' and 'song_settings.json' files.{Style.RESET_ALL}")
            confirm_overwrite = get_user_input("Do you want to continue? (y/n):", "y").lower()
            if confirm_overwrite != 'y':
                print(Fore.YELLOW + "Setup cancelled. Returning to main menu.")
                continue

            # --- PHASE 1: GATHER ALL USER INPUTS ---
            print_header("STEP 1: DEFINE YOUR SONG")
            print("In this step, you'll provide the core creative direction for your new song.")
            print("Based on your answers, the AI will generate a complete song structure for you to review.")
            print("After the initial song is created, you will have the option to further refine it with an optimization pass.\n")
            genre = get_user_input("Enter the genre (e.g., 'Progressive Psytrance', 'House'):")
            inspiration = get_user_input("Describe the style, artists, or a creative concept for inspiration:")
            num_instruments_str = get_user_input("How many instruments should be used? (default: 5):", "5")
            part_length_str = get_user_input("How many bars long should each part be? (e.g., 8, 16, 32, default: 16):", "16")
            num_parts_str = get_user_input("How many parts should the song have? (default: 6):", "6")

            try:
                num_instruments = int(num_instruments_str)
                part_length = int(part_length_str)
                num_parts = int(num_parts_str)
            except ValueError:
                print(Fore.RED + "Invalid number entered. Please use integers only. Restarting setup.")
                continue
            
            # --- PHASE 2: PROCESS WITH AI ---
            print_header("PROCESSING WITH AI - PLEASE WAIT")
            
            # 1. Get Musical Parameters (BPM, Key)
            print("\nDetermining musical parameters (BPM, Key)...")
            musical_params = get_musical_parameters_with_ai(genre, inspiration, config, config_details)
            if musical_params:
                bpm = musical_params.get('bpm', 120)
                key_scale = musical_params.get('key_scale', 'C minor')
                print(f"{Fore.CYAN}AI Suggestion: {bpm} BPM, {key_scale}{Style.RESET_ALL}")
            else:
                print(Fore.YELLOW + "AI failed to suggest musical parameters. Using defaults from config.yaml.")
                bpm = config.get('bpm', 120)
                key_scale = config.get('key_scale', 'C minor')

            # 2. Expand Inspiration
            print("\nExpanding creative direction...")
            expanded_inspiration = expand_inspiration_with_ai(genre, inspiration, config)
            if not expanded_inspiration:
                continue # Exit if AI call fails and user cancels

            # 3. Generate Instrument List
            print("\nGenerating instrument list...")
            instruments = generate_instrument_list_with_ai(genre, expanded_inspiration, num_instruments, config, config_details)
            if not instruments:
                print(Fore.YELLOW + "AI failed to generate instruments. Using default list from config.yaml.")
                instruments = config.get('instruments', []) # Fallback to config
                if not instruments:
                    print(Fore.RED + "No instruments found in config.yaml. Cannot proceed.")
                    continue

            # 4. Generate Song Structure
            print("\nGenerating song structure...")
            theme_definitions = generate_song_structure_with_ai(genre, expanded_inspiration, instruments, num_parts, part_length, config)
            if not theme_definitions:
                 print(Fore.YELLOW + f"AI failed to generate song structure. Creating {num_parts} placeholder parts.")
                 theme_definitions = [{"label": f"Part_{i+1}", "description": "This is a placeholder description."} for i in range(num_parts)]

            # --- PHASE 3: FINALIZE AND EXECUTE ---
            print("\nFinalizing configuration...")
            
            # Update config with generated values
            config['genre'] = genre
            config['inspiration'] = expanded_inspiration
            config['bpm'] = bpm
            config['key_scale'] = key_scale
            config['instruments'] = instruments

            # Create the song settings structure
            song_settings = {
                'length': part_length,
                'theme_definitions': theme_definitions
            }
            
            # Confirm, save, and execute the main generation script
            if not confirm_and_execute(SONG_GENERATOR_SCRIPT, config, song_settings):
                # If user cancels, the loop in main() will restart the process.
                continue
            
            # --- PHASE 4: OPTIMIZE (NEW) ---
            details_file = os.path.join(script_dir, "last_generation_details.json")
            if os.path.exists(details_file):
                print_header("STEP 4: OPTIMIZE SONG")
                print("The optimization pass revisits each track of your song with a 'producer's mindset'.")
                print("The AI will analyze the musical context and try to:")
                print(f"  - {Fore.CYAN}Enhance Groove:{Style.RESET_ALL} Add subtle syncopation or timing shifts to make rhythms feel more human.")
                print(f"  - {Fore.CYAN}Exaggerate Dynamics:{Style.RESET_ALL} Make quiet parts quieter and loud parts louder for more expression.")
                print(f"  - {Fore.CYAN}Develop Ideas:{Style.RESET_ALL} Introduce subtle variations to looping parts to keep them interesting over time.")
                print(f"  - {Fore.CYAN}Improve Ensemble Playing:{Style.RESET_ALL} Refine how different instrument parts interact with each other.")
                
                optimize_choice = get_user_input("\nDo you want to run optimizations on the generated song? (y/n):", "y").lower()
                if optimize_choice == 'y':
                    user_opt_prompt = get_user_input("Enter an optional English prompt to guide this optimization (e.g., 'make it sound more melancholic'), or press Enter for a general enhancement pass:")
                    
                    print_header("Starting Optimization")
                    try:
                        # Construct command with the prompt
                        command = [sys.executable, SONG_GENERATOR_SCRIPT, '--optimize']
                        if user_opt_prompt:
                            # Pass the prompt as separate arguments if it contains spaces
                            command.extend(user_opt_prompt.split())

                        subprocess.run(command, check=True)
                        print(Fore.GREEN + "\nOptimization script finished successfully!")
                    except subprocess.CalledProcessError as e:
                        print(Fore.RED + f"\nError during optimization script execution: {e}")
                    except FileNotFoundError:
                        print(Fore.RED + f"\nError: Could not find the script '{SONG_GENERATOR_SCRIPT}'.")
                else:
                    # Clean up the details file if user chooses not to optimize
                    os.remove(details_file)
                    print(Fore.YELLOW + "Skipping optimization.")

            print(Fore.CYAN + "\nMusic Crafter setup complete. Exiting.")
            break

if __name__ == "__main__":
    main() 