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
from ruamel.yaml.comments import CommentedMap, CommentedSeq
if sys.platform == "win32":
    import msvcrt

# --- CONFIGURATION ---
init(autoreset=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
SONG_SETTINGS_FILE = os.path.join(script_dir, "song_settings.json")
PART_GENERATOR_SCRIPT = os.path.join(script_dir, "song_generator.py")
SONG_GENERATOR_SCRIPT = os.path.join(script_dir, "song_generator.py")

# --- NEW: Global state for API key rotation ---
API_KEYS = []
CURRENT_KEY_INDEX = 0

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
        yaml = YAML(typ='rt')  # round-trip to preserve comments
        yaml.preserve_quotes = True
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.load(f)
    except FileNotFoundError:
        print(Fore.RED + f"Error: The configuration file '{CONFIG_FILE}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"Error loading configuration: {str(e)}")
        sys.exit(1)

def _sanitize_instruments_for_yaml(inst_list):
    """Ensure instruments is a proper sequence of mappings for YAML dump."""
    seq = CommentedSeq()
    if isinstance(inst_list, list):
        for inst in inst_list:
            if not isinstance(inst, dict):
                continue
            m = CommentedMap()
            # preserve order: name, program_num, role
            m["name"] = inst.get("name", "Instrument")
            try:
                pn = int(inst.get("program_num", 0))
            except Exception:
                pn = 0
            # Clamp to valid GM range 0..127
            if pn < 0 or pn > 127:
                pn = max(0, min(127, pn))
            m["program_num"] = pn
            m["role"] = inst.get("role", "complementary")
            seq.append(m)
    return seq

def _merge_config_values(doc: "CommentedMap", new_values: dict) -> "CommentedMap":
    """Merge selected keys from plain dict into the round-trip doc to preserve comments."""
    if not isinstance(doc, CommentedMap):
        return doc
    # Simple scalars/structures we allow updating
    keys_to_update = [
        "genre", "inspiration", "bpm", "key_scale", "model_name", "temperature",
        "automation_settings", "max_output_tokens", "context_window_size",
        "use_call_and_response", "number_of_iterations", "time_signature"
    ]
    for k in keys_to_update:
        if k in new_values:
            doc[k] = new_values[k]
    # Instruments handled with sanitized sequence
    if isinstance(new_values.get("instruments"), list):
        doc["instruments"] = _sanitize_instruments_for_yaml(new_values["instruments"])
    return doc

def save_config(config_data):
    """Saves the updated config data to the YAML file while preserving comments."""
    try:
        yaml = YAML(typ='rt')  # round-trip to preserve comments
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Load existing doc to keep all comments and only merge updated fields
        with open(CONFIG_FILE, 'r', encoding='utf-8') as rf:
            existing_doc = yaml.load(rf)
            if existing_doc is None:
                existing_doc = CommentedMap()
        merged_doc = _merge_config_values(existing_doc, config_data if isinstance(config_data, dict) else {})

        # Ensure trailing reference comments persist after the last key (usually 'instruments')
        def _ensure_trailing_reference_comments(doc: "CommentedMap") -> "CommentedMap":
            try:
                if not isinstance(doc, CommentedMap):
                    return doc
                # Global guard: if the reference block already exists anywhere in the file, don't add it again
                try:
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as _rf_check:
                        _raw = _rf_check.read()
                    if "MIDI Technical Reference (informational)" in _raw or "Troubleshooting (quick reference)" in _raw:
                        return doc
                except Exception:
                    pass
                keys = list(doc.keys())
                if not keys:
                    return doc
                last_key = keys[-1]
                ca = getattr(doc, 'ca', None)
                existing_after = ""
                if ca and isinstance(ca.items, dict) and last_key in ca.items:
                    entry = ca.items[last_key]
                    # entry: [pre, eol, post, ...], 'post' at index 1 holds after-comment lines
                    if entry and len(entry) > 1 and entry[1]:
                        existing_after = "\n".join([t.value if hasattr(t, 'value') else str(t) for t in entry[1]])
                if "Troubleshooting (quick reference)" not in existing_after and "MIDI Technical Reference (informational)" not in existing_after:
                    block = """
----------------------------------------------------------------------------- 
Troubleshooting (quick reference)
----------------------------------------------------------------------------- 
- 429 / quota exceeded: rotate API keys or pause and resume later.
- 5xx internal errors/timeouts: retry later; switching model (e.g., gemini-2.5-flash) can help.
- Empty/invalid response: usually transient under load; try again or lower temperature.

--- MIDI Technical Reference (informational) ---
General MIDI Channel Assignments:
- Channel 10 (index 9): Drums & Percussion.
- Channels 1-9 & 11-16: Melodic & Tonal Instruments. Assigned sequentially, skipping channel 10.

ALL MIDI Program Numbers

1-8: Pianos
1. Acoustic Grand Piano
2. Bright Acoustic Piano
3. Electric Grand Piano
4. Honky-Tonk Piano
5. Electric Piano 1
6. Electric Piano 2
7. Harpsichord
8. Clavinet

9-16: Chromatic Percussion
9. Celesta
10. Glockenspiel
11. Music Box
12. Vibraphone
13. Marimba
14. Xylophone
15. Tubular Bells
16. Dulcimer

17-24: Organs
17. Drawbar Organ
18. Percussive Organ
19. Rock Organ
20. Church Organ
21. Reed Organ
22. Accordion
23. Harmonica
24. Tango Accordion

25-32: Guitars
25. Acoustic Guitar (nylon)
26. Acoustic Guitar (steel)
27. Electric Guitar (jazz)
28. Electric Guitar (clean)
29. Electric Guitar (muted)
30. Overdriven Guitar
31. Distortion Guitar
32. Guitar Harmonics

33-40: Basses
33. Acoustic Bass
34. Electric Bass (finger)
35. Electric Bass (pick)
36. Fretless Bass
37. Slap Bass 1
38. Slap Bass 2
39. Synth Bass 1
40. Synth Bass 2

41-48: Strings
41. Violin
42. Viola
43. Cello
44. Contrabass
45. Tremolo Strings
46. Pizzicato Strings
47. Orchestral Harp
48. Timpani

49-56: Ensemble
49. String Ensemble 1
50. String Ensemble 2
51. SynthStrings 1
52. SynthStrings 2
53. Choir Aahs
54. Voice Oohs
55. Synth Voice
56. Orchestra Hit

57-64: Brass
57. Trumpet
58. Trombone
59. Tuba
60. Muted Trumpet
61. French Horn
62. Brass Section
63. Synth Brass 1
64. Synth Brass 2

65-72: Reed
65. Soprano Sax
66. Alto Sax
67. Tenor Sax
68. Baritone Sax
69. Oboe
70. English Horn
71. Bassoon
72. Clarinet

73-80: Pipe
73. Piccolo
74. Flute
75. Recorder
76. Pan Flute
77. Blown Bottle
78. Shakuhachi
79. Whistle
80. Ocarina

81-88: Synth Lead
81. Lead 1 (square)
82. Lead 2 (sawtooth)
83. Lead 3 (calliope)
84. Lead 4 (chiff)
85. Lead 5 (charang)
86. Lead 6 (voice)
87. Lead 7 (fifths)
88. Lead 8 (bass + lead)

89-96: Synth Pad
89. Pad 1 (new age)
90. Pad 2 (warm)
91. Pad 3 (polysynth)
92. Pad 4 (choir)
93. Pad 5 (bowed)
94. Pad 6 (metallic)
95. Pad 7 (halo)
96. Pad 8 (sweep)

97-104: Synth Effects
97. FX 1 (rain)
98. FX 2 (soundtrack)
99. FX 3 (crystal)
100. FX 4 (atmosphere)
101. FX 5 (brightness)
102. FX 6 (goblins)
103. FX 7 (echoes)
104. FX 8 (sci-fi)

105-112: Ethnic
105. Sitar
106. Banjo
107. Shamisen
108. Koto
109. Kalimba
110. Bagpipe
111. Fiddle
112. Shanai

113-120: Percussive
113. Tinkle Bell
114. Agogo
115. Steel Drums
116. Woodblock
117. Taiko Drum
118. Melodic Tom
119. Synth Drum
120. Reverse Cymbal

121-128: Sound Effects
121. Guitar Fret Noise
122. Breath Noise
123. Seashore
124. Bird Tweet
125. Telephone Ring
126. Helicopter
127. Applause
128. Gunshot
"""
                    # Attach as after-comment to the last key
                    doc.yaml_set_comment_before_after_key(last_key, after=block, indent=2)
            except Exception:
                pass
            return doc

        merged_doc = _ensure_trailing_reference_comments(merged_doc)

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(merged_doc, f)
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

        # Extract Scales (multi-line aware)
        scales_section_match = re.search(r"# Available scales:(.*?)(?=\n\w)", content, re.DOTALL)
        if scales_section_match:
            scales_text = scales_section_match.group(1)
            # Clean up the text: remove newlines and comment characters
            cleaned_text = scales_text.replace('\n', ' ').replace('#', '')
            # Split by comma and clean up each item, removing empty strings
            details["scales"] = [s.strip().strip('"') for s in cleaned_text.split(',') if s.strip()]

        # Extract MIDI Programs
        midi_section_match = re.search(r"# ALL MIDI Program Numbers\n(.*)", content, re.DOTALL)
        if midi_section_match:
            midi_text = midi_section_match.group(1)
            details["midi_programs"] = re.findall(r"# (\d+\..*)", midi_text)

    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not parse details from config.yaml: {e}")
    
    return details

def normalize_automation_settings(config):
    """Ensures automation settings are sane; fills defaults and guardrails.
    - If use_cc_automation == 1 and allowed_cc_numbers is missing/empty, set defaults and warn.
    - Ensures boolean-like flags are 0/1 integers.
    """
    try:
        if "automation_settings" not in config or not isinstance(config.get("automation_settings"), dict):
            config["automation_settings"] = {}
        a = config["automation_settings"]
        # Normalize flags to 0/1
        for key in ["use_pitch_bend", "use_cc_automation", "use_sustain_pedal"]:
            val = a.get(key, 0)
            try:
                a[key] = 1 if int(val) == 1 else 0
            except Exception:
                a[key] = 0
        # Ensure allowed_cc_numbers exists
        if "allowed_cc_numbers" not in a or not isinstance(a.get("allowed_cc_numbers"), list):
            a["allowed_cc_numbers"] = []
        # Guardrail: if CC automation on but no CCs, set sensible defaults
        if a.get("use_cc_automation", 0) == 1 and not a.get("allowed_cc_numbers"):
            a["allowed_cc_numbers"] = [11, 74, 1]  # Expression, Filter Cutoff, Mod Wheel
            print(Fore.YELLOW + "Automation guardrail: 'allowed_cc_numbers' was empty while CC automation was enabled. Using defaults: [11, 74, 1]." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not normalize automation settings: {e}" + Style.RESET_ALL)

def validate_instruments(instruments, config_details):
    """Validates and fixes a list of instrument dicts in-place.
    Rules:
    - Ensure keys: name (non-empty str), program_num in [1,128], role in allowed roles (or fallback 'complementary')
    - Convert types where possible; clamp program numbers; warn on fixes
    Returns the validated list.
    """
    try:
        if not isinstance(instruments, list):
            print(Fore.YELLOW + "Warning: Instruments is not a list. Using empty list." + Style.RESET_ALL)
            return []
        allowed_roles = config_details.get("roles", []) or []

        # Canonical set as fallback when no allowed list is present
        canonical_roles = {
            "drums", "kick_and_snare", "percussion", "sub_bass", "bass",
            "pads", "atmosphere", "texture", "chords", "harmony", "arp",
            "guitar", "lead", "melody", "vocal", "fx", "complementary"
        }

        # Direct synonyms mapping (extendable)
        role_synonyms = {
            "main_drums": "drums",
            "drumkit": "drums",
            "drum_kit": "drums",
            "drum-set": "drums",
            "drum set": "drums",
            "kit": "drums",
            "kick": "kick_and_snare",
            "kick_snare": "kick_and_snare",
            "kick+snare": "kick_and_snare",
            "kick/snare": "kick_and_snare",
            "kicksnare": "kick_and_snare",
            "snare": "kick_and_snare",
            "percussion_shaker": "percussion",
            "shaker": "percussion",
            "perc": "percussion",
            "hi-hat": "percussion",
            "hihat": "percussion",
            "hat": "percussion",
            "cowbell": "percussion",
            "clap": "percussion",
            "claps": "percussion",
            "bassline": "bass",
            "bass_line": "bass",
            "synth_bass": "bass",
            "bass_synth": "bass",
            "bass-guitar": "bass",
            "subbass": "sub_bass",
            "sub-bass": "sub_bass",
            "sub": "sub_bass",
            "low_bass": "sub_bass",
            "pad": "pads",
            "chords_pad": "pads",
            "harmony_pad": "pads",
            "string_pad": "pads",
            "harmonic_accent": "texture",
            "texture_fx": "texture",
            "ear_candy": "texture",
            "details": "texture",
            "lead_melody": "lead",
            "lead_synth": "lead",
            "main_lead": "lead",
            "hook": "lead",
            "melodic_lead": "melody",
            "melodic_line": "melody",
            "theme": "melody",
            "vox": "vocal",
            "vocal_chops": "vocal",
            "chops": "vocal",
            "sfx": "fx",
            "riser": "fx",
            "impact": "fx",
            "sweep": "fx",
            "uplifter": "fx",
            "downlifter": "fx",
            "transition": "fx",
            "whoosh": "fx",
            "noise_fx": "fx"
        }

        def keyword_map(raw_role: str) -> str:
            r = raw_role.lower().replace(" ", "_")
            if any(k in r for k in ["kick_and_snare", "kick+snare", "kick_snare", "kick", "snare"]):
                return "kick_and_snare"
            if any(k in r for k in ["drum", "drums", "drumkit", "kit"]):
                return "drums"
            if any(k in r for k in ["perc", "shaker", "tamb", "conga", "bongo", "clap", "hat", "hihat", "hi-hat", "cowbell"]):
                return "percussion"
            if "sub" in r and "bass" in r:
                return "sub_bass"
            if "bass" in r:
                return "bass"
            if any(k in r for k in ["pad", "pads"]):
                return "pads"
            if any(k in r for k in ["atmo", "atmosphere", "ambient", "soundscape", "drone"]):
                return "atmosphere"
            if "texture" in r:
                return "texture"
            if any(k in r for k in ["chord", "stabs"]):
                return "chords"
            if "harm" in r:
                return "harmony"
            if "arp" in r:
                return "arp"
            if "guitar" in r:
                return "guitar"
            if "lead" in r:
                return "lead"
            if "melody" in r or "melod" in r:
                return "melody"
            if any(k in r for k in ["vocal", "vox", "chops"]):
                return "vocal"
            if "fx" in r or any(k in r for k in ["riser", "impact", "sweep", "uplifter", "downlifter", "transition", "whoosh", "noise"]):
                return "fx"
            return "complementary"

        for idx, inst in enumerate(instruments):
            if not isinstance(inst, dict):
                instruments[idx] = {"name": f"Instrument_{idx+1}", "program_num": 1, "role": "complementary"}
                print(Fore.YELLOW + f"Warning: Replacing invalid instrument at index {idx} with a default." + Style.RESET_ALL)
                continue
            # Name
            name = inst.get("name")
            if not isinstance(name, str) or not name.strip():
                inst["name"] = f"Instrument_{idx+1}"
                print(Fore.YELLOW + f"Warning: Missing/invalid instrument name at index {idx}. Using '{inst['name']}'." + Style.RESET_ALL)
            # Program number
            prog = inst.get("program_num", 0)
            try:
                prog_int = int(prog)
            except Exception:
                prog_int = 0
            if prog_int < 0 or prog_int > 127:
                clamped = min(127, max(0, prog_int))
                print(Fore.YELLOW + f"Warning: program_num {prog} out of range at index {idx}. Clamped to {clamped}." + Style.RESET_ALL)
                prog_int = clamped
            inst["program_num"] = prog_int
            # Role
            role = inst.get("role", "complementary")
            if not isinstance(role, str) or not role:
                role = "complementary"
            # 1) direct synonyms
            mapped = role_synonyms.get(role.strip().lower(), role.strip().lower())
            # 2) keyword heuristics if still not standard
            if mapped not in (allowed_roles or canonical_roles):
                mapped = keyword_map(mapped)
            # 3) enforce allowed list if present, else canonical set
            valid_set = set(allowed_roles) if allowed_roles else canonical_roles
            if mapped not in valid_set:
                print(Fore.YELLOW + f"Warning: role '{role}' not allowed/mappable. Using 'complementary'." + Style.RESET_ALL)
                mapped = "complementary"
            role = mapped
            inst["role"] = role
        return instruments
    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not validate instruments: {e}" + Style.RESET_ALL)
        return instruments

def initialize_api_keys(config):
    """Loads API keys from config and prepares them for rotation."""
    global API_KEYS, CURRENT_KEY_INDEX
    
    api_key_config = config.get("api_key")
    if isinstance(api_key_config, list):
        API_KEYS = [key for key in api_key_config if key and "YOUR_" not in key]
    elif isinstance(api_key_config, str) and "YOUR_" not in api_key_config:
        API_KEYS = [api_key_config]
    else:
        API_KEYS = []

    CURRENT_KEY_INDEX = 0
    if not API_KEYS:
        print(Fore.RED + "Error: No valid API key found in 'config.yaml'. Please add your key(s).")
        return False
    
    try:
        genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
    except Exception as e:
        print(Fore.RED + f"Error configuring API key: {e}")
        return False

    print(Fore.CYAN + f"Found {len(API_KEYS)} API key(s). Starting with key #1.")
    return True

def get_next_api_key():
    """Rotates to the next available API key."""
    global CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    print(Fore.YELLOW + f"Switching to API key #{CURRENT_KEY_INDEX + 1}...")
    return API_KEYS[CURRENT_KEY_INDEX]

def call_generative_model(prompt_text, config):
    """Call the generative model with retries where quota/rate limits DO NOT consume attempts.

    - Quota/429/rate-limit errors trigger API key rotation and/or backoff without increasing the attempt counter.
    - Only functional failures (safety blocks, empty responses, other exceptions) consume attempts.
    """
    global CURRENT_KEY_INDEX
    max_retries = 3
    while True:  # Loop for user-prompted retries after attempt budget exhausted
        attempt_count = 0
        while attempt_count < max_retries:
            # Descriptive task label
            task_description = "Expanding inspiration" if "expand this" in (prompt_text or "") else "Generating content"
            print(Fore.BLUE + f"Attempt {attempt_count + 1}/{max_retries}: Calling generative AI ({task_description})..." + Style.RESET_ALL)

            # Build generation config; enforce JSON MIME only for JSON prompts
            wants_json = "JSON" in (prompt_text or "")
            generation_config = {
                "temperature": config.get("temperature", 1.0)
            }
            if wants_json:
                generation_config["response_mime_type"] = "application/json"
            if isinstance(config.get("max_output_tokens"), int):
                generation_config["max_output_tokens"] = config.get("max_output_tokens")

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            # Inner rotation loop: try current key, rotate on quota without consuming attempt
            started_key_index = CURRENT_KEY_INDEX
            keys_tried = 0
            quota_only_failures = True
            while keys_tried < max(1, len(API_KEYS)):
                try:
                    model = genai.GenerativeModel(
                        model_name=config["model_name"],
                        generation_config=generation_config
                    )
                    response = model.generate_content(
                        prompt_text,
                        safety_settings=safety_settings,
                    )

                    # Safety block check (counts as a functional failure)
                    if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                        quota_only_failures = False
                        print(Fore.RED + f"Error on attempt {attempt_count + 1}: Your prompt was blocked by the safety filter." + Style.RESET_ALL)
                        try:
                            reason_name = response.prompt_feedback.block_reason.name
                        except Exception:
                            reason_name = str(getattr(response.prompt_feedback, 'block_reason', 'UNKNOWN'))
                        print(Fore.YELLOW + f"Reason: {reason_name}. Consider rephrasing to be less aggressive/explicit." + Style.RESET_ALL)
                        break  # exit rotation loop; will consume attempt below

                    # Empty/incomplete response (counts as a functional failure)
                    if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                        quota_only_failures = False
                        finish_reason_name = "UNKNOWN"
                        if response.candidates:
                            try:
                                finish_reason_name = response.candidates[0].finish_reason.name
                            except AttributeError:
                                finish_reason_name = str(response.candidates[0].finish_reason)
                        print(Fore.RED + f"Error on attempt {attempt_count + 1}: The AI returned an empty/incomplete response." + Style.RESET_ALL)
                        print(Fore.YELLOW + f"Finish Reason: {finish_reason_name}. The script will retry." + Style.RESET_ALL)
                        break  # exit rotation loop; will consume attempt below

                    print(Fore.GREEN + "AI call successful." + Style.RESET_ALL)
                    total_token_count = 0
                    if hasattr(response, 'usage_metadata'):
                        total_token_count = response.usage_metadata.total_token_count
                    return response.text, total_token_count

                except Exception as e:
                    err = str(e).lower()
                    if ("429" in err) or ("quota" in err) or ("rate limit" in err):
                        print(Fore.YELLOW + f"Warning: API quota/rate limit for key #{CURRENT_KEY_INDEX + 1}. Rotating..." + Style.RESET_ALL)
                        # Rotate key and retry without consuming the attempt
                        if len(API_KEYS) > 1:
                            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
                            try:
                                genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                            except Exception:
                                pass
                        # Count this rotation step
                        keys_tried += 1
                        # If we've looped back to the starting key or tried all, break to backoff
                        if keys_tried >= len(API_KEYS) or CURRENT_KEY_INDEX == started_key_index:
                            break
                        continue
                    else:
                        quota_only_failures = False
                        print(Fore.RED + f"An unexpected error occurred: {str(e)}" + Style.RESET_ALL)
                        break  # functional failure; consume attempt

            # After trying keys: decide whether to consume attempt
            if quota_only_failures:
                # All failures were quota/rate-limit; do not consume attempt, backoff then continue
                wait_time = 10
                print(Fore.YELLOW + f"All keys hit quota/rate limits. Waiting {wait_time}s before probing again..." + Style.RESET_ALL)
                time.sleep(wait_time)
                continue  # same attempt_count
            else:
                # Consume an attempt for functional failure
                attempt_count += 1
                if attempt_count < max_retries:
                    wait_time = 5
                    print(Fore.YELLOW + f"Waiting {wait_time}s before retrying..." + Style.RESET_ALL)
                    time.sleep(wait_time)

        # Out of attempts → user‑prompted retry window
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
    - "program_num": An appropriate General MIDI program number (integer between 0 and 127).
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
    response_text, tokens_used = call_generative_model(prompt, config)
    
    if response_text:
        try:
            # Clean up the response to ensure it's valid JSON
            json_text = response_text.strip().replace("```json", "").replace("```", "")
            instrument_list = json.loads(json_text)
            if isinstance(instrument_list, list) and len(instrument_list) == num_instruments:
                return instrument_list, tokens_used
            else:
                print(Fore.YELLOW + "Warning: AI did not return the expected number of instruments. Using a default list.")
                return None, tokens_used
        except json.JSONDecodeError:
            print(Fore.YELLOW + "Warning: Failed to decode JSON from AI response for instruments. Using a default list.")
            return None, tokens_used
    return None, tokens_used

def generate_song_structure_with_ai(genre, inspiration, instruments, num_parts, part_length, config):
    """Uses AI to generate theme definitions for a full song."""
    instrument_list_str = "\n".join([f"- {i['name']} (Role: {i['role']})" for i in instruments])
    
    # --- NEW: Dynamically add automation instructions to the prompt ---
    automation_instructions = ""
    automation_settings = config.get("automation_settings", {})
    use_pitch_bend = automation_settings.get("use_pitch_bend", 0) == 1
    use_cc_automation = automation_settings.get("use_cc_automation", 0) == 1
    use_sustain_pedal = automation_settings.get("use_sustain_pedal", 0) == 1

    if use_pitch_bend or use_cc_automation or use_sustain_pedal:
        automation_instructions += "\n5. **Describe Expressive Automations:**"
        if use_pitch_bend:
            automation_instructions += "\n   - **Pitch Bend:** For expressive roles like `lead`, `bass`, `melody`, `vocal`, or `guitar`, describe pitch slides and bends (e.g., 'a fast pitch bend up into the note')."
        if use_cc_automation:
            allowed_ccs = ", ".join(map(str, automation_settings.get("allowed_cc_numbers", [])))
            automation_instructions += f"\n   - **CC Automation:** For synth-based or textural roles like `pads`, `atmosphere`, `lead`, `bass`, `arp`, `texture`, `fx`, or `riser`, describe automations like filter sweeps using allowed CCs ({allowed_ccs})."
        if use_sustain_pedal:
            automation_instructions += "\n   - **Sustain Pedal:** For sustaining instruments like `piano`, `pads`, `chords`, `harmony`, `atmosphere`, `guitar`, or `melody`, describe when the pedal is pressed/released."
    # --- END NEW ---

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

    **CRITICAL INSTRUCTIONS FOR DESCRIPTIONS (NEW CREATIVE APPROACH):**
    1.  **Focus on Vibe and Emotion:** Your primary goal is to describe the *feeling*, *mood*, and *musical role* of each instrument. Use evocative and creative language (e.g., "a melancholic piano melody," "an aggressive, driving bassline," "a floating, atmospheric pad").
    2.  **Describe the Musical Narrative:** Explain how the parts evolve. Does the energy build? Does an instrument become more complex? Does a new feeling emerge? Tell a small story for each part.
    3.  **Use Technical Terms as Support:** You can and should include specific musical details to guide the next AI, but they should support your creative description, not replace it. Instead of just "Kick on all four beats," write "A powerful, driving four-on-the-floor kick pattern that anchors the track's high energy."
    4.  **Instrument by Instrument:** For each part, describe what EACH of the available instruments is doing, or state if it is "silent." This structure remains critical.{automation_instructions}

    **Output Format:**
    Your response MUST be a valid JSON array of objects.

    Example for 2 parts (NEW CREATIVE APPROACH):
    [
      {{
        "label": "Tense_Intro",
        "description": "The track begins with a sense of suspense. The 'Progressive Kick' is completely silent. The 'Rolling Bass' introduces a simple, hypnotic pulse on the root note, with a low velocity, creating a feeling of something lurking. The 'Floating Lead' is silent. The 'Ethereal Pad' holds a single, sustained minor chord that feels cold and spacious. A slow filter sweep (CC 74) gradually opens over the full {part_length} bars, slowly building tension."
      }},
      {{
        "label": "Groove_Establishment",
        "description": "The energy level rises as the core groove is established. The 'Progressive Kick' now lays down a solid four-on-the-floor beat with a confident, high velocity. The 'Rolling Bass' becomes more energetic and complex, playing a syncopated 16th-note pattern that adds momentum. The 'Floating Lead' is still silent, saving its entry for later. The 'Ethereal Pad' shifts to a rhythmic, pulsing pattern that complements the bassline, adding to the driving feel."
      }}
    ]

    Now, generate the JSON for {num_parts} parts, following the new creative instructions precisely.
    """
    response_text, tokens_used = call_generative_model(prompt, config)
    if response_text:
        try:
            json_text = response_text.strip().replace("```json", "").replace("```", "")
            themes = json.loads(json_text)
            if isinstance(themes, list) and len(themes) == num_parts:
                return themes, tokens_used
            else:
                 print(Fore.YELLOW + "Warning: AI did not return the expected number of song parts. Using placeholders.")
                 return None, tokens_used
        except json.JSONDecodeError:
            print(Fore.YELLOW + "Warning: Failed to decode JSON from AI response for song structure. Using placeholders.")
            return None, tokens_used
    return None, tokens_used

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
    response_text, tokens_used = call_generative_model(prompt, config)
    if response_text:
        # Remove potential markdown and quotes
        return response_text.strip().replace("```", "").replace('"', ''), tokens_used
    return inspiration, tokens_used # Fallback to original inspiration

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
    """Uses AI to suggest BPM and Key/Scale based on genre and inspiration, leveraging the AI's musical knowledge."""
    
    scales_list_str = ", ".join(f'"{s}"' for s in config_details.get("scales", []))
    scales_prompt_part = ""
    if scales_list_str:
        scales_prompt_part = f"You MUST choose a scale from this specific list: {scales_list_str}."

    prompt = f"""
    You are an expert music producer and historian with a deep knowledge of music theory across all genres.
    Based on the user's creative direction, suggest the most appropriate BPM and Key/Scale.

    **User's Input:**
    - **Genre:** {genre}
    - **Inspiration/Style:** {inspiration}

    **Your Task & Logic:**
    1.  **Analyze for Specifics:** First, scan the 'Inspiration/Style' for any mention of specific artists (e.g., "like Infected Mushroom"), niche subgenres (e.g., "Future Garage"), or specific songs.
    2.  **Apply Your Knowledge:**
        *   **If specifics are found:** Use your deep musical knowledge to determine the most common or characteristic key and scale used by that artist or within that subgenre. For example, you know that many classic Psytrance artists frequently use keys like E minor, F# minor, or G minor, and often employ exotic scales like Phrygian or Byzantine.
        *   **If no specifics are found:** Analyze the emotional keywords (e.g., "melancholic," "energetic," "mysterious") and choose a scale that best fits that mood.
    3.  **Select from Available Scales:** Choose a scale from the provided list that is the closest match to your expert analysis. {scales_prompt_part}
    4.  **Determine Root Note & BPM:** Finally, choose a root note and BPM that are characteristic of the genre and inspiration.

    **Output Format:**
    Your response MUST be a single, valid JSON object with two keys: "bpm" (integer) and "key_scale" (string).

    **Example 1 (Artist-specific):**
    - **User Input:** "Psytrance in the style of Astrix"
    - **Your Output:** {{"bpm": 140, "key_scale": "F# minor"}}

    **Example 2 (Mood-specific):**
    - **User Input:** "A slow, mysterious ambient track."
    - **Your Output:** {{"bpm": 75, "key_scale": "C# phrygian"}}

    Now, provide the JSON for the given user input.
    """
    response_text, tokens_used = call_generative_model(prompt, config)
    if response_text:
        try:
            json_text = response_text.strip().replace("```json", "").replace("```", "")
            params = json.loads(json_text)
            if isinstance(params, dict) and "bpm" in params and "key_scale" in params:
                return params, tokens_used
        except json.JSONDecodeError:
            print(Fore.YELLOW + "Warning: Failed to decode JSON from AI response for musical parameters.")
    return None, tokens_used

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
        total_tokens_used = 0  # Reset token counter for each run
        config = load_config()
        normalize_automation_settings(config)
        config_details = extract_config_details(CONFIG_FILE) # Extract details from comments
        try:
            if not initialize_api_keys(config):
                break # Exit if no valid keys are found
            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
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
            prompt_lines.append("4. MPE Single Track (Standalone)")
        else:
            prompt_lines.append("2. Single Part Generator")
            prompt_lines.append("3. MPE Single Track (Standalone)")

        mode = get_user_input("Choose an option:\n" + "\n".join(prompt_lines) + "\n> ", "1")

        # Map input to a consistent action
        action = None
        if mode == '1':
            action = 'new_song'
        elif mode == '2' and progress_files:
            action = 'resume'
        elif (mode == '2' and not progress_files) or (mode == '3' and progress_files):
            action = 'part_generator'
        elif (mode == '3' and not progress_files) or (mode == '4' and progress_files):
            action = 'mpe_single_track'
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
            musical_params, tokens_used = get_musical_parameters_with_ai(genre, inspiration, config, config_details)
            total_tokens_used += tokens_used
            if musical_params:
                bpm = musical_params.get('bpm', 120)
                key_scale = musical_params.get('key_scale', 'C minor')
                print(f"{Fore.CYAN}AI Suggestion: {bpm} BPM, {key_scale}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
            else:
                print(Fore.YELLOW + "AI failed to suggest musical parameters. Using defaults from config.yaml.")
                bpm = config.get('bpm', 120)
                key_scale = config.get('key_scale', 'C minor')

            print("\nExpanding creative direction...")
            expanded_inspiration, tokens_used = expand_inspiration_with_ai(genre, inspiration, config)
            total_tokens_used += tokens_used
            print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
            if not expanded_inspiration: continue

            print("\nGenerating instrument list...")
            instruments, tokens_used = generate_instrument_list_with_ai(genre, expanded_inspiration, num_instruments, config, config_details)
            total_tokens_used += tokens_used
            print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
            if not instruments:
                print(Fore.YELLOW + "AI failed to generate instruments. Using default list from config.yaml.")
                instruments = config.get('instruments', [])
                if not instruments:
                    print(Fore.RED + "No instruments found in config.yaml. Cannot proceed.")
                    continue
            # Validate & normalize instruments
            instruments = validate_instruments(instruments, config_details)

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
            
            print(f"\n{Style.BRIGHT}{Fore.MAGENTA}Total tokens used for setup: {total_tokens_used:,}{Style.RESET_ALL}")
            
            # Save and execute
            if confirm_and_execute(PART_GENERATOR_SCRIPT, config, song_settings):
                 print(Fore.GREEN + "\nSingle Part Generator finished.")
            
            continue # Go back to mode selection

        if action == 'mpe_single_track':
            print_header("MPE SINGLE TRACK (STANDALONE)")
            print("Create one expressive standalone track with optional MPE (per-note pitch bends).")
            confirm = get_user_input("Continue? (y/n):", "y").lower()
            if confirm != 'y':
                continue

            # Load last settings if available
            last_path = os.path.join(script_dir, "last_mpe_single_track.json")
            last_settings = None
            if os.path.exists(last_path):
                try:
                    with open(last_path, 'r') as f:
                        last_settings = json.load(f)
                    print("\nFound previous MPE Single Track settings:")
                    try:
                        print(f"  Name: {last_settings.get('name','')} | Intent: {last_settings.get('role','')} | Program: {last_settings.get('program',80)}")
                        print(f"  Length: {last_settings.get('length_bars',16)} bars | MPE: {bool(last_settings.get('use_mpe',True))} | PBR: {last_settings.get('pbr',48)}")
                        print(f"  Desc: {(last_settings.get('desc','') or '')[:100]}{'...' if last_settings.get('desc') and len(last_settings.get('desc'))>100 else ''}")
                    except Exception:
                        pass
                    reuse = get_user_input("Reuse these settings? (y/n):", "y").lower()
                    if reuse == 'y':
                        name = last_settings.get('name', "Lead Synth")
                        role = last_settings.get('role', "free")
                        program_num = int(last_settings.get('program', 80))
                        length_bars = int(last_settings.get('length_bars', 16))
                        use_mpe = bool(last_settings.get('use_mpe', True))
                        pbr_semi = int(last_settings.get('pbr', 48))
                        desc = last_settings.get('desc', '')
                    else:
                        last_settings = None
                except Exception:
                    last_settings = None

            if not last_settings:
                # Inputs
                genre = get_user_input("Genre (for AI description enrichment, optional):", config.get('genre', ''))
                idea = get_user_input("Short creative idea/goal for this track:")
                name = get_user_input("Instrument name (e.g., 'Lead Synth'):", "Lead Synth")
                role_options = ["free","mpe_lead","mpe_chords","mpe_pads","mpe_arp","lead","chords","pads","arp"]
                print("Choose intent (optional, 'free' allows maximum freedom):")
                for idx, r in enumerate(role_options):
                    print(f"  {idx+1}. {r}")
                role_sel = get_user_input("Role number:", "1")
                try:
                    role = role_options[max(1, min(len(role_options), int(role_sel))) - 1]
                except Exception:
                    role = role_options[0]
                prog = get_user_input("MIDI Program (0-127, default 80):", "80")
                length_str = get_user_input("Length in bars (e.g., 8/16/32, default 16):", "16")
                try:
                    program_num = max(0, min(127, int(prog)))
                except Exception:
                    program_num = 80
                try:
                    length_bars = max(1, int(length_str))
                except Exception:
                    length_bars = 16
                use_mpe = get_user_input("Enable MPE enrichment? (y/n, default y):", "y").lower() == 'y'
                pbr = get_user_input("MPE Pitch Bend Range in semitones (default 48):", "48")
                try:
                    pbr_semi = int(pbr)
                except Exception:
                    pbr_semi = 48

                # Optional AI expansion of description
                desc = idea
                if genre and idea:
                    print("\nExpanding your idea with AI...")
                    expanded, tokens_used = expand_inspiration_with_ai(genre, idea, config)
                    total_tokens_used += tokens_used
                    if expanded:
                        print(Fore.CYAN + f"Tokens used: {tokens_used:,}" + Style.RESET_ALL)
                        print("Preview (first 240 chars):")
                        print((expanded or '')[:240] + ("..." if expanded and len(expanded) > 240 else ""))
                        accept = get_user_input("Use expanded description? (y/n):", "y").lower()
                        if accept == 'y':
                            desc = expanded

            # Launch song_generator in single-track mode
            print("\nStarting standalone generation...\n")
            try:
                cmd = [sys.executable, SONG_GENERATOR_SCRIPT, '--single-track',
                       '--st-name', name,
                       '--st-role', role,
                       '--st-program', str(program_num),
                       '--st-length', str(length_bars),
                       '--st-desc', desc]
                if use_mpe:
                    cmd += ['--st-mpe', '--st-pbr', str(pbr_semi)]
                subprocess.run(cmd, check=True)
                # Save last settings for reuse
                try:
                    with open(last_path, 'w') as f:
                        json.dump({
                            'name': name,
                            'role': role,
                            'program': program_num,
                            'length_bars': length_bars,
                            'use_mpe': use_mpe,
                            'pbr': pbr_semi,
                            'desc': desc
                        }, f, indent=2)
                except Exception:
                    pass
                print(Fore.GREEN + "\nStandalone track process finished." + Style.RESET_ALL)

                # Ask for optimization loop
                do_opt = get_user_input("Optimize this track now using MPE-aware refinement? (y/n):", "y").lower()
                if do_opt == 'y':
                    try:
                        # Load just created track by reusing last settings to reconstruct generation context
                        # For simplicity, re-run a lightweight single-track generation to get track JSON, then iterate optimization
                        import copy
                        tmp_cfg = copy.deepcopy(config)
                        # Use the same parameters
                        from song_generator import generate_single_track_data, generate_mpe_single_track_optimization_data
                        st_track, _tok = generate_single_track_data(tmp_cfg, length_bars, name, program_num, role, desc, use_mpe)
                        if st_track:
                            iterations = int(tmp_cfg.get('optimization_iterations', tmp_cfg.get('number_of_iterations', 1)) or 1)
                            # Use a stable run timestamp and version suffix to avoid overwriting files within the same second
                            run_ts = time.strftime("%Y%m%d-%H%M%S")
                            for it in range(iterations):
                                print(Fore.CYAN + f"Optimization iteration {it+1}/{iterations}..." + Style.RESET_ALL)
                                opt_track, _tok2 = generate_mpe_single_track_optimization_data(tmp_cfg, length_bars, st_track, desc, use_mpe)
                                if opt_track:
                                    st_track = opt_track
                                else:
                                    print(Fore.YELLOW + "Optimization step returned no changes; stopping early." + Style.RESET_ALL)
                                    break
                                # Export result for this iteration (versioned filename _opt_v{it+1})
                                out_base = f"Single_{name.replace(' ','_')}_{length_bars}bars_{int(tmp_cfg.get('bpm',120))}bpm_{run_ts}_opt_v{it+1}.mid"
                                out_path = os.path.join(script_dir, out_base)
                                part_len_beats = length_bars * tmp_cfg["time_signature"]["beats_per_bar"]
                                ok = False
                                try:
                                    from song_generator import create_part_midi_from_theme
                                    ok = create_part_midi_from_theme({"tracks":[st_track]}, tmp_cfg, out_path, time_offset_beats=0, section_length_beats=part_len_beats)
                                except Exception:
                                    pass
                                if ok:
                                    print(Fore.GREEN + f"Optimized standalone track saved: {out_path}" + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"Optimization flow failed: {e}" + Style.RESET_ALL)
            except subprocess.CalledProcessError as e:
                print(Fore.RED + f"\nError during standalone generation: {e}" + Style.RESET_ALL)
            except FileNotFoundError:
                print(Fore.RED + f"\nError: Could not find '{SONG_GENERATOR_SCRIPT}'." + Style.RESET_ALL)
            continue

        if action == 'resume':
            print_header("Resume In-Progress Song")
            for i, pfile in enumerate(progress_files[:10]): # Show top 10 recent
                basename = os.path.basename(pfile)
                time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(pfile)))
                info = ""
                try:
                    with open(pfile, 'r') as f:
                        pdata = json.load(f)
                    # Check for new and old key format for robustness
                    ptype = pdata.get('type') or pdata.get('generation_type', 'unknown')
                    if 'generation' in ptype:
                        info = f"Gen: Theme {pdata.get('current_theme_index', 0) + 1}, Track {pdata.get('current_track_index', 0) + 1}"
                    elif 'optimization' in ptype:
                        info = f"Opt: Theme {pdata.get('current_theme_index',0)+1}, Track {pdata.get('current_track_index',0) + 1}"
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
                # Call song_generator.py directly with the file to resume, bypassing its menu.
                subprocess.run([sys.executable, SONG_GENERATOR_SCRIPT, '--resume-file', selected_file], check=True)
                print(Fore.GREEN + "\nResumed process finished successfully!")
            except subprocess.CalledProcessError as e:
                print(Fore.RED + f"\nError during resumed execution: {e}")
            except FileNotFoundError:
                print(Fore.RED + f"\nError: Could not find the script '{SONG_GENERATOR_SCRIPT}'.")
    
            # After resuming, return to the main menu.
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
            musical_params, tokens_used = get_musical_parameters_with_ai(genre, inspiration, config, config_details)
            total_tokens_used += tokens_used
            if musical_params:
                bpm = musical_params.get('bpm', 120)
                key_scale = musical_params.get('key_scale', 'C minor')
                print(f"{Fore.CYAN}AI Suggestion: {bpm} BPM, {key_scale}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
            else:
                print(Fore.YELLOW + "AI failed to suggest musical parameters. Using defaults from config.yaml.")
                bpm = config.get('bpm', 120)
                key_scale = config.get('key_scale', 'C minor')

            # 2. Expand Inspiration
            print("\nExpanding creative direction...")
            expanded_inspiration, tokens_used = expand_inspiration_with_ai(genre, inspiration, config)
            total_tokens_used += tokens_used
            print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
            if not expanded_inspiration:
                continue # Exit if AI call fails and user cancels

            # 3. Generate Instrument List
            print("\nGenerating instrument list...")
            instruments, tokens_used = generate_instrument_list_with_ai(genre, expanded_inspiration, num_instruments, config, config_details)
            total_tokens_used += tokens_used
            print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
            if not instruments:
                print(Fore.YELLOW + "AI failed to generate instruments. Using default list from config.yaml.")
                instruments = config.get('instruments', []) # Fallback to config
                if not instruments:
                    print(Fore.RED + "No instruments found in config.yaml. Cannot proceed.")
                    continue
            # Validate & normalize instruments
            instruments = validate_instruments(instruments, config_details)

            # 4. Generate Song Structure
            print("\nGenerating song structure...")
            theme_definitions, tokens_used = generate_song_structure_with_ai(genre, expanded_inspiration, instruments, num_parts, part_length, config)
            total_tokens_used += tokens_used
            print(f"{Fore.CYAN}Tokens used for this step: {tokens_used:,}{Style.RESET_ALL}")
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
            
            print(f"\n{Style.BRIGHT}{Fore.MAGENTA}Total tokens used for setup: {total_tokens_used:,}{Style.RESET_ALL}")
            
            # Confirm, save, and execute the main generation script
            if confirm_and_execute(SONG_GENERATOR_SCRIPT, config, song_settings):
                print(f"\n{Fore.GREEN}Generation process started successfully.{Style.RESET_ALL}")
                print(f"{Fore.CYAN}The Song Generator has taken over. Check its window for progress.{Style.RESET_ALL}")
            else:
                # User cancelled in confirm_and_execute, message is already printed
                pass

            print(f"\n{Fore.CYAN}Music Crafter setup complete. Returning to main menu.{Style.RESET_ALL}")
            # We use 'continue' to go back to the main menu
            continue

if __name__ == "__main__":
    main() 