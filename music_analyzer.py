import os
import sys
import json
import time
import math
import glob
from typing import List, Dict, Tuple

import mido
from colorama import Fore, Style, init

# Prefer reusing existing helpers from music_crafter when possible
try:
    import music_crafter as mc
except Exception:
    mc = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- Analyzer-local API key rotation & hotkeys (aligned with song_generator) ---
API_KEYS = []
CURRENT_KEY_INDEX = 0
HOTKEY_MONITOR_STARTED = False
REQUESTED_SWITCH_MODEL = None
REQUEST_SET_SESSION_DEFAULT = False
AUTO_ESCALATE_TO_PRO = False
AUTO_ESCALATE_THRESHOLD = 6
_LAST_HOTKEY_TS = {'1': 0.0, '2': 0.0, '3': 0.0, '0': 0.0, 'a': 0.0, 'r': 0.0}
HOTKEY_DEBOUNCE_SEC = 0.8

KEY_COOLDOWN_UNTIL = {}
PER_MINUTE_COOLDOWN_SECONDS = 60
PER_HOUR_COOLDOWN_SECONDS = 3600
KEY_QUOTA_TYPE = {}
LAST_PER_DAY_SEEN_TS = 0.0
NEXT_HOURLY_PROBE_TS = 0.0

def _classify_quota_error(err_text: str) -> str:
    try:
        t = (err_text or "").lower().replace('-', ' ')
        if any(k in t for k in ["per day", "daily", "per 24 hours", "per 1 day"]):
            return "per-day"
        if any(k in t for k in ["per hour", "per 60 minutes", "hourly", "per 3600 seconds"]):
            return "per-hour"
        if any(k in t for k in ["per minute", "per 60 seconds", "per 60s"]):
            return "per-minute"
        if "rate limit" in t:
            return "rate-limit"
    except Exception:
        pass
    return "unknown"

def _is_key_available(idx: int) -> bool:
    return time.time() >= KEY_COOLDOWN_UNTIL.get(idx, 0)

def _set_key_cooldown(idx: int, seconds: float) -> None:
    KEY_COOLDOWN_UNTIL[idx] = max(KEY_COOLDOWN_UNTIL.get(idx, 0), time.time() + max(1.0, seconds))

def _next_available_key(start_idx: int | None = None) -> int | None:
    if not API_KEYS:
        return None
    n = len(API_KEYS)
    s = CURRENT_KEY_INDEX if start_idx is None else start_idx
    for off in range(1, n+1):
        idx = (s + off) % n
        if _is_key_available(idx):
            return idx
    return None

def _all_keys_cooling_down() -> bool:
    if not API_KEYS:
        return True
    return all(not _is_key_available(i) for i in range(len(API_KEYS)))

def _seconds_until_first_available() -> float:
    if not API_KEYS:
        return 0.0
    return max(0.0, min(KEY_COOLDOWN_UNTIL.get(i, 0) - time.time() for i in range(len(API_KEYS))))

def _all_keys_daily_exhausted() -> bool:
    if not API_KEYS:
        return False
    return all(KEY_QUOTA_TYPE.get(i) == 'per-day' for i in range(len(API_KEYS)))

def _schedule_hourly_probe_if_needed() -> None:
    global NEXT_HOURLY_PROBE_TS
    now = time.time()
    if NEXT_HOURLY_PROBE_TS <= now:
        NEXT_HOURLY_PROBE_TS = now + PER_HOUR_COOLDOWN_SECONDS

def _seconds_until_hourly_probe() -> float:
    now = time.time()
    if NEXT_HOURLY_PROBE_TS <= now:
        return 1.0
    return max(1.0, NEXT_HOURLY_PROBE_TS - now)

def _clear_all_cooldowns() -> None:
    for i in range(len(API_KEYS)):
        KEY_COOLDOWN_UNTIL[i] = 0

def _interruptible_backoff(wait_time: float, context_label: str = "") -> None:
    try:
        end_t = time.time() + max(0.0, wait_time)
        if sys.platform != "win32":
            time.sleep(max(0.0, wait_time)); return
        print(Fore.CYAN + (f"Waiting {wait_time:.1f}s" + (f" [{context_label}]" if context_label else "") + 
              "; press 1/2/3/0 (model), 'a' (auto-escalate), 'r' (reset cooldowns), 's' (skip wait)...") + Style.RESET_ALL)
        import msvcrt
        while time.time() < end_t:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode(errors='ignore').lower()
                now = time.time()
                if ch in _LAST_HOTKEY_TS and now - _LAST_HOTKEY_TS.get(ch, 0.0) < HOTKEY_DEBOUNCE_SEC:
                    continue
                if ch == '1': _LAST_HOTKEY_TS['1'] = now; globals()['REQUESTED_SWITCH_MODEL'] = 'gemini-2.5-pro'; return
                if ch == '2': _LAST_HOTKEY_TS['2'] = now; globals()['REQUESTED_SWITCH_MODEL'] = 'gemini-2.5-flash'; return
                if ch == '3': _LAST_HOTKEY_TS['3'] = now; globals()['REQUESTED_SWITCH_MODEL'] = (globals().get('REQUESTED_SWITCH_MODEL') or 'gemini-2.5-pro'); return
                if ch == '0': _LAST_HOTKEY_TS['0'] = now; globals()['REQUEST_SET_SESSION_DEFAULT'] = True; return
                if ch == 'a': _LAST_HOTKEY_TS['a'] = now; globals()['AUTO_ESCALATE_TO_PRO'] = not globals().get('AUTO_ESCALATE_TO_PRO', False); return
                if ch == 'r': _LAST_HOTKEY_TS['r'] = now; _clear_all_cooldowns(); print(Fore.CYAN + "Cooldowns reset." + Style.RESET_ALL); return
                if ch == 's': return
            time.sleep(0.2)
    except Exception:
        time.sleep(max(0.0, wait_time))

def _print_hotkey_hint(context: str = "") -> None:
    try:
        if sys.platform != "win32":
            return
        ctx = f" [{context}]" if context else ""
        esc = " [ON]" if AUTO_ESCALATE_TO_PRO else " [OFF]"
        print(Style.DIM + Fore.CYAN + (
            f"Hotkeys{ctx}: 1=pro, 2=flash, 3=custom, 0=set session default, a=auto-escalate{esc}, r=reset cooldowns" ) + Style.RESET_ALL)
    except Exception:
        pass
# Reuse generation helpers from song_generator when available
try:
    from song_generator import (
        merge_themes_to_song_data,
        create_midi_from_json,
        generate_optimization_data,
        generate_instrument_track_data,
        build_final_song_basename,
        save_progress,
        get_progress_filename,
        initialize_api_keys as sg_initialize_api_keys,
        save_final_artifact,
    )
except Exception:
    merge_themes_to_song_data = None
    create_midi_from_json = None
    generate_optimization_data = None
    generate_instrument_track_data = None
    build_final_song_basename = None
    save_progress = None
    get_progress_filename = None
    sg_initialize_api_keys = None



# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
SONG_GENERATOR_SCRIPT = os.path.join(script_dir, "song_generator.py")


# --- Console ---
init(autoreset=True)


# --- Local fallbacks if music_crafter is not importable ---
def _print_header(title: str) -> None:
    print("\n" + "=" * 50)
    print(f"--- {title.upper()} ---")
    print("=" * 50 + "\n")


def _get_user_input(prompt: str, default: str | None = None) -> str:
    try:
        response = input(f"{Fore.GREEN}{prompt}{Style.RESET_ALL} ").strip()
        return response or (default or "")
    except (KeyboardInterrupt, EOFError):
        print(Fore.RED + "\nCancelled by user." + Style.RESET_ALL)
        sys.exit(1)


def _load_config_roundtrip() -> Dict:
    if mc and hasattr(mc, "load_config"):
        return mc.load_config()
    # Minimal fallback loader
    import yaml
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _initialize_api_keys(config: Dict) -> Tuple[List[str], int]:
    # Prefer music_crafter's initialization
    if mc and hasattr(mc, "initialize_api_keys") and hasattr(mc, "API_KEYS") and hasattr(mc, "CURRENT_KEY_INDEX"):
        ok = mc.initialize_api_keys(config)
        if not ok:
            return [], 0
        try:
            if genai:
                genai.configure(api_key=mc.API_KEYS[mc.CURRENT_KEY_INDEX])
        except Exception:
            pass
        # Mirror into analyzer and song_generator state
        try:
            keys_m = list(getattr(mc, "API_KEYS", []))
            idx_m = int(getattr(mc, "CURRENT_KEY_INDEX", 0))
            globals()['API_KEYS'] = list(keys_m)
            globals()['CURRENT_KEY_INDEX'] = idx_m
            if sg_initialize_api_keys:
                sg_initialize_api_keys(config)
        except Exception:
            pass
        return list(getattr(mc, "API_KEYS", [])), int(getattr(mc, "CURRENT_KEY_INDEX", 0))

    # Fallback
    api_key_cfg = config.get("api_key")
    keys: List[str] = []
    if isinstance(api_key_cfg, list):
        keys = [k for k in api_key_cfg if isinstance(k, str) and k and "YOUR_" not in k]
    elif isinstance(api_key_cfg, str) and "YOUR_" not in api_key_cfg:
        keys = [api_key_cfg]
    if not keys:
        print(Fore.RED + "Error: No valid API key found in 'config.yaml'." + Style.RESET_ALL)
        return [], 0
    if genai:
        try:
            genai.configure(api_key=keys[0])
        except Exception:
            pass
    # Mirror keys into analyzer and song_generator states for rotation/backoff
    try:
        globals()['API_KEYS'] = list(keys)
        globals()['CURRENT_KEY_INDEX'] = 0
        if sg_initialize_api_keys:
            # Keep song_generator module's state in sync
            sg_initialize_api_keys(config)
    except Exception:
        pass
    return keys, 0


def _call_llm(prompt_text: str, config: Dict, expects_json: bool = False) -> Tuple[str, int]:
    # Use analyzer's robust rotation/backoff so attempts are not incremented on 429

    # Minimal fallback
    if not genai:
        print(Fore.RED + "google.generativeai not available. Install and configure to use AI features." + Style.RESET_ALL)
        return "", 0
    try:
        generation_config = {
            "temperature": config.get("temperature", 1.0)
        }
        if expects_json:
            generation_config["response_mime_type"] = "application/json"
        if isinstance(config.get("max_output_tokens"), int):
            generation_config["max_output_tokens"] = config.get("max_output_tokens")
        # Apply hotkey-based model override
        model_name = config.get("model_name", "gemini-2.5-pro")
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        json_failure_count = 0
        quota_rotation_count = 0
        while True:
            try:
                _print_hotkey_hint("Analyzer LLM call")
                response = model.generate_content(prompt_text, safety_settings=safety_settings)
                out_text = (getattr(response, "text", "") or "")
                _print_llm_debug("LLM call", prompt_text, out_text, expects_json)
                return out_text, int(getattr(getattr(response, "usage_metadata", None), "total_token_count", 0) or 0)
            except Exception as e:
                err = str(e).lower()
                # Quota/429 handling with rotation & cooldowns
                if ('429' in err or 'quota' in err or 'rate limit' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    # Try immediate rotation across all keys before waiting
                    n = len(API_KEYS)
                    rotated = False
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off) % n
                        if not _is_key_available(idx):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            print(Fore.CYAN + f"Switching to API key #{idx+1}..." + Style.RESET_ALL)
                            # rebind model to respect any model switch hotkey
                            model_name = config.get("model_name", "gemini-2.5-pro")
                            if REQUESTED_SWITCH_MODEL:
                                model_name = REQUESTED_SWITCH_MODEL
                            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                            response = model.generate_content(prompt_text, safety_settings=safety_settings)
                            out_text = (getattr(response, "text", "") or "")
                            _print_llm_debug("LLM call (rotated)", prompt_text, out_text, expects_json)
                            return out_text, int(getattr(getattr(response, "usage_metadata", None), "total_token_count", 0) or 0)
                        except Exception as e2:
                            err2 = str(e2).lower()
                            qt2 = _classify_quota_error(err2)
                            KEY_QUOTA_TYPE[idx] = qt2
                            # set cooldown for this key and try next
                            cd = PER_MINUTE_COOLDOWN_SECONDS
                            if qt2 in ('per-hour', 'rate-limit'):
                                cd = PER_HOUR_COOLDOWN_SECONDS
                            elif qt2 == 'per-day':
                                cd = PER_HOUR_COOLDOWN_SECONDS
                            _set_key_cooldown(idx, cd)
                            rotated = True
                            continue
                    # Cooldowns
                    cooldown = PER_MINUTE_COOLDOWN_SECONDS
                    if qtype in ('per-hour', 'rate-limit'):
                        cooldown = PER_HOUR_COOLDOWN_SECONDS
                    elif qtype == 'per-day':
                        cooldown = PER_HOUR_COOLDOWN_SECONDS
                    _set_key_cooldown(CURRENT_KEY_INDEX, cooldown)
                    # Wait strategy
                    if _all_keys_cooling_down():
                        if _all_keys_daily_exhausted():
                            _schedule_hourly_probe_if_needed(); wait_s = _seconds_until_hourly_probe()
                        else:
                            wait_s = max(5.0, min(_seconds_until_first_available(), PER_HOUR_COOLDOWN_SECONDS))
                        _interruptible_backoff(wait_s, context_label="Analyzer 429 cooldown")
                        # after wait, try rotate again
                        nxt = _next_available_key()
                        if nxt is not None:
                            globals()['CURRENT_KEY_INDEX'] = nxt
                            try:
                                genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                            except Exception:
                                pass
                            continue
                        else:
                            # if still none, retry loop with same key
                            continue
                    else:
                        # If not all cooling, immediately retry to switch
                        nxt = _next_available_key()
                        if nxt is not None:
                            globals()['CURRENT_KEY_INDEX'] = nxt
                            try:
                                genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                            except Exception:
                                pass
                            continue
                # Other transient errors → small exponential backoff
                json_failure_count += 1
                base = 3
                wait_time = min(30, base * (2 ** max(0, json_failure_count - 1)))
                time.sleep(wait_time)
                continue
    except Exception as e:
        print(Fore.RED + f"LLM call failed: {e}" + Style.RESET_ALL)
        return "", 0


# --- MIDI Analysis ---
def _midi_key_to_note_scale(key_sig: int, mode: int) -> Tuple[str, str]:
    """Convert MIDI key signature to note name and scale type."""
    # Major keys with sharps/flats
    major_keys = {
        0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
        -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb", -7: "Cb"
    }
    
    # Minor keys (relative minors)
    minor_keys = {
        0: "A", 1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#",
        -1: "D", -2: "G", -3: "C", -4: "F", -5: "Bb", -6: "Eb", -7: "Ab"
    }
    
    if mode == 0:  # Major
        root_note = major_keys.get(key_sig, "C")
        scale_type = "major"
    else:  # Minor
        root_note = minor_keys.get(key_sig, "A")
        scale_type = "minor"
    
    return root_note, scale_type

def _analyze_key_from_pitches(notes: List[Dict]) -> Tuple[str, str]:
    """Analyze key/scale from pitch class histogram of all notes."""
    if not notes:
        return "C", "major"
    
    # Count pitch classes (0-11)
    pitch_class_counts = [0] * 12
    for note in notes:
        pitch = note.get("pitch", 60)
        pc = pitch % 12
        pitch_class_counts[pc] += 1
    
    # Find most common pitch classes
    total_notes = sum(pitch_class_counts)
    if total_notes == 0:
        return "C", "major"
    
    # Normalize to percentages
    pc_percentages = [count / total_notes for count in pitch_class_counts]
    
    # Scale intervals (same as song_generator.py) - intervals from root in semitones
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
    
    # Root note names (all 12 chromatic notes)
    root_pc_map = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 
                   6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    best_score = 0
    best_key = "C"
    best_scale = "major"
    
    # Test all combinations of root notes and scale types
    for scale_type, intervals in scale_intervals.items():
        for root_pc in range(12):
            # Calculate pitch classes for this root+scale combination
            scale_pcs = sorted([(root_pc + interval) % 12 for interval in intervals])
            
            # Score: sum of percentages for pitch classes in this scale
            score = sum(pc_percentages[pc] for pc in scale_pcs)
            
            if score > best_score:
                best_score = score
                best_key = root_pc_map[root_pc]
                best_scale = scale_type
    
    return best_key, best_scale

def _analyze_key_with_llm(config: Dict, track_summaries: List[Dict], genre: str) -> Tuple[str, str]:
    """Use LLM to analyze key/scale from track summaries as fallback."""
    try:
        prompt = (
            "Analyze the musical key and scale from these track summaries.\n"
            f"Genre context: {genre}\n\n"
            "Return ONLY a JSON object: {\"root_note\": \"C\", \"scale_type\": \"major\"}\n"
            "Valid root_notes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B\n"
            "Valid scale_types: major, ionian, minor, natural minor, aeolian, harmonic minor, melodic minor, dorian, phrygian, lydian, mixolydian, locrian, major pentatonic, minor pentatonic, chromatic, whole tone, diminished, augmented, byzantine, hungarian minor, persian, arabic, jewish, ahava raba, blues, major blues\n\n"
            "Track summaries:\n" + json.dumps(track_summaries)
        )
        text, _ = _call_llm(prompt, config, expects_json=True)
        if text:
            cleaned = text.strip().replace("```json", "").replace("```", "")
            result = json.loads(cleaned)
            root_note = result.get("root_note", "C")
            scale_type = result.get("scale_type", "major")
            return root_note, scale_type
    except Exception:
        pass
    return "C", "major"
def analyze_midi_file(file_path: str) -> Tuple[List[Dict], float, int, Dict, str, str]:
    """
    Extracts tracks with absolute-beat note timing, initial BPM, time signature, and key/scale.
    Returns: (tracks, bpm, total_bars, time_signature, root_note, scale_type)
    """
    midi = mido.MidiFile(file_path)
    ticks_per_beat = midi.ticks_per_beat or 480

    bpm = 120.0
    time_signature = {"beats_per_bar": 4, "beat_value": 4}
    root_note = "C"
    scale_type = "major"

    # Use first track meta as global
    for msg in midi.tracks[0]:
        if msg.is_meta and msg.type == "set_tempo":
            bpm = float(mido.tempo2bpm(msg.tempo))
        if msg.is_meta and msg.type == "time_signature":
            time_signature["beats_per_bar"] = msg.numerator
            # MIDI denominator is a power of 2: 0=1, 1=2, 2=4, 3=8, 4=16
            # Standard MIDI format uses exponent: 0=1, 1=2, 2=4, 3=8, 4=16
            # BUT: Some buggy MIDI files store the direct value (4, 8, 16) instead of the exponent (2, 3, 4)
            # Detect and handle both cases
            if msg.denominator <= 4:
                # Likely a correct exponent (0-4 → 1, 2, 4, 8, 16)
                time_signature["beat_value"] = 2 ** msg.denominator
            elif msg.denominator in [1, 2, 4, 8, 16]:
                # Likely already the direct value (non-standard but common)
                # This happens when MIDI files store 16 instead of exponent 4
                time_signature["beat_value"] = msg.denominator
            else:
                # Unknown/invalid - default to 4
                time_signature["beat_value"] = 4
            
            # Special case: Normalize 4/16 to 4/4 (common bug from old song_generator versions)
            # 4/16 is very unusual and likely a mistake - old song_generator had a bug where it
            # stored beat_value=16 incorrectly. Most users don't intentionally use 4/16.
            if time_signature["beats_per_bar"] == 4 and time_signature["beat_value"] == 16:
                time_signature["beat_value"] = 4
        if msg.is_meta and msg.type == "key_signature":
            # MIDI key signature: 0=C, 1=G, 2=D, etc. (sharps) or -1=F, -2=Bb, etc. (flats)
            # mode: 0=major, 1=minor
            key_sig = msg.key
            mode = msg.mode
            root_note, scale_type = _midi_key_to_note_scale(key_sig, mode)

    tracks: List[Dict] = []
    total_ticks = 0
    for i, track in enumerate(midi.tracks):
        current_ticks = 0
        notes_on: Dict[int, Tuple[int, int]] = {}
        note_list: List[Dict] = []
        tname = f"Track {i+1}"
        is_drums = False
        program = 0

        for msg in track:
            current_ticks += msg.time
            if msg.is_meta and msg.type == "track_name":
                tname = msg.name
            if msg.type == "program_change":
                program = msg.program
            if hasattr(msg, "channel") and msg.channel == 9:
                is_drums = True

            if msg.type == "note_on" and msg.velocity > 0:
                notes_on[msg.note] = (current_ticks, msg.velocity)
            elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in notes_on:
                    start_ticks, vel = notes_on.pop(msg.note)
                    dur_ticks = current_ticks - start_ticks
                    note_list.append({
                        "pitch": int(msg.note),
                        "start_beat": float(start_ticks / ticks_per_beat),
                        "duration_beats": float(max(0, dur_ticks) / ticks_per_beat),
                        "velocity": int(vel)
                    })

        if note_list:
            role = "drums" if is_drums else "context"
            tracks.append({
                "instrument_name": tname,
                "program_num": int(program),
                "role": role,
                "notes": note_list
            })
        total_ticks = max(total_ticks, current_ticks)

    if total_ticks <= 0:
        return [], bpm, 0, time_signature, root_note, scale_type

    # If no key signature found in MIDI, analyze from pitch content
    if root_note == "C" and scale_type == "major":
        all_notes = []
        for track in tracks:
            # CRITICAL FIX: Exclude drums/percussion and FX from scale analysis!
            # Drums use MIDI pitches for different drum sounds, not musical pitches
            # FX tracks contain sound effects/samples that don't contribute to harmonic key
            # Note: "texture" and "atmosphere" may contain harmonic content, so we include them
            role = track.get("role", "").lower()
            track_name = track.get("instrument_name", "").lower()
            
            # Check if it's drums by role or channel
            is_drums = (role in ["drums", "percussion", "perc", "drum", "kick_and_snare"] or 
                       track.get("is_drum", False) or
                       track.get("channel") == 9)  # MIDI channel 10 (index 9) is drums
            
            # Check if it's FX by role OR by track name (for early detection before role assignment)
            is_fx = (role == "fx" or 
                    any(keyword in track_name for keyword in ["snare roll", "snare_roll", "accelerating", "fx", "effect", "noise", "sweep"]))
            
            # Also exclude tracks that sound like percussion in the name
            is_named_percussion = any(keyword in track_name for keyword in ["snare", "kick", "hi-hat", "hihat", "cymbal", "perc", "drum"])
            
            if not is_drums and not is_fx and not is_named_percussion:
                all_notes.extend(track.get("notes", []))
        
        analyzed_root, analyzed_scale = _analyze_key_from_pitches(all_notes)
        if analyzed_root != "C" or analyzed_scale != "major":
            root_note = analyzed_root
            scale_type = analyzed_scale

    total_beats = total_ticks / float(ticks_per_beat)
    total_bars = int(math.ceil(total_beats / float(time_signature["beats_per_bar"])))
    return tracks, bpm, total_bars, time_signature, root_note, scale_type


def split_tracks_into_sections(tracks: List[Dict], bars_per_section: int, beats_per_bar: int) -> List[Dict]:
    """Split analyzed absolute-beat tracks into sections (themes) of fixed length.
    Returns a list of themes: [{"label", "description", "tracks": [...] }].
    """
    if bars_per_section <= 0 or beats_per_bar <= 0:
        return []
    section_len_beats = float(bars_per_section * beats_per_bar)
    # Compute end in beats
    max_end = 0.0
    for t in tracks:
        for n in t.get("notes", []):
            try:
                s = float(n.get("start_beat", 0.0))
                d = float(n.get("duration_beats", 0.0))
                max_end = max(max_end, s + max(0.0, d))
            except Exception:
                continue
    if max_end <= 0:
        return []
    section_count = max(1, int(math.ceil(max_end / section_len_beats)))
    themes: List[Dict] = []
    for idx in range(section_count):
        start_b = idx * section_len_beats
        end_b = start_b + section_len_beats
        theme_tracks = []
        for tr in tracks:
            rel_notes = []
            for n in tr.get("notes", []):
                try:
                    s = float(n.get("start_beat", 0.0))
                    d = float(n.get("duration_beats", 0.0))
                    e = s + max(0.0, d)
                    if e <= start_b or s >= end_b:
                        continue
                    ns = max(start_b, s)
                    ne = min(end_b, e)
                    nd = max(0.0, ne - ns)
                    if nd <= 0:
                        continue
                    rel_notes.append({
                        "pitch": int(n.get("pitch", 0)),
                        "start_beat": round(ns - start_b, 6),
                        "duration_beats": round(nd, 6),
                        "velocity": int(n.get("velocity", 0)),
                        **({"automations": n.get("automations")} if isinstance(n.get("automations"), dict) else {}),
                    })
                except Exception:
                    continue
            if rel_notes:
                theme_tracks.append({
                    "instrument_name": tr.get("instrument_name", "Track"),
                    "program_num": int(tr.get("program_num", 0)),
                    "role": tr.get("role", "context"),
                    "notes": rel_notes,
                })
        themes.append({
            "label": f"Part_{idx+1}",
            "description": "Imported from analysis.",
            "tracks": theme_tracks,
        })
    return themes

def summarize_track_features(tracks: List[Dict], beats_per_bar: int) -> List[Dict]:
    """
    Build compact, LLM-friendly summaries for each track to avoid huge context.
    """
    summaries: List[Dict] = []

    def _percentiles(values: List[float], ps: List[float]) -> List[float]:
        if not values:
            return [0.0 for _ in ps]
        xs = sorted(values)
        out = []
        for p in ps:
            if p <= 0:
                out.append(xs[0])
                continue
            if p >= 1:
                out.append(xs[-1])
                continue
            k = (len(xs) - 1) * p
            f = int(math.floor(k))
            c = min(f + 1, len(xs) - 1)
            if f == c:
                out.append(xs[f])
            else:
                out.append(xs[f] + (xs[c] - xs[f]) * (k - f))
        return out

    def _compute_polyphony_metrics(notes: List[Dict]) -> Tuple[float, float, float]:
        # Returns: (avg_voices, overlap_ratio, polyphony_score)
        if not notes:
            return 1.0, 0.0, 0.0
        events = []
        for n in notes:
            s = float(n.get("start_beat", 0.0))
            d = max(0.0, float(n.get("duration_beats", 0.0)))
            e = s + d
            events.append((s, 1))
            events.append((e, -1))
        events.sort()
        voices = 0
        last_t = events[0][0]
        time_weighted_voices = 0.0
        overlapping_time = 0.0
        for t, delta in events:
            dt = max(0.0, t - last_t)
            if dt > 0:
                time_weighted_voices += voices * dt
                if voices > 1:
                    overlapping_time += dt
            voices += delta
            last_t = t
        total_time = (events[-1][0] - events[0][0]) if events[-1][0] > events[0][0] else 0.0
        avg_voices = (time_weighted_voices / total_time) if total_time > 0 else 1.0
        overlap_ratio = (overlapping_time / total_time) if total_time > 0 else 0.0
        # polyphony_score normalized: (avg_voices - 1) / 3 clipped to 0..1 (assume up to 4 voices typical)
        polyphony_score = max(0.0, min(1.0, (avg_voices - 1.0) / 3.0))
        # Also compute note-level overlap ratio (any overlap against previous)
        return avg_voices, overlap_ratio, polyphony_score

    def _compute_ioi_stats(starts: List[float]) -> Tuple[float, float]:
        if len(starts) < 2:
            return 0.0, 0.0
        s = sorted(starts)
        iois = [s[i+1] - s[i] for i in range(len(s) - 1)]
        if not iois:
            return 0.0, 0.0
        mean = sum(iois) / len(iois)
        var = sum((x - mean) ** 2 for x in iois) / len(iois)
        return mean, var

    def _onset_grid_histogram(starts: List[float]) -> Dict[str, int]:
        buckets = {"0": 0, "0.25": 0, "0.5": 0, "0.75": 0, "other": 0}
        for s in starts:
            frac = s - math.floor(s)
            q = round(frac * 4) / 4.0
            if abs(q - 0.0) < 1e-3:
                buckets["0"] += 1
            elif abs(q - 0.25) < 1e-3:
                buckets["0.25"] += 1
            elif abs(q - 0.5) < 1e-3:
                buckets["0.5"] += 1
            elif abs(q - 0.75) < 1e-3:
                buckets["0.75"] += 1
            else:
                buckets["other"] += 1
        return buckets

    for t in tracks:
        notes = t.get("notes", [])
        if not notes:
            continue
        starts = [n["start_beat"] for n in notes]
        durs = [n["duration_beats"] for n in notes]
        pitches = [n["pitch"] for n in notes]
        vels = [n["velocity"] for n in notes]
        if not starts:
            continue
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        mean_vel = sum(vels) / max(1, len(vels))
        mean_dur = sum(durs) / max(1, len(durs))
        total_beats = max(starts) + (durs[pitches.index(max(pitches))] if durs else 0)
        bars_span = max(1.0, total_beats / float(beats_per_bar))
        density_per_bar = len(notes) / bars_span
        on_integer_grid = sum(1 for s in starts if abs(s - round(s)) < 1e-3)
        syncopation_ratio = 1.0 - (on_integer_grid / max(1, len(starts)))

        # Pitch class histogram (12 bins)
        pch = [0] * 12
        for p in pitches:
            pch[p % 12] += 1

        # Advanced metrics
        avg_voices, overlap_ratio, polyphony_score = _compute_polyphony_metrics(notes)
        ioi_mean, ioi_var = _compute_ioi_stats(starts)
        dur_p50, dur_p90 = _percentiles(durs, [0.5, 0.9])
        onset_hist = _onset_grid_histogram(starts)
        register_center = sum(pitches) / max(1, len(pitches))
        drum_hits = {}
        if t.get("role") == "drums":
            for dp in (36, 38, 42, 46):
                drum_hits[str(dp)] = sum(1 for p in pitches if p == dp)

        summaries.append({
            "instrument_name": t.get("instrument_name", "Track"),
            "program_num": t.get("program_num", 0),
            "is_drums": t.get("role") == "drums",
            "pitch_range": [int(min_pitch), int(max_pitch)],
            "note_count": len(notes),
            "density_per_bar": round(density_per_bar, 3),
            "mean_duration_beats": round(mean_dur, 3),
            "duration_p50": round(dur_p50, 3),
            "duration_p90": round(dur_p90, 3),
            "mean_velocity": round(mean_vel, 1),
            "syncopation_ratio": round(syncopation_ratio, 3),
            "avg_voices": round(avg_voices, 3),
            "overlap_ratio": round(overlap_ratio, 3),
            "polyphony_score": round(polyphony_score, 3),
            "ioi_mean": round(ioi_mean, 3),
            "ioi_var": round(ioi_var, 3),
            "onset_grid_histogram": onset_hist,
            "register_center": round(register_center, 2),
            "pitch_class_histogram": pch,
            **({"drum_hits": drum_hits} if drum_hits else {}),
        })
    return summaries


# --- Role assignment via LLM ---
def assign_roles_with_llm(config: Dict, genre: str, user_inspiration: str, track_summaries: List[Dict], allowed_roles: List[str]) -> List[Dict]:
    roles_list = ", ".join([f'"{r}"' for r in allowed_roles]) if allowed_roles else ""

    def _build_prompt(error_hint: str | None = None) -> str:
        hint = f"\nVALIDATION ERROR: {error_hint}\nPlease fix the JSON and follow the rules exactly.\n" if error_hint else ""
        return (
            "You are a meticulous music analyst. Assign a musical role to each track based on compact MIDI-derived features and the genre context.\n\n"
            f"Allowed roles (choose one): [{roles_list}]\n"
            "Rules:\n"
            "- Output ONE JSON array only (no markdown). Same order/length as input tracks.\n"
            "- Each object: {\"instrument_name\": string (copy input), \"role\": string (from list), \"confidence\": number 0..1, \"rationale\": string (<=20 words)}.\n"
            "- Do not invent/rename/reorder/drop tracks.\n"
            "- Heuristics: kick_and_snare if drum ch.9 with dominant 36/38; percussion if 42/46 dominant without 36/38; bass if low register (<= 57) & high density; pads/chords if long durations & polyphony.\n\n"
            f"Global: Genre={genre}; User Notes={user_inspiration or ''}\n\n"
            "Track summaries (JSON):\n" + json.dumps(track_summaries) + "\n\n"
            + hint +
            "Return ONLY the JSON array."
        )

    def _validate(arr: List[Dict]) -> str | None:
        # return None if valid, else error message
        if not isinstance(arr, list):
            return "Result is not a JSON array."
        if len(arr) != len(track_summaries):
            return f"Expected {len(track_summaries)} items, got {len(arr)}."
        for i, (a, ts) in enumerate(zip(arr, track_summaries)):
            if not isinstance(a, dict):
                return f"Item {i+1} is not an object."
            if a.get("instrument_name") != ts.get("instrument_name"):
                return f"Item {i+1} instrument_name mismatch."
            r = a.get("role")
            if r not in allowed_roles:
                return f"Item {i+1} role '{r}' not in allowed set."
            conf = a.get("confidence")
            if not isinstance(conf, (int, float)) or not (0 <= float(conf) <= 1):
                return f"Item {i+1} confidence invalid."
            if not isinstance(a.get("rationale", ""), str):
                return f"Item {i+1} rationale missing or not a string."
        return None

    # First attempt
    text, _ = _call_llm(_build_prompt(), config, expects_json=True)
    def _try_parse(txt: str) -> List[Dict] | None:
        try:
            cleaned = (txt or "").strip().replace("```json", "").replace("```", "")
            arr = json.loads(cleaned)
            return arr if isinstance(arr, list) else None
        except Exception:
            return None

    arr = _try_parse(text)
    err = _validate(arr) if arr is not None else "Could not parse JSON."
    if err is None:
        return arr

    # Retry once with validation hint
    text2, _ = _call_llm(_build_prompt(err), config, expects_json=True)
    arr2 = _try_parse(text2)
    err2 = _validate(arr2) if arr2 is not None else "Could not parse JSON."
    return arr2 if err2 is None else (arr if arr is not None else [])


# --- Inspiration and per-section descriptions via LLM ---
def generate_inspiration_with_llm(config: Dict, genre: str, user_inspiration: str, track_summaries: List[Dict], bpm: float | int = 120, ts: Dict | None = None, bars_per_section: int | None = None, instruments_for_ref: List[Dict] | None = None) -> str:
    ts = ts or {"beats_per_bar": 4, "beat_value": 4}
    instruments_for_ref = instruments_for_ref or []
    prompt = (
        "Write ONE English paragraph of MIDI-focused creative direction.\n\n"
        f"Must include: tempo {round(float(bpm))} BPM, time signature {ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}, section length {bars_per_section or '?'} bars.\n"
        "Describe rhythm, melody, harmony, phrasing, and energy over bars; avoid audio/synthesis FX terms.\n"
        "Reference each instrument by name and intended role; if silent, say 'silent'.\n\n"
        f"Genre: {genre}\n"
        f"User Notes: {user_inspiration or ''}\n"
        "Instruments (name, role):\n" + json.dumps(instruments_for_ref) + "\n\n"
        "Output: plain text paragraph only."
    )
    text, _ = _call_llm(prompt, config)
    return (text or "").strip().replace("```", "")


def generate_section_descriptions_with_llm(config: Dict, genre: str, bars_per_section: int, section_count: int, assigned_tracks: List[Dict]) -> List[Dict]:
    basic_tracks = [
        {
            "instrument_name": t.get("instrument_name"),
            "role": t.get("role", "complementary"),
            "program_num": t.get("program_num", 0),
        }
        for t in assigned_tracks
    ]
    prompt = (
        "Design structured section descriptions for a MIDI generator.\n\n"
        f"Context: Genre={genre}; Sections={section_count} of {bars_per_section} bars (fixed).\n"
        "Instruments (name, role):\n" + json.dumps(basic_tracks) + "\n\n"
        f"Output: JSON array of exactly {section_count} objects. Each: {{\"label\": string, \"description\": string}}.\n"
        "Constraints:\n"
        "- In description, state for EACH instrument what it plays or 'silent'.\n"
        "- Use beats/bars phrasing (e.g., 'on beats 2 and 4', 'every 2 bars').\n"
        "- No synthesis/FX words.\n\n"
        "Return ONLY the JSON array."
    )
    text, _ = _call_llm(prompt, config, expects_json=True)
    try:
        cleaned = (text or "").strip().replace("```json", "").replace("```", "")
        arr = json.loads(cleaned)
        if isinstance(arr, list) and len(arr) == section_count:
            return arr
    except Exception:
        pass
    # Simple fallback labels
    return [{"label": f"Part_{i+1}", "description": f"Section {i+1} ({bars_per_section} bars)."} for i in range(section_count)]


# --- Helpers ---
def select_midi_file() -> str:
    print(Fore.CYAN + "Searching for MIDI files..." + Style.RESET_ALL)
    midi_files = glob.glob("**/*.mid", recursive=True) + glob.glob("**/*.midi", recursive=True)
    if not midi_files:
        print(Fore.RED + "No MIDI files found in current tree." + Style.RESET_ALL)
        sys.exit(1)
    print(Fore.GREEN + "Found these MIDI files:" + Style.RESET_ALL)
    for i, p in enumerate(midi_files):
        print(f"  {i+1}: {p}")
    while True:
        sel = _get_user_input("Enter number to analyze:")
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(midi_files):
                print(Fore.CYAN + f"Selected: {midi_files[idx]}" + Style.RESET_ALL)
                return midi_files[idx]
        except Exception:
            pass
        print(Fore.YELLOW + "Invalid selection. Try again." + Style.RESET_ALL)


def get_allowed_roles_from_config(config: Dict) -> List[str]:
    # Try to extract from comments via music_crafter
    if mc and hasattr(mc, "extract_config_details"):
        try:
            details = mc.extract_config_details(CONFIG_FILE)
            roles = details.get("roles") or []
            if roles:
                return roles
        except Exception:
            pass
    # Fallback canonical roles
    return [
        "drums", "kick_and_snare", "percussion", "sub_bass", "bass", "pads", "atmosphere",
        "texture", "chords", "harmony", "arp", "guitar", "lead", "melody", "vocal", "fx", "complementary"
    ]


def interactive_role_review(assigned: List[Dict], allowed_roles: List[str]) -> List[Dict]:
    print(Fore.CYAN + "\nProposed role assignments:" + Style.RESET_ALL)
    for i, a in enumerate(assigned):
        rn = a.get("instrument_name", f"Track {i+1}")
        print(f"  {i+1}. {rn} -> {Fore.YELLOW}{a.get('role','?')}{Style.RESET_ALL} (conf {a.get('confidence', 0):.2f})")
    resp = _get_user_input("Change any? Enter indices like '2,5' or press Enter to accept:", "").strip()
    if not resp:
        return assigned
    try:
        indices = sorted({int(x.strip()) - 1 for x in resp.split(',') if x.strip()})
    except Exception:
        print(Fore.YELLOW + "Invalid input. Skipping changes." + Style.RESET_ALL)
        return assigned
    role_set = set(allowed_roles)
    for idx in indices:
        if not (0 <= idx < len(assigned)):
            continue
        current = assigned[idx].get("role", "complementary")
        print(f"Track {idx+1} current role: {current}")
        new_role = _get_user_input(f"Enter new role (allowed: {', '.join(allowed_roles)}):", current).strip()
        if new_role and new_role in role_set:
            assigned[idx]["role"] = new_role
        else:
            print(Fore.YELLOW + "Invalid role. Keeping previous." + Style.RESET_ALL)
    return assigned


def save_analysis_artifact(path: str, data: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(Fore.GREEN + f"Analysis saved to: {os.path.basename(path)}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Failed to save analysis: {e}" + Style.RESET_ALL)


def preview_settings_then_confirm(config_update: Dict, theme_definitions: List[Dict]) -> bool:
    try:
        print("\n" + "-" * 60)
        print(Style.BRIGHT + "Settings Preview (no files written yet)" + Style.RESET_ALL)
        print("-" * 60)

        print(f"{Fore.CYAN}Genre:{Style.RESET_ALL} {Style.BRIGHT}{config_update.get('genre','N/A')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}BPM:{Style.RESET_ALL} {Style.BRIGHT}{config_update.get('bpm','N/A')}{Style.RESET_ALL}")
        ts = config_update.get('time_signature', {}) or {}
        print(f"{Fore.CYAN}Time Signature:{Style.RESET_ALL} {Style.BRIGHT}{ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Key/Scale:{Style.RESET_ALL} {Style.BRIGHT}{config_update.get('key_scale','N/A')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Length per Part:{Style.RESET_ALL} {Style.BRIGHT}{config_update.get('part_length', 'N/A')}{Style.RESET_ALL} bars")

        insp = (config_update.get('inspiration') or '').strip()
        insp_prev = (insp[:160] + '...') if len(insp) > 163 else insp
        print(f"{Fore.CYAN}Inspiration (preview):{Style.RESET_ALL} {Style.DIM}{insp_prev}{Style.RESET_ALL}")

        instruments = config_update.get('instruments') or []
        print(f"\n{Fore.CYAN}Instruments ({len(instruments)}):{Style.RESET_ALL}")
        for i, inst in enumerate(instruments):
            name = inst.get('name', f'Instrument {i+1}')
            role = inst.get('role', 'complementary')
            pn = inst.get('program_num', 0)
            print(f"  {i+1}. {Fore.GREEN}{name}{Style.RESET_ALL} (Role: {Fore.YELLOW}{role}{Style.RESET_ALL}, Program: {pn})")

        print(f"\n{Fore.CYAN}Song Structure ({len(theme_definitions)} parts):{Style.RESET_ALL}")
        for i, th in enumerate(theme_definitions):
            label = th.get('label', f'Part_{i+1}')
            desc = (th.get('description') or '').strip()
            desc_prev = (desc[:90] + '...') if len(desc) > 93 else desc
            print(f"  Part {i+1}: {Fore.GREEN}{label}{Style.RESET_ALL} - {Style.DIM}{desc_prev}{Style.RESET_ALL}")

        confirm = _get_user_input("\nProceed to save these settings to config.yaml and song_settings.json? (y/n):", "y").lower()
        return confirm == 'y'
    except Exception as e:
        print(Fore.YELLOW + f"Preview failed (continuing without save): {e}" + Style.RESET_ALL)
        return False

def apply_to_song_generator(config_update: Dict, theme_definitions: List[Dict]) -> None:
    # Save into config.yaml and song_settings.json using music_crafter helpers if available
    if mc and hasattr(mc, "save_config") and hasattr(mc, "save_song_settings"):
        try:
            mc.save_config(config_update)
        except Exception as e:
            print(Fore.YELLOW + f"Warning: save_config failed: {e}" + Style.RESET_ALL)
        try:
            settings = {"length": config_update.get("part_length", 16), "theme_definitions": theme_definitions}
            mc.save_song_settings(settings)
        except Exception as e:
            print(Fore.YELLOW + f"Warning: save_song_settings failed: {e}" + Style.RESET_ALL)
    else:
        # Minimal fallback: write JSON alongside for manual use
        try:
            with open(os.path.join(script_dir, "song_settings.json"), "w", encoding="utf-8") as f:
                json.dump({"length": config_update.get("part_length", 16), "theme_definitions": theme_definitions}, f, indent=2)
        except Exception:
            pass

    # Only saving here; launching is handled by post-save action menu in main
    print(Fore.GREEN + "Settings saved. Choose next action from the menu." + Style.RESET_ALL)


def _ensure_scale_fields(cfg: Dict) -> None:
    """Populate cfg['root_note'] and cfg['scale_type'] from cfg['key_scale'] if missing.
    Expects key_scale like 'F# harmonic minor', 'C minor', 'A dorian', etc.
    """
    try:
        if not isinstance(cfg, dict):
            return
        ks = str(cfg.get("key_scale", "")).strip()
        if not ks:
            return
        if cfg.get("root_note") and cfg.get("scale_type"):
            return
        parts = ks.split()
        if not parts:
            return
        root = parts[0]
        scale_type = " ".join(parts[1:]).strip() if len(parts) > 1 else "major"
        # Normalize note spelling (keep #/b as-is); title-case scale type
        cfg.setdefault("root_note", root)
        cfg.setdefault("scale_type", scale_type.lower())
    except Exception:
        pass

def _remap_scale_for_generator(cfg: Dict) -> None:
    """Map cfg['root_note'] string (e.g., 'F#') to a MIDI center note (e.g., 60+pc) for song_generator.
    Leaves integers untouched. Normalizes scale_type to lower-case.
    """
    try:
        if not isinstance(cfg, dict):
            return
        rn = cfg.get("root_note")
        if isinstance(rn, int):
            # Already numeric
            pass
        else:
            NOTE_TO_PC = {
                "C": 0, "B#": 0,
                "C#": 1, "DB": 1, "Db": 1,
                "D": 2,
                "D#": 3, "EB": 3, "Eb": 3,
                "E": 4, "FB": 4, "Fb": 4,
                "F": 5, "E#": 5,
                "F#": 6, "GB": 6, "Gb": 6,
                "G": 7,
                "G#": 8, "AB": 8, "Ab": 8,
                "A": 9,
                "A#": 10, "BB": 10, "Bb": 10,
                "B": 11, "CB": 11, "Cb": 11,
            }
            name = str(rn or "C").strip()
            pc = NOTE_TO_PC.get(name, NOTE_TO_PC.get(name.upper(), 0))
            cfg["root_note"] = 60 + (pc % 12)
        st = cfg.get("scale_type")
        if isinstance(st, str):
            cfg["scale_type"] = st.strip().lower()
    except Exception:
        pass

def _print_llm_debug(label: str, prompt_text: str, output_text: str, expects_json: bool) -> None:
    try:
        print(Style.DIM + f"\n[LLM DEBUG] {label}: expects_json={expects_json}" + Style.RESET_ALL)
        pprev = prompt_text.strip().replace("\n\n", "\n")
        if len(pprev) > 600:
            pprev = pprev[:600] + "..."
        print(Style.DIM + f"Prompt preview:\n{pprev}\n" + Style.RESET_ALL)
        oprev = (output_text or "").strip()
        if len(oprev) > 600:
            oprev = oprev[:600] + "..."
        print(Style.DIM + f"Output preview:\n{oprev}\n" + Style.RESET_ALL)
    except Exception:
        pass

def _choose_tracks_subset(tracks: List[Dict]) -> List[str]:
    """Ask user to select a subset of tracks and return list of track names."""
    try:
        print(Fore.CYAN + "\nTracks:" + Style.RESET_ALL)
        for i, t in enumerate(tracks):
            print(f"  {i+1}. {t.get('instrument_name','Track')} (role: {t.get('role','context')})")
        resp = _get_user_input("Select tracks to process (e.g., '1,3,5') or Enter for all:", "").strip()
        if not resp:
            return [t.get('instrument_name', f'Track_{i+1}') for i, t in enumerate(tracks)]
        ids = sorted({int(x.strip())-1 for x in resp.split(',') if x.strip().isdigit()})
        selected_tracks = []
        for i in ids:
            if 0 <= i < len(tracks):
                track_name = tracks[i].get('instrument_name', f'Track_{i+1}')
                selected_tracks.append(track_name)
        return selected_tracks
    except Exception:
        return [t.get('instrument_name', f'Track_{i+1}') for i, t in enumerate(tracks)]

def _get_all_unique_tracks_from_themes(themes: List[Dict]) -> List[Dict]:
    """Extract all unique tracks from all themes, preserving order and metadata."""
    unique_tracks = []
    seen_names = set()
    
    for theme in themes:
        for track in theme.get('tracks', []):
            track_name = track.get('instrument_name', '')
            if track_name and track_name not in seen_names:
                seen_names.add(track_name)
                unique_tracks.append(track)
    
    return unique_tracks


def _optimize_selected_tracks(config: Dict, themes: List[Dict], bars_per_section: int, selected_track_names: List[str]) -> List[Dict]:
    """Optimize only selected tracks across all themes using generator's optimization step."""
    if not generate_optimization_data:
        print(Fore.YELLOW + "Optimization helper not available. Skipping." + Style.RESET_ALL)
        return themes
    updated = []
    for theme_idx, th in enumerate(themes):
        new_tracks = []
        inner_ctx = [t for t in th.get('tracks', [])]
        for i, tr in enumerate(th.get('tracks', [])):
            track_name = tr.get('instrument_name', '')
            if track_name in selected_track_names:
                role = tr.get('role', 'complementary')
                label = th.get('label', f'Part_{theme_idx+1}')
                desc = th.get('description', '')
                try:
                    opt_tr, _ = generate_optimization_data(
                        config, bars_per_section, tr, role, label, desc, [], [t for j,t in enumerate(inner_ctx) if j!=i], theme_idx, user_optimization_prompt=""
                    )
                    # Keep instrument metadata so the export does not collapse to "Unknown Instrument"
                    if opt_tr and isinstance(opt_tr, dict) and opt_tr.get('notes'):
                        opt_tr['instrument_name'] = tr.get('instrument_name', opt_tr.get('instrument_name'))
                        opt_tr['program_num'] = tr.get('program_num', opt_tr.get('program_num', 0))
                        opt_tr['role'] = tr.get('role', opt_tr.get('role', 'complementary'))
                        new_tracks.append(opt_tr)
                    else:
                        new_tracks.append(tr)
                except Exception:
                    new_tracks.append(tr)
            else:
                new_tracks.append(tr)
        updated.append({**th, 'tracks': new_tracks})
    return updated


def _add_new_track_across_parts(config: Dict, themes: List[Dict], bars_per_section: int) -> List[Dict]:
    """Create a new track for each theme in context to the others. Allows user description or minimal prompt."""
    if not generate_instrument_track_data:
        print(Fore.YELLOW + "Track generation helper not available. Skipping." + Style.RESET_ALL)
        return themes
    # Explain modes briefly
    print(Fore.CYAN + "\nTrack creation modes:" + Style.RESET_ALL)
    print("  1) Manual: you enter a detailed description (full control).")
    print("  2) Minimal guided: short auto-prompt; fast and simple.")
    print("  3) Guided full spec: you choose role + minimal idea; AI expands and completes name/program/description.")
    print("  4) Auto full spec: AI proposes name/program/description; you choose the role.")
    # Mode selection
    mode = _get_user_input(
        "Track creation mode (1/2/3/4):",
        "2"
    ).strip()

    def _auto_propose_track_spec(target_role: str | None, user_min_desc: str | None = None) -> Tuple[str, int, str, str]:
        base_instrs = []
        try:
            base_instrs = [
                {
                    "instrument_name": t.get("instrument_name"),
                    "role": t.get("role", "complementary"),
                    "program_num": int(t.get("program_num", 0)),
                }
                for t in (themes[0].get("tracks", []) if themes else [])
            ]
        except Exception:
            base_instrs = []
        # Build rich global context
        genre = str(config.get("genre", ""))
        key_scale = str(config.get("key_scale", ""))
        ts = config.get("time_signature", {}) or {}
        beats_per_bar = ts.get("beats_per_bar")
        beat_value = ts.get("beat_value")
        insp_all = str(config.get("inspiration", "")).strip()
        inspiration_short = (insp_all[:320] + "...") if len(insp_all) > 340 else insp_all
        sections_ctx = []
        try:
            for th in themes:
                sections_ctx.append({
                    "label": th.get("label"),
                    "description": th.get("description")
                })
        except Exception:
            sections_ctx = []

        role_line = (f"Target role: {target_role}. The output role MUST be exactly this value.\n" if target_role else "")
        min_line = (f"Minimal idea to elaborate: {user_min_desc}\n" if user_min_desc else "")
        prompt = (
            "You are adding ONE new track to this MIDI song. Return ONLY one JSON object in this schema:\n"
            "{\"name\": str, \"program_num\": int 0..127, \"role\": str, \"description\": str}\n\n"
            "Constraints:\n"
            "- role MUST be exactly the target role (if provided).\n"
            "- description MUST be genre-typical and context-aware for the role, over exactly " + str(bars_per_section) + " bars.\n"
            "- Use bar/beat phrasing (e.g., 'bars 1-4: …, bars 5-8: …').\n"
            "- Do NOT invent unavailable harmonic fields (no 'root_note' keys, etc.). Describe performance behavior instead.\n\n"
            "Global context:\n"
            f"- Genre: {genre}\n"
            f"- Key/Scale: {key_scale}\n"
            f"- Time Signature: {beats_per_bar}/{beat_value}\n"
            f"- Inspiration (short): {inspiration_short}\n"
            f"- Existing instruments (name, role, program): {json.dumps(base_instrs)}\n"
            f"- Sections (label+description, in order): {json.dumps(sections_ctx)}\n\n"
            + role_line + min_line +
            "Return ONLY the JSON object with the schema above."
        )
        text, _ = _call_llm(prompt, config, expects_json=True)
        try:
            cleaned = (text or "").strip().replace("```json", "").replace("```", "")
            obj = json.loads(cleaned)
            proposed_role = str(obj.get("role", "complementary"))
            if target_role and proposed_role != target_role:
                proposed_role = target_role
            return (
                str(obj.get("name", "New Track")),
                int(obj.get("program_num", 0)),
                proposed_role,
                str(obj.get("description", "Add a complementary line that fits the style.")),
            )
        except Exception:
            return ("New Track", 0, "complementary", f"Add a complementary line over {bars_per_section} bars.")

    if mode == "4":
        # Let user choose role from allowed list
        roles = get_allowed_roles_from_config(config)
        try:
            print(Fore.CYAN + "\nAvailable roles:" + Style.RESET_ALL)
            print("  " + ", ".join(roles))
        except Exception:
            pass
        target_role = _get_user_input("Choose role for the new track:", (roles[0] if roles else "complementary")).strip() or (roles[0] if roles else "complementary")
        name, program, role, custom_desc = _auto_propose_track_spec(target_role)
        try:
            print(Fore.CYAN + "\nUsing new track spec (Auto full spec):" + Style.RESET_ALL)
            print(f"  Name: {name}\n  Program: {program}\n  Role: {role}")
            desc_prev = (custom_desc[:400] + '...') if isinstance(custom_desc, str) and len(custom_desc) > 403 else (custom_desc or "")
            print(f"  Description: {desc_prev}")
        except Exception:
            pass
    elif mode == "3":
        # Guided full spec: user provides role + minimal description; AI completes the rest
        roles = get_allowed_roles_from_config(config)
        try:
            print(Fore.CYAN + "\nAvailable roles:" + Style.RESET_ALL)
            print("  " + ", ".join(roles))
        except Exception:
            pass
        target_role = _get_user_input("Choose role for the new track:", (roles[0] if roles else "complementary")).strip() or (roles[0] if roles else "complementary")
        user_min_desc = _get_user_input("Enter a minimal idea (1-2 sentences):", "A supportive line that enhances the groove without clutter.").strip()
        name, program, role, custom_desc = _auto_propose_track_spec(target_role, user_min_desc)
        try:
            print(Fore.CYAN + "\nUsing new track spec (Guided full spec):" + Style.RESET_ALL)
            print(f"  Name: {name}\n  Program: {program}\n  Role: {role}")
            print(f"  Your idea: {user_min_desc}")
            desc_prev = (custom_desc[:400] + '...') if isinstance(custom_desc, str) and len(custom_desc) > 403 else (custom_desc or "")
            print(f"  Expanded description: {desc_prev}")
        except Exception:
            pass
    else:
        # For manual/minimal, also show allowed roles for clarity
        roles = get_allowed_roles_from_config(config)
        try:
            print(Fore.CYAN + "\nAvailable roles:" + Style.RESET_ALL)
            print("  " + ", ".join(roles))
        except Exception:
            pass
        name = _get_user_input("New track name (e.g., 'New Lead'):", "New Track").strip() or "New Track"
        try:
            program = int(_get_user_input("MIDI program number (0-127):", "81").strip())
        except Exception:
            program = 0
        role = _get_user_input("Role:", (roles[0] if roles else "complementary")).strip() or (roles[0] if roles else "complementary")
        if mode == "1":
            custom_desc = _get_user_input("Enter your detailed description (one paragraph):", "").strip()
        elif mode == "2":
            custom_desc = f"Add a {role} line that complements existing parts. Use clear phrasing over {bars_per_section} bars and leave space where needed."
        else:
            # Fallback to minimal guided if an unsupported mode was entered
            custom_desc = f"Add a {role} line that complements existing parts. Use clear phrasing over {bars_per_section} bars and leave space where needed."
        try:
            mode_label = "Manual" if mode == "1" else "Minimal guided"
            print(Fore.CYAN + f"\nUsing new track spec ({mode_label}):" + Style.RESET_ALL)
            print(f"  Name: {name}\n  Program: {program}\n  Role: {role}")
            desc_prev = (custom_desc[:400] + '...') if isinstance(custom_desc, str) and len(custom_desc) > 403 else (custom_desc or "")
            print(f"  Description: {desc_prev}")
        except Exception:
            pass

    updated = []
    appended_count = 0
    for theme_idx, th in enumerate(themes):
        ctx_tracks = [t for t in th.get('tracks', [])]
        label = th.get('label', f'Part_{theme_idx+1}')
        desc = th.get('description', '')
        try:
            # Ensure we pass the same argument structure as song_generator expects,
            # including full previous themes history for context linking.
            track_theme_desc = (custom_desc or desc)
            if custom_desc:
                # Merge part description + instrument directive for stronger guidance
                track_theme_desc = f"{desc}\n\nFor the {role} track: {custom_desc}"
            tr_data, _ = generate_instrument_track_data(
                config,
                length=bars_per_section,
                instrument_name=name,
                program_num=program,
                context_tracks=ctx_tracks,
                role=role,
                current_track_index=len(ctx_tracks),
                total_tracks=len(ctx_tracks) + 1,
                dialogue_role='none',
                theme_label=label,
                theme_description=track_theme_desc,
                previous_themes_full_history=themes,
                current_theme_index=theme_idx,
            )
            new_tracks = list(ctx_tracks)
            if tr_data and isinstance(tr_data, dict) and tr_data.get('notes'):
                tr_data['instrument_name'] = name
                tr_data['program_num'] = program
                tr_data['role'] = role
                new_tracks.append(tr_data)
                updated.append({**th, 'tracks': new_tracks})
                appended_count += 1
            else:
                print(Fore.YELLOW + f"No notes generated for '{name}' in '{label}'. Skipping append." + Style.RESET_ALL)
                updated.append(th)
        except Exception as e:
            print(Fore.YELLOW + f"Track generation failed for '{name}' in '{label}': {e}" + Style.RESET_ALL)
            updated.append(th)
    if appended_count == 0:
        print(Fore.YELLOW + f"No parts received a new '{role}' track. Consider using Guided full spec (3) with a short idea, or try again." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"Added '{name}' ({role}) to {appended_count} part(s)." + Style.RESET_ALL)
        # Ensure the generator knows about the new instrument (channel assignment/export)
        try:
            insts = config.get('instruments')
            if not isinstance(insts, list):
                insts = []
            # check by name uniqueness
            exists = any(isinstance(i, dict) and str(i.get('name')).strip().lower() == str(name).strip().lower() for i in insts)
            if not exists:
                insts.append({"name": name, "program_num": int(program), "role": role})
                config['instruments'] = insts
                print(Fore.CYAN + f"Config instruments updated: appended '{name}' (program {program}, role {role})." + Style.RESET_ALL)
        except Exception:
            pass
    return updated


def offer_integrated_actions(config_update: Dict, themes_from_analysis: List[Dict], bars_per_section: int) -> None:
    print("\n" + "-" * 60)
    print(Style.BRIGHT + "Integrated actions (after analysis)" + Style.RESET_ALL)
    print("-" * 60)
    print("  1) Optimize selected tracks only (keeps other tracks unchanged)")
    print("  2) Add a new track to the whole song (context-aware)")
    print("  3) Generate a NEW MIDI from the analyzed descriptions")
    print("  4) Full optimization of the ORIGINAL imported MIDI")
    print("  5) Finish")
    choice = _get_user_input("Choose 1/2/3/4/5:", "5").strip()

    # Build an effective runtime config by merging current config.yaml with the proposed updates
    try:
        base_cfg = _load_config_roundtrip() or {}
    except Exception:
        base_cfg = {}
    effective_config = dict(base_cfg)
    try:
        if isinstance(config_update, dict):
            effective_config.update(config_update)
    except Exception:
        pass
    # Ensure API keys/model configured for downstream generator helpers
    try:
        _ensure_scale_fields(effective_config)
        _remap_scale_for_generator(effective_config)
        _initialize_api_keys(effective_config)
    except Exception:
        pass

    themes = list(themes_from_analysis)
    if choice == "1":
        if not themes:
            print(Fore.YELLOW + "No themes available from analysis; skipping." + Style.RESET_ALL)
            return
        print(Fore.CYAN + "Select which tracks you want to optimize. Others will be left as-is." + Style.RESET_ALL)
        all_tracks = _get_all_unique_tracks_from_themes(themes)
        subset = _choose_tracks_subset(all_tracks)
        themes = _optimize_selected_tracks(effective_config, themes, bars_per_section, subset)
        print(Fore.GREEN + "Done: optimized selected tracks across all parts." + Style.RESET_ALL)
        # Save resume progress for song_generator
        try:
            if save_progress and get_progress_filename:
                progress_payload = {
                    'type': 'optimization',
                    'config': effective_config,
                    'theme_length': bars_per_section,
                    'themes': themes,
                    'user_optimization_prompt': '',
                    'timestamp': time.strftime("%Y%m%d-%H%M%S"),
                }
                save_progress(progress_payload, script_dir, progress_payload['timestamp'])
                print(Fore.CYAN + "Resume progress file saved for song_generator." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"Could not save resume progress: {e}" + Style.RESET_ALL)
        # Offer immediate export
        try:
            do_exp = _get_user_input("Export a new MIDI with these updates now? (y/n):", "y").strip().lower()
            if do_exp == 'y' and merge_themes_to_song_data and create_midi_from_json and build_final_song_basename:
                song_data = merge_themes_to_song_data(themes, effective_config, bars_per_section)
                base = build_final_song_basename(effective_config, themes, time.strftime("%Y%m%d-%H%M%S"), resumed=True)
                out_path = os.path.join(script_dir, f"{base}_updated.mid")
                ok = create_midi_from_json(song_data, effective_config, out_path)
                print((Fore.GREEN + f"Exported: {out_path}") if ok else (Fore.RED + "MIDI export failed."))
        except Exception as e:
            print(Fore.YELLOW + f"Export failed: {e}" + Style.RESET_ALL)
        # Also persist a new final artifact to include these updated themes
        try:
            if save_final_artifact and isinstance(themes, list) and themes:
                tdefs = []
                try:
                    for th in themes:
                        tdefs.append({"label": th.get("label"), "description": th.get("description", "")})
                except Exception:
                    tdefs = []
                ts_now = time.strftime("%Y%m%d-%H%M%S")
                save_final_artifact(effective_config, themes, bars_per_section, tdefs, script_dir, ts_now)
                print(Fore.CYAN + "Saved updated final artifact (includes optimized tracks)." + Style.RESET_ALL)
        except Exception:
            pass
    elif choice == "2":
        if not themes:
            print(Fore.YELLOW + "No themes available from analysis; skipping." + Style.RESET_ALL)
            return
        themes = _add_new_track_across_parts(effective_config, themes, bars_per_section)
        print(Fore.GREEN + "Done: added the new track to every part." + Style.RESET_ALL)
        # Save resume progress for song_generator
        try:
            if save_progress and get_progress_filename:
                progress_payload = {
                    'type': 'generation',
                    'config': effective_config,
                    'theme_length': bars_per_section,
                    'themes': themes,
                    'timestamp': time.strftime("%Y%m%d-%H%M%S"),
                }
                save_progress(progress_payload, script_dir, progress_payload['timestamp'])
                print(Fore.CYAN + "Resume progress file saved for song_generator." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"Could not save resume progress: {e}" + Style.RESET_ALL)
        # Offer immediate export
        try:
            do_exp = _get_user_input("Export a new MIDI with the new track now? (y/n):", "y").strip().lower()
            if do_exp == 'y' and merge_themes_to_song_data and create_midi_from_json and build_final_song_basename:
                song_data = merge_themes_to_song_data(themes, effective_config, bars_per_section)
                base = build_final_song_basename(effective_config, themes, time.strftime("%Y%m%d-%H%M%S"), resumed=True)
                out_path = os.path.join(script_dir, f"{base}_updated.mid")
                ok = create_midi_from_json(song_data, effective_config, out_path)
                print((Fore.GREEN + f"Exported: {out_path}") if ok else (Fore.RED + "MIDI export failed."))
        except Exception as e:
            print(Fore.YELLOW + f"Export failed: {e}" + Style.RESET_ALL)
        # Also persist a new final artifact so lyrics/menu see the added track in [F]
        try:
            if save_final_artifact and isinstance(themes, list) and themes:
                tdefs = []
                try:
                    for th in themes:
                        tdefs.append({"label": th.get("label"), "description": th.get("description", "")})
                except Exception:
                    tdefs = []
                ts_now = time.strftime("%Y%m%d-%H%M%S")
                save_final_artifact(effective_config, themes, bars_per_section, tdefs, script_dir, ts_now)
                print(Fore.CYAN + "Saved updated final artifact (includes newly added track)." + Style.RESET_ALL)
        except Exception:
            pass
    elif choice == "3":
        # Use current config_update + derived themes to build a new song via generator
        if not (merge_themes_to_song_data and create_midi_from_json and build_final_song_basename):
            print(Fore.YELLOW + "Generation helpers not available." + Style.RESET_ALL)
            return
        try:
            song_data = merge_themes_to_song_data(themes, effective_config, bars_per_section)
            base = build_final_song_basename(effective_config, themes, time.strftime("%Y%m%d-%H%M%S"), resumed=False)
            out_path = os.path.join(script_dir, f"{base}_generated_from_analysis.mid")
            ok = create_midi_from_json(song_data, effective_config, out_path)
            print((Fore.GREEN + f"Generated: {out_path}") if ok else (Fore.RED + "Generation/export failed."))
        except Exception as e:
            print(Fore.RED + f"Generation error: {e}" + Style.RESET_ALL)
        # Save a final artifact snapshot matching the generated-from-analysis state
        try:
            if save_final_artifact and isinstance(themes, list) and themes:
                tdefs = []
                try:
                    for th in themes:
                        tdefs.append({"label": th.get("label"), "description": th.get("description", "")})
                except Exception:
                    tdefs = []
                ts_now = time.strftime("%Y%m%d-%H%M%S")
                save_final_artifact(effective_config, themes, bars_per_section, tdefs, script_dir, ts_now)
                print(Fore.CYAN + "Saved final artifact from analysis-based generation." + Style.RESET_ALL)
        except Exception:
            pass
    elif choice == "4":
        # Full-file optimization path: treat imported themes as baseline and run per-track optimization for all
        if not themes:
            print(Fore.YELLOW + "No themes available from analysis; skipping." + Style.RESET_ALL)
            return
        all_tracks = _get_all_unique_tracks_from_themes(themes)
        subset_all = [t.get('instrument_name', f'Track_{i+1}') for i, t in enumerate(all_tracks)]
        if not subset_all:
            print(Fore.YELLOW + "No tracks found to optimize." + Style.RESET_ALL)
            return
        print(Fore.CYAN + "Running full optimization across all parts and tracks..." + Style.RESET_ALL)
        optimized = _optimize_selected_tracks(effective_config, themes, bars_per_section, subset_all)
        # Offer immediate export
        try:
            song_data = merge_themes_to_song_data(optimized, effective_config, bars_per_section)
            base = build_final_song_basename(effective_config, themes, time.strftime("%Y%m%d-%H%M%S"), resumed=True)
            out_path = os.path.join(script_dir, f"{base}_fully_optimized.mid")
            ok = create_midi_from_json(song_data, effective_config, out_path)
            print((Fore.GREEN + f"Exported: {out_path}") if ok else (Fore.RED + "MIDI export failed."))
            # Save resume progress so song_generator can pick it up
            try:
                if save_progress and get_progress_filename:
                    progress_payload = {
                        'type': 'optimization',
                        'config': effective_config,
                        'theme_length': bars_per_section,
                        'themes': optimized,
                        'user_optimization_prompt': '',
                        'timestamp': time.strftime("%Y%m%d-%H%M%S"),
                    }
                    save_progress(progress_payload, script_dir, progress_payload['timestamp'])
                    print(Fore.CYAN + "Resume progress file saved for song_generator." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.YELLOW + f"Could not save resume progress: {e}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"Export after optimization failed: {e}" + Style.RESET_ALL)
        # Persist a final artifact reflecting the fully optimized themes
        try:
            if save_final_artifact and isinstance(optimized, list) and optimized:
                tdefs = []
                try:
                    for th in optimized:
                        tdefs.append({"label": th.get("label"), "description": th.get("description", "")})
                except Exception:
                    tdefs = []
                ts_now = time.strftime("%Y%m%d-%H%M%S")
                save_final_artifact(effective_config, optimized, bars_per_section, tdefs, script_dir, ts_now)
                print(Fore.CYAN + "Saved updated final artifact (fully optimized)." + Style.RESET_ALL)
        except Exception:
            pass
    else:
        print(Fore.YELLOW + "No action selected." + Style.RESET_ALL)

    # Offer loop for multiple actions
    if choice in {"1", "2"}:
        try:
            again = _get_user_input("Perform another integrated action? (y/n):", "n").strip().lower()
            if again == 'y':
                offer_integrated_actions(config_update, themes, bars_per_section)
        except Exception:
            pass

    print("\n" + "-" * 60)
    print(Style.BRIGHT + "What would you like to do next?" + Style.RESET_ALL)
    print("-" * 60)
    print("  1) Generate a new MIDI now (auto run)")
    print("  2) Open Song Generator menu (choose Optimization cycle there)")
    print("  3) Do nothing")
    choice = _get_user_input("Select 1/2/3:", "1").strip()
    try:
        import subprocess
        if choice == "1":
            print(Fore.CYAN + "Launching song_generator (auto run)..." + Style.RESET_ALL)
            subprocess.run([sys.executable, SONG_GENERATOR_SCRIPT, "--run"], check=False)
        elif choice == "2":
            print(Fore.CYAN + "Opening Song Generator main menu..." + Style.RESET_ALL)
            subprocess.run([sys.executable, SONG_GENERATOR_SCRIPT], check=False)
        else:
            print(Fore.YELLOW + "No action selected." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + f"Could not launch song_generator: {e}" + Style.RESET_ALL)


# --- Main ---
def main():
    _print_header("Music Analyzer")

    # Load config and LLM setup
    config = _load_config_roundtrip()
    allowed_roles = get_allowed_roles_from_config(config)
    api_keys, _ = _initialize_api_keys(config)
    if not api_keys:
        return

    # File selection
    midi_path = select_midi_file()

    # Ask for genre and mode
    print(Fore.CYAN + "\nGenre options:" + Style.RESET_ALL)
    print("- Enter a genre manually (recommended).")
    print("- Or type 'ai' to let the model infer genre from features (compact).")
    raw_genre = _get_user_input("Enter genre (or 'ai'):", "")
    user_inspiration = _get_user_input("Optional: Additional notes about the MIDI (press Enter to skip):", "")

    # Fixed segmentation length
    seg_str = _get_user_input("Analyze in fixed sections of 8, 16, or 32 bars? (8/16/32):", "16").strip()
    if seg_str not in {"8", "16", "32"}:
        seg_str = "16"
    bars_per_section = int(seg_str)

    # MIDI analysis
    tracks, bpm, total_bars, ts, root_note, scale_type = analyze_midi_file(midi_path)
    if not tracks or total_bars == 0:
        print(Fore.RED + "No note data found. Aborting." + Style.RESET_ALL)
        return
    print(Fore.CYAN + f"Analyzed MIDI: {bpm:.2f} BPM, {ts['beats_per_bar']}/{ts['beat_value']}, ~{total_bars} bars, Key: {root_note} {scale_type}" + Style.RESET_ALL)

    # Summaries for LLM
    summaries = summarize_track_features(tracks, ts["beats_per_bar"])

    # Optional AI genre guess from summaries
    genre = raw_genre
    if raw_genre.lower() == "ai":
        g_prompt = (
            "From the following compact track summaries, return ONE concise genre label (<=4 words).\n"
            "No punctuation or prose, one line only.\n\n" + json.dumps(summaries)
        )
        g_text, _ = _call_llm(g_prompt, config)
        genre = (g_text or "").strip().splitlines()[0]
        # Trim to <=4 words
        words = [w for w in genre.split() if w]
        genre = " ".join(words[:4]) if words else "Electronic"

    # If key analysis from MIDI failed, try LLM-based analysis
    if root_note == "C" and scale_type == "major":
        print(Fore.CYAN + "No key signature found in MIDI, trying LLM-based key analysis..." + Style.RESET_ALL)
        llm_root, llm_scale = _analyze_key_with_llm(config, summaries, genre)
        if llm_root != "C" or llm_scale != "major":
            root_note = llm_root
            scale_type = llm_scale
            print(Fore.GREEN + f"LLM detected key: {root_note} {scale_type}" + Style.RESET_ALL)

    # Role assignment
    assigned = assign_roles_with_llm(config, genre, user_inspiration, summaries, allowed_roles) or []
    # Merge assigned roles back to track list (by instrument_name)
    name_to_role = {a.get("instrument_name"): a.get("role", "complementary") for a in assigned}
    merged_tracks = []
    for t in tracks:
        role = name_to_role.get(t.get("instrument_name"), t.get("role", "context"))
        mt = dict(t)
        mt["role"] = role
        merged_tracks.append(mt)

    # Review & override
    assigned = [
        {
            "instrument_name": t.get("instrument_name"),
            "role": t.get("role", "complementary"),
            "confidence": next((a.get("confidence", 0.7) for a in assigned if a.get("instrument_name") == t.get("instrument_name")), 0.7)
        }
        for t in merged_tracks
    ]
    assigned = interactive_role_review(assigned, allowed_roles)

    # Re-analyze key with correct role assignments (FX tracks may have been included incorrectly)
    # Update merged_tracks with reviewed roles
    name_to_role_reviewed = {a.get("instrument_name"): a.get("role", "complementary") for a in assigned}
    for t in merged_tracks:
        reviewed_role = name_to_role_reviewed.get(t.get("instrument_name"), t.get("role", "context"))
        t["role"] = reviewed_role
    
    # Re-analyze key from pitch content with correct roles
    # This is important because initial analysis happens before roles are assigned
    all_notes_corrected = []
    excluded_tracks = []
    for track in merged_tracks:
        role = track.get("role", "").lower()
        track_name = track.get("instrument_name", "")
        is_drums = (role in ["drums", "percussion", "perc", "drum", "kick_and_snare"] or 
                   track.get("is_drum", False) or
                   track.get("channel") == 9)
        is_fx = (role == "fx")
        
        if not is_drums and not is_fx:
            all_notes_corrected.extend(track.get("notes", []))
        else:
            excluded_tracks.append(f"{track_name} ({role})")
    
    if all_notes_corrected:
        corrected_root, corrected_scale = _analyze_key_from_pitches(all_notes_corrected)
        # Always show re-analysis result, even if it's the same
        old_key = f"{root_note} {scale_type}"
        new_key = f"{corrected_root} {corrected_scale}"
        
        if excluded_tracks:
            print(Fore.CYAN + f"Re-analyzing key with correct role assignments (excluding: {', '.join(excluded_tracks[:3])}{'...' if len(excluded_tracks) > 3 else ''})" + Style.RESET_ALL)
        
        # Always update if we got a different result (and it's not just default C major)
        if (corrected_root != root_note or corrected_scale != scale_type):
            if not (corrected_root == "C" and corrected_scale == "major"):
                print(Fore.GREEN + f"Corrected key detection: {old_key} → {new_key}" + Style.RESET_ALL)
                root_note = corrected_root
                scale_type = corrected_scale
            else:
                print(Fore.YELLOW + f"Re-analysis returned default C major (keeping {old_key})" + Style.RESET_ALL)
        else:
            print(Fore.CYAN + f"Re-analysis confirmed key: {new_key}" + Style.RESET_ALL)

    # Build instruments list proposal for config
    instruments_cfg = [
        {"name": a["instrument_name"], "program_num": next((t.get("program_num", 0) for t in merged_tracks if t.get("instrument_name") == a["instrument_name"]), 0), "role": a.get("role", "complementary")}
        for a in assigned
    ]

    # Inspiration text (English)
    inspiration_text = generate_inspiration_with_llm(
        config, genre, user_inspiration, summaries, bpm=bpm, ts=ts, bars_per_section=bars_per_section, instruments_for_ref=instruments_cfg
    )
    _print_llm_debug("Inspiration", "(see function generate_inspiration_with_llm)", inspiration_text, expects_json=False)

    # Sections
    section_count = max(1, total_bars // bars_per_section)
    sections = generate_section_descriptions_with_llm(config, genre, bars_per_section, section_count, instruments_cfg)
    try:
        _print_llm_debug("Sections", "(see function generate_section_descriptions_with_llm)", json.dumps(sections)[:1200], expects_json=True)
    except Exception:
        pass

    # Timestamp and artifact
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    analysis = {
        "type": "analysis",
        "timestamp": run_ts,
        "source_midi": os.path.relpath(midi_path, script_dir),
        "genre": genre,
        "bars_per_section": bars_per_section,
        "bpm": round(float(bpm), 2),
        "time_signature": ts,
        "key_scale": f"{root_note} {scale_type}",
        "root_note": root_note,
        "scale_type": scale_type,
        "inspiration": inspiration_text,
        "tracks": instruments_cfg,
        "structure": sections,
    }
    out_path = os.path.join(script_dir, f"analysis_run_{run_ts}.json")
    save_analysis_artifact(out_path, analysis)

    # Build potential settings and show a full preview BEFORE asking to write
    config_update = {
        "genre": genre,
        "inspiration": inspiration_text,
        "bpm": round(float(bpm)),
        # Use analyzed key/scale from MIDI instead of config
        "key_scale": f"{root_note} {scale_type}",
        "root_note": root_note,
        "scale_type": scale_type,
        "instruments": instruments_cfg,
        "time_signature": ts,
        # retain existing model fields automatically via merge
    }
    theme_defs = [{"label": s.get("label", f"Part_{i+1}"), "description": s.get("description", "")} for i, s in enumerate(sections)]
    config_update["part_length"] = bars_per_section

    # Build themes from analysis for integrated actions
    themes_from_analysis = split_tracks_into_sections(merged_tracks, bars_per_section, ts["beats_per_bar"]) if 'beats_per_bar' in ts else []
    # Overlay rich labels/descriptions from section plan to ensure LLM has strong guidance per part
    try:
        for i, th in enumerate(themes_from_analysis):
            if i < len(sections):
                sec = sections[i]
                if isinstance(sec, dict):
                    if sec.get('label'):
                        th['label'] = sec.get('label')
                    if sec.get('description'):
                        th['description'] = sec.get('description')
    except Exception:
        pass

    # Also persist a final artifact for downstream workflows (optimization/lyrics)
    try:
        # Build effective config similarly to integrated actions to keep fields consistent
        try:
            base_cfg = _load_config_roundtrip() or {}
        except Exception:
            base_cfg = {}
        effective_config = dict(base_cfg)
        try:
            if isinstance(config_update, dict):
                effective_config.update(config_update)
        except Exception:
            pass
        try:
            _ensure_scale_fields(effective_config)
            _remap_scale_for_generator(effective_config)
            if 'time_signature' not in effective_config and isinstance(ts, dict):
                effective_config['time_signature'] = ts
        except Exception:
            pass
        # Save final artifact using derived themes and section definitions
        if save_final_artifact and themes_from_analysis:
            save_final_artifact(effective_config, themes_from_analysis, bars_per_section, theme_defs, script_dir, run_ts)
    except Exception:
        pass

    # Full preview and single confirmation prompt
    if preview_settings_then_confirm(config_update, theme_defs):
        apply_to_song_generator(config_update, theme_defs)
        # Offer integrated post-analysis actions
        offer_integrated_actions(config_update, themes_from_analysis, bars_per_section)
    else:
        print(Fore.YELLOW + "Skipped saving settings." + Style.RESET_ALL)

    print(Fore.GREEN + "\nDone." + Style.RESET_ALL)


if __name__ == "__main__":
    main()


