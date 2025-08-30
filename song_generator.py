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
import argparse
if sys.platform == "win32":
    import msvcrt
import threading
from ruamel.yaml import YAML

# --- NEW: Global state for API key rotation ---
API_KEYS = []
CURRENT_KEY_INDEX = 0
SESSION_MODEL_OVERRIDE = None  # Optional session-wide model override via hotkey
# Hotkey runtime state
PROMPTED_CUSTOM_THIS_STEP = False  # Guards custom prompt per step
HOTKEY_MONITOR_STARTED = False
REQUESTED_SWITCH_MODEL = None     # 'gemini-2.5-pro' | 'gemini-2.5-flash' | '__ASK__' | custom
REQUEST_SET_SESSION_DEFAULT = False
ABORT_CURRENT_STEP = False        # Soft-abort signal; we discard current result when it returns
AUTO_ESCALATE_TO_PRO = False      # If True and using flash, auto-switch to pro after N failures per track
AUTO_ESCALATE_THRESHOLD = 6
DEFER_CURRENT_TRACK = False       # If True, immediately defer the current track (skip and push to end)
HOTKEY_DEBOUNCE_SEC = 0.8         # Debounce window for hotkeys
_LAST_HOTKEY_TS = {'1': 0.0, '2': 0.0, '3': 0.0, '0': 0.0, 'a': 0.0, 'd': 0.0, 'h': 0.0}
REDUCE_CONTEXT_THIS_STEP = False  # If True, halve historical context for the current step
REDUCE_CONTEXT_HALVES = 0        # Number of times to halve context for the current step
LAST_CONTEXT_COUNT = 0           # Last known number of context themes (for hotkey preview)
PLANNED_CONTEXT_COUNT = 0        # Planned context size after pending halvings (preview)

# --- Helper: classify 429 quota messages (best-effort) ---
def _classify_quota_error(err_text: str) -> str:
    """Heuristic classification of quota type from error text.
    Priority: per-day > per-hour > per-minute > rate-limit > user/project/unknown
    """
    try:
        t = (err_text or "").lower()
        # Normalize separators
        t = t.replace("-", " ")
        # Prefer explicit daily/hourly/minute windows first
        if any(k in t for k in ["per day", "daily", "per 24 hours", "per 24hrs", "per 24 hr", "per 1 day", "per day per user"]):
            return "per-day"
        if any(k in t for k in ["per hour", "hourly", "per 1 hour", "per 60 minutes", "per 3600 seconds", "per 3600s"]):
            return "per-hour"
        # Minute windows: allow variants like "per 60 seconds", "per minute"
        if any(k in t for k in ["per minute", "per 1 minute", "per 60 seconds", "per 60s", "per 60 sec"]):
            return "per-minute"
        # Generic rate limit hints with timeframe tokens
        if ("rate limit" in t or "rate limit".replace(" ","") in t) and "hour" in t:
            return "per-hour"
        if ("rate limit" in t or "rate limit".replace(" ","") in t) and "day" in t:
            return "per-day"
        if "rate limit" in t or "rate limit".replace(" ","") in t or "rate-limit" in t:
            return "rate-limit"
        if "quota" in t and "user" in t and "project" in t:
            return "project-quota"
        if "quota" in t and "user" in t:
            return "user-quota"
    except Exception:
        pass
    return "unknown"

# --- Per-key cooldown management (per-minute quota) ---
KEY_COOLDOWN_UNTIL = {}  # index->unix_timestamp until which key is cooling down
PER_MINUTE_COOLDOWN_SECONDS = 60
PER_HOUR_COOLDOWN_SECONDS = 3600
PER_DAY_COOLDOWN_SECONDS = 86400

def _is_key_available(idx: int) -> bool:
    until = KEY_COOLDOWN_UNTIL.get(idx, 0)
    return time.time() >= until

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

def _hotkey_monitor_loop(config: Dict):
    """Background loop to catch hotkeys continuously while long API calls run."""
    global PROMPTED_CUSTOM_THIS_STEP, REQUESTED_SWITCH_MODEL, REQUEST_SET_SESSION_DEFAULT, ABORT_CURRENT_STEP
    try:
        while True:
            # Allow disabling hotkeys via config
            if not config.get('enable_hotkeys', 1):
                time.sleep(0.2)
                continue
            if sys.platform == "win32" and msvcrt.kbhit():
                ch = msvcrt.getch().decode(errors='ignore').lower()
                now = time.time()
                if ch == '1':
                    if now - _LAST_HOTKEY_TS.get('1', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['1'] = now
                    REQUESTED_SWITCH_MODEL = 'gemini-2.5-pro'; ABORT_CURRENT_STEP = True
                    print(Fore.YELLOW + "\nModel switch requested: gemini-2.5-pro (will restart current step)" + Style.RESET_ALL)
                elif ch == '2':
                    if now - _LAST_HOTKEY_TS.get('2', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['2'] = now
                    REQUESTED_SWITCH_MODEL = 'gemini-2.5-flash'; ABORT_CURRENT_STEP = True
                    print(Fore.YELLOW + "\nModel switch requested: gemini-2.5-flash (will restart current step)" + Style.RESET_ALL)
                elif ch == 'a':
                    # Toggle auto-escalate mode
                    if now - _LAST_HOTKEY_TS.get('a', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['a'] = now
                    global AUTO_ESCALATE_TO_PRO
                    AUTO_ESCALATE_TO_PRO = not AUTO_ESCALATE_TO_PRO
                    state = 'ON' if AUTO_ESCALATE_TO_PRO else 'OFF'
                    print(Fore.CYAN + f"\nAuto-escalate to pro after {AUTO_ESCALATE_THRESHOLD} failures: {state}" + Style.RESET_ALL)
                elif ch == '3':
                    if now - _LAST_HOTKEY_TS.get('3', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['3'] = now
                    if not PROMPTED_CUSTOM_THIS_STEP:
                        PROMPTED_CUSTOM_THIS_STEP = True
                        suggestion = config.get('custom_model_name') or ''
                        prompt_txt = f"Enter custom model name (e.g., gemini-2.5-pro){' ['+suggestion+']' if suggestion else ''}: "
                        try:
                            entered = input(Fore.GREEN + prompt_txt + Style.RESET_ALL).strip()
                        except Exception:
                            entered = ''
                        custom = entered or suggestion
                        if custom:
                            config['custom_model_name'] = custom
                            REQUESTED_SWITCH_MODEL = custom; ABORT_CURRENT_STEP = True
                            print(Fore.YELLOW + f"\nModel switch requested: {custom} (will restart current step)" + Style.RESET_ALL)
                elif ch == '0':
                    if now - _LAST_HOTKEY_TS.get('0', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['0'] = now
                    REQUEST_SET_SESSION_DEFAULT = True
                    print(Fore.CYAN + "\nSession default will be set to current model after restart of step." + Style.RESET_ALL)
                elif ch == 'd':
                    if now - _LAST_HOTKEY_TS.get('d', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['d'] = now
                    # Defer current track
                    global DEFER_CURRENT_TRACK
                    DEFER_CURRENT_TRACK = True
                    print(Fore.MAGENTA + "\nDeferred: current track will be moved to the end of the queue." + Style.RESET_ALL)
                elif ch == 'h':
                    if now - _LAST_HOTKEY_TS.get('h', 0.0) < HOTKEY_DEBOUNCE_SEC:
                        continue
                    _LAST_HOTKEY_TS['h'] = now
                    globals()['REDUCE_CONTEXT_THIS_STEP'] = True
                    globals()['REDUCE_CONTEXT_HALVES'] = globals().get('REDUCE_CONTEXT_HALVES', 0) + 1
                    globals()['ABORT_CURRENT_STEP'] = True
                    # Show a preview using LAST_CONTEXT_COUNT if known
                    try:
                        prev = int(globals().get('PLANNED_CONTEXT_COUNT') or globals().get('LAST_CONTEXT_COUNT') or 0)
                        if prev > 0:
                            new_count = max(1, prev // 2)
                            globals()['PLANNED_CONTEXT_COUNT'] = new_count
                            print(Fore.CYAN + f"\nRequested: halve context (#{globals()['REDUCE_CONTEXT_HALVES']}) for THIS step: {prev} → {new_count} parts (will restart current step)." + Style.RESET_ALL)
                            print(Fore.CYAN + f"Using {new_count}/{prev} previous parts next." + Style.RESET_ALL)
                        else:
                            print(Fore.CYAN + f"\nRequested: halve context (#{globals()['REDUCE_CONTEXT_HALVES']}) for THIS step (will restart current step)." + Style.RESET_ALL)
                    except Exception:
                        print(Fore.CYAN + f"\nRequested: halve context (#{globals()['REDUCE_CONTEXT_HALVES']}) for THIS step (will restart current step)." + Style.RESET_ALL)
            time.sleep(0.05)
    except Exception:
        pass

# --- CONFIGURATION HELPERS (NEW) ---

def print_header(title):
    """Prints a stylized header to the console."""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*80)
    print(f"--- {title}".ljust(79, " ") + "-")
    print("="*80 + Style.RESET_ALL + "\n")

def get_user_input(prompt, default=None):
    """Gets user input with a default value."""
    response = input(f"{Fore.GREEN}{prompt}{Style.RESET_ALL} ").strip()
    return response or default

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
        # This is not a fatal error here, as the user might just be using the menu
        return False
    
    print(Fore.CYAN + f"Found {len(API_KEYS)} API key(s). Starting with key #1.")
    return True

def get_next_api_key():
    """Rotates to the next available API key."""
    global CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    print(Fore.YELLOW + f"Switching to API key #{CURRENT_KEY_INDEX + 1}...")
    return API_KEYS[CURRENT_KEY_INDEX]

def get_instrument_name(track_dict: Dict) -> str:
    """Robustly gets the instrument name from a track dictionary, checking for common key variations."""
    return track_dict.get('instrument_name') or track_dict.get('instrument', 'Unknown Instrument')

# --- ROBUST CONFIG FILE PATH ---
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join the script's directory with the config file name to create an absolute path
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
# --- END ROBUST PATH ---

MAX_CONTEXT_CHARS = 1000000  # A safe buffer below the 1M token limit for Gemini
BEATS_PER_BAR = 4
TICKS_PER_BEAT = 480
# Limit for generated automation steps per curve to avoid huge MIDI files
MAX_AUTOMATION_STEPS = 200

# --- RESPONSE PARSING HELPERS (Robust JSON extraction) ---
def _extract_text_from_response(response) -> str:
    """Safely extracts raw text from a Gemini response. Returns '' on failure."""
    try:
        # Prefer candidates/parts if available
        if hasattr(response, 'candidates') and response.candidates:
            cand = response.candidates[0]
            if hasattr(cand, 'content') and cand.content and getattr(cand.content, 'parts', None):
                parts = cand.content.parts
                text_parts = []
                for p in parts:
                    # Each part may have a .text attribute
                    t = getattr(p, 'text', None)
                    if isinstance(t, str):
                        text_parts.append(t)
                if text_parts:
                    return "\n".join(text_parts).strip()
        # Fallback to response.text if present
        t = getattr(response, 'text', None)
        if isinstance(t, str):
            return t.strip()
    except Exception:
        pass
    return ""

def _extract_json_object(raw: str) -> str:
    """Extracts the largest plausible JSON object substring from raw text.
    Removes code fences and tries to match braces. Returns '' if not found.
    """
    if not raw:
        return ""
    # Remove common code fences
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    # If it already looks like a JSON object, return as-is
    if cleaned.startswith('{') and cleaned.endswith('}'):
        return cleaned
    # Find first '{' and attempt to balance braces
    start = cleaned.find('{')
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return cleaned[start:i+1]
    return ""

# Add new constants at the beginning of the file
AVAILABLE_LENGTHS = [4, 8, 16, 32, 64, 128]
DEFAULT_LENGTH = 16

# Initialize Colorama for console color support
init(autoreset=True)
# Cap for embedding context notes into prompts; too many causes malformed JSON on some models
MAX_NOTES_IN_CONTEXT = 500

# Constants
GENERATED_CODE_FILE = os.path.join(script_dir, "generated_code.py")

# --- Hotkey utilities (Windows only) ---
def _poll_model_switch(local_model_name: str, config: Dict) -> str:
    """Non-blocking hotkey listener to switch model at any time.
    1=gemini-2.5-pro, 2=gemini-2.5-flash, 3=custom (config.custom_model_name), 0=set session default to current local.
    Returns possibly updated local_model_name. Session override stored globally on '0'.
    """
    global PROMPTED_CUSTOM_THIS_STEP, REQUESTED_SWITCH_MODEL, REQUEST_SET_SESSION_DEFAULT, ABORT_CURRENT_STEP
    try:
        if sys.platform == "win32" and msvcrt.kbhit():
            ch = msvcrt.getch().decode().lower()
            if ch == '1':
                REQUESTED_SWITCH_MODEL = 'gemini-2.5-pro'
                ABORT_CURRENT_STEP = True
                print(Fore.YELLOW + "Model switch requested: gemini-2.5-pro (will restart current step)" + Style.RESET_ALL)
                return local_model_name
            if ch == '2':
                REQUESTED_SWITCH_MODEL = 'gemini-2.5-flash'
                ABORT_CURRENT_STEP = True
                print(Fore.YELLOW + "Model switch requested: gemini-2.5-flash (will restart current step)" + Style.RESET_ALL)
                return local_model_name
            if ch == '3':
                if not PROMPTED_CUSTOM_THIS_STEP:
                    PROMPTED_CUSTOM_THIS_STEP = True
                    try:
                        suggestion = config.get('custom_model_name') or ''
                        prompt_txt = f"Enter custom model name (e.g., gemini-2.5-pro){' ['+suggestion+']' if suggestion else ''}: "
                        entered = input(Fore.GREEN + prompt_txt + Style.RESET_ALL).strip()
                        custom = entered or suggestion
                        if custom:
                            config['custom_model_name'] = custom
                            REQUESTED_SWITCH_MODEL = custom
                            ABORT_CURRENT_STEP = True
                            print(Fore.YELLOW + f"Model switch requested: {custom} (will restart current step)" + Style.RESET_ALL)
                            return local_model_name
                    except Exception:
                        pass
            if ch == '0':
                # Persist current local as session-wide override
                REQUEST_SET_SESSION_DEFAULT = True
                print(Fore.CYAN + f"Session default will be set to current model after restart of step." + Style.RESET_ALL)
                return local_model_name
            if ch == 'd':
                # Request deferral of current track
                global DEFER_CURRENT_TRACK
                DEFER_CURRENT_TRACK = True
                print(Fore.MAGENTA + "Deferred: current track will be moved to the end of the queue." + Style.RESET_ALL)
                return local_model_name
    except Exception:
        pass
    return local_model_name

# --- Hotkey hint banner ---
def print_hotkey_hint(config: Dict, context: str = "") -> None:
    """Shows a concise, non-blocking hotkey hint (Windows only)."""
    try:
        if sys.platform != "win32":
            return
        custom = config.get('custom_model_name') or 'custom'
        ctx = f" [{context}]" if context else ""
        escalate_state = " [ON]" if ('AUTO_ESCALATE_TO_PRO' in globals() and AUTO_ESCALATE_TO_PRO) else " [OFF]"
        print(
            Style.DIM
            + Fore.CYAN
            + (
                f"Hotkeys{ctx}: 1=pro (this step), 2=flash (this step), 3={custom} (this step), 0=set session default, "
                f"h=halve context (this step), a=auto-escalate (flash→pro after {AUTO_ESCALATE_THRESHOLD} fails{escalate_state}), d=defer track"
            )
            + "\n  Note: 1/2/3 switch only this step. Press 0 to keep current as session default."
            + "\n        Changes take effect after the current request/backoff is interrupted (press any hotkey to interrupt waits)."
            + Style.RESET_ALL
        )
    except Exception:
        pass

# --- Interruptible backoff (hotkey-aware) ---
def _interruptible_backoff(wait_time: float, config: Dict, context_label: str = "") -> None:
    """Sleeps up to wait_time seconds but polls for hotkeys to interrupt immediately.
    If user presses 1/2/3/0: set model switch/session default and abort current step.
    If 'd': defer current track.
    If 'a': toggle auto-escalate.
    """
    try:
        end_t = time.time() + max(0.0, wait_time)
        if sys.platform != "win32":
            # Non-Windows: simple sleep
            time.sleep(max(0.0, wait_time))
            return
        print(Fore.CYAN + (f"Waiting {wait_time:.1f}s" + (f" [{context_label}]" if context_label else "") + 
              "; press 1/2/3/0 (model), 'h' (halve context), 'd' (defer), 'a' (auto-escalate)...") + Style.RESET_ALL)
        while time.time() < end_t:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode(errors='ignore').lower()
                now = time.time()
                if ch in _LAST_HOTKEY_TS and now - _LAST_HOTKEY_TS.get(ch, 0.0) < HOTKEY_DEBOUNCE_SEC:
                    continue
                if ch == '1':
                    _LAST_HOTKEY_TS['1'] = now
                    globals()['REQUESTED_SWITCH_MODEL'] = 'gemini-2.5-pro'
                    globals()['ABORT_CURRENT_STEP'] = True
                    print(Fore.YELLOW + "Model switch requested: gemini-2.5-pro (will restart current step)" + Style.RESET_ALL)
                    return
                if ch == '2':
                    _LAST_HOTKEY_TS['2'] = now
                    globals()['REQUESTED_SWITCH_MODEL'] = 'gemini-2.5-flash'
                    globals()['ABORT_CURRENT_STEP'] = True
                    print(Fore.YELLOW + "Model switch requested: gemini-2.5-flash (will restart current step)" + Style.RESET_ALL)
                    return
                if ch == '3':
                    _LAST_HOTKEY_TS['3'] = now
                    # Use saved custom model if present; no prompt during backoff
                    custom = config.get('custom_model_name')
                    if custom:
                        globals()['REQUESTED_SWITCH_MODEL'] = custom
                        globals()['ABORT_CURRENT_STEP'] = True
                        print(Fore.YELLOW + f"Model switch requested: {custom} (will restart current step)" + Style.RESET_ALL)
                        return
                if ch == '0':
                    _LAST_HOTKEY_TS['0'] = now
                    globals()['REQUEST_SET_SESSION_DEFAULT'] = True
                    print(Fore.CYAN + "Session default will be set to current model after restart of step." + Style.RESET_ALL)
                    return
                if ch == 'd':
                    _LAST_HOTKEY_TS['d'] = now
                    globals()['DEFER_CURRENT_TRACK'] = True
                    print(Fore.MAGENTA + "Deferred: current track will be moved to the end of the queue." + Style.RESET_ALL)
                    return
                if ch == 'a':
                    _LAST_HOTKEY_TS['a'] = now
                    globals()['AUTO_ESCALATE_TO_PRO'] = not globals().get('AUTO_ESCALATE_TO_PRO', False)
                    state = 'ON' if globals().get('AUTO_ESCALATE_TO_PRO', False) else 'OFF'
                    print(Fore.CYAN + f"Auto-escalate to pro after {AUTO_ESCALATE_THRESHOLD} failures: {state}" + Style.RESET_ALL)
                    return
                if ch == 'h':
                    _LAST_HOTKEY_TS['h'] = now
                    globals()['REDUCE_CONTEXT_THIS_STEP'] = True
                    globals()['REDUCE_CONTEXT_HALVES'] = globals().get('REDUCE_CONTEXT_HALVES', 0) + 1
                    globals()['ABORT_CURRENT_STEP'] = True
                    try:
                        prev = int(globals().get('PLANNED_CONTEXT_COUNT') or globals().get('LAST_CONTEXT_COUNT') or 0)
                        if prev > 0:
                            new_count = max(1, prev // 2)
                            globals()['PLANNED_CONTEXT_COUNT'] = new_count
                            print(Fore.CYAN + f"\nRequested: halve context (#{globals()['REDUCE_CONTEXT_HALVES']}) for THIS step: {prev} → {new_count} parts (will restart current step)." + Style.RESET_ALL)
                            print(Fore.CYAN + f"Using {new_count}/{prev} previous parts next." + Style.RESET_ALL)
                        else:
                            print(Fore.CYAN + f"\nRequested: halve context (#{globals()['REDUCE_CONTEXT_HALVES']}) for THIS step (will restart current step)." + Style.RESET_ALL)
                    except Exception:
                        print(Fore.CYAN + f"\nRequested: halve context (#{globals()['REDUCE_CONTEXT_HALVES']}) for THIS step (will restart current step)." + Style.RESET_ALL)
                    return
            time.sleep(0.2)
    except Exception:
        # Fallback to plain sleep on any error
        time.sleep(max(0.0, wait_time))

# --- Windowed Optimization Helpers (beta) ---
def _build_window_from_themes(themes: List[Dict], start_index: int, num_themes_in_window: int, theme_length_bars: int, beats_per_bar: int) -> Dict:
    """Creates a synthetic 'window theme' by merging consecutive themes into one timeline (notes in window-relative beats)."""
    window_tracks: Dict[str, Dict] = {}
    for offset in range(num_themes_in_window):
        idx = start_index + offset
        if idx >= len(themes):
            break
        theme = themes[idx]
        part_start_beats = offset * theme_length_bars * beats_per_bar
        for tr in theme.get('tracks', []):
            name = get_instrument_name(tr)
            target = window_tracks.setdefault(name, {
                'instrument_name': name,
                'program_num': tr.get('program_num', 0),
                'role': tr.get('role', 'complementary'),
                'notes': []
            })
            # Merge notes with offset
            for n in tr.get('notes', []):
                try:
                    sb = float(n.get('start_beat', 0)) + part_start_beats
                    dur = float(n.get('duration_beats', 0))
                    vel = int(n.get('velocity', 0))
                    if dur > 0 and 1 <= vel <= 127:
                        target['notes'].append({
                            'pitch': int(n.get('pitch', 0)),
                            'start_beat': sb,
                            'duration_beats': dur,
                            'velocity': vel
                        })
                except Exception:
                    continue
    return {
        'label': f"Window_{start_index+1}_{min(start_index+num_themes_in_window, len(themes))}",
        'tracks': list(window_tracks.values())
    }

def _split_window_back_into_themes(window_tracks: List[Dict], themes: List[Dict], start_index: int, num_themes_in_window: int, theme_length_bars: int, beats_per_bar: int) -> None:
    """Splits optimized window tracks back into the original themes by clamping to each sub-part and converting to part-relative time."""
    part_len_beats = theme_length_bars * beats_per_bar
    for tr in window_tracks:
        name = get_instrument_name(tr)
        role = tr.get('role', 'complementary')
        program_num = tr.get('program_num', 0)
        notes = tr.get('notes', [])
        for offset in range(num_themes_in_window):
            idx = start_index + offset
            if idx >= len(themes):
                break
            part_start = offset * part_len_beats
            part_end = part_start + part_len_beats
            # Extract notes that fall into this part, convert to part-relative
            rel_notes = []
            for n in notes:
                try:
                    sb_abs = float(n.get('start_beat', 0))
                    dur = float(n.get('duration_beats', 0))
                    vel = int(n.get('velocity', 0))
                    if dur <= 0 or vel < 1:
                        continue
                    nbeg = sb_abs
                    nend = sb_abs + dur
                    if nend <= part_start or nbeg >= part_end:
                        continue
                    # Clamp to boundaries
                    new_start = max(part_start, nbeg)
                    new_end = min(part_end, nend)
                    new_dur = max(0.0, new_end - new_start)
                    if new_dur <= 0:
                        continue
                    rel_notes.append({
                        'pitch': int(n.get('pitch', 0)),
                        'start_beat': round(new_start - part_start, 4),
                        'duration_beats': round(new_dur, 4),
                        'velocity': vel
                    })
                except Exception:
                    continue
            # Update or create the corresponding track in the theme
            theme = themes[idx]
            placed = False
            for t in theme.get('tracks', []):
                if get_instrument_name(t) == name:
                    t['notes'] = rel_notes
                    placed = True
                    break
            if not placed:
                theme.setdefault('tracks', []).append({
                    'instrument_name': name,
                    'program_num': program_num,
                    'role': role,
                    'notes': rel_notes
                })

def create_windowed_optimization(config: Dict, themes: List[Dict], theme_length_bars: int, window_bars: int, script_dir: str, run_timestamp: str, user_optimization_prompt: str = "", resume_start_index: int = 0, seam_mode: bool = False) -> List[Dict]:
    """Optimizes the song in larger temporal windows (beta). Non-overlapping windows for simplicity."""
    beats_per_bar = config["time_signature"]["beats_per_bar"]
    # Validate divisibility
    if window_bars % max(1, theme_length_bars) != 0:
        print(Fore.YELLOW + f"Window {window_bars} bars is not divisible by part length {theme_length_bars}. Skipping." + Style.RESET_ALL)
        # After all windows are processed, create a combined final song MIDI for this pass
        try:
            final_song_data = merge_themes_to_song_data(themes, config, theme_length_bars)
            base = build_final_song_basename(config, themes, run_timestamp, resumed=True)
            suffix = f"_win{window_bars}" + ("_seam" if seam_mode else "")
            final_path = os.path.join(script_dir, f"{base}{suffix}.mid")
            create_midi_from_json(final_song_data, config, final_path)
        except Exception:
            pass
        # After all windows processed, always export a final combined MIDI for this pass
        try:
            final_song_data = merge_themes_to_song_data(themes, config, theme_length_bars)
            base = build_final_song_basename(config, themes, run_timestamp, resumed=True)
            suffix = f"_win{window_bars}" + ("_seam" if seam_mode else "")
            final_path = os.path.join(script_dir, f"{base}{suffix}.mid")
            create_midi_from_json(final_song_data, config, final_path)
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Could not create final combined MIDI after windowed optimization: {e}" + Style.RESET_ALL)
        return themes
    window_parts = window_bars // theme_length_bars
    if window_parts <= 1:
        print(Fore.YELLOW + "Window size must cover at least 2 parts for meaningful context." + Style.RESET_ALL)
        return themes

    try:
        for start in range(max(0, resume_start_index), len(themes), window_parts):
            end = min(len(themes), start + window_parts)
            actual_parts = end - start
            if actual_parts < 2:
                break
            print(Fore.MAGENTA + f"\n--- Windowed Optimization (beta): Parts {start+1}-{end} ({window_bars} bars) ---" + Style.RESET_ALL)

            window_theme = _build_window_from_themes(themes, start, actual_parts, theme_length_bars, beats_per_bar)

            # Optimize each track in the window using the existing single-track optimizer
            optimized_tracks = []
            inner_context = [t for t in window_theme['tracks']]  # shallow copy for reference
            historical_context = themes[:start]

            # Track-level resume support: try to read last track index
            track_resume_index = 0
            try:
                rf_list = find_progress_files(script_dir)
                for rf in rf_list:
                    pdata = _load_progress_silent(rf)
                    if pdata and pdata.get('type') == 'window_optimization' and pdata.get('window_bars') == window_bars and int(pdata.get('current_window_start_index', -1)) == start:
                        track_resume_index = int(pdata.get('current_track_in_window', 0))
                        break
            except Exception:
                track_resume_index = 0

            # Build window summaries (full descriptions of the parts in this window, lightly capped)
            window_summaries = []
            try:
                for offset in range(actual_parts):
                    idx = start + offset
                    if idx < len(themes):
                        lbl = themes[idx].get('label', f'Part_{idx+1}')
                        desc = str(themes[idx].get('description', ''))
                        if len(desc) > 800:
                            desc = desc[:800] + '...'
                        window_summaries.append(f"- {lbl}: {desc}")
            except Exception:
                pass

            for track_idx, base_tr in enumerate(window_theme['tracks']):
                if track_idx < track_resume_index:
                    continue
                inst_name = get_instrument_name(base_tr)
                role = base_tr.get('role', 'complementary')
                # Build minimal track_to_optimize payload
                track_to_optimize = {
                    'instrument_name': inst_name,
                    'program_num': base_tr.get('program_num', 0),
                    'role': role,
                    'notes': list(base_tr.get('notes', []))
                }
                # Inner context excluding current track
                other_tracks = [t for i, t in enumerate(inner_context) if i != track_idx]
                print(Fore.BLUE + f"Optimizing window track: {inst_name} (role: {role})" + Style.RESET_ALL)
                opt_prompt = user_optimization_prompt or ""
                if seam_mode and "[SEAM_MODE]" not in opt_prompt:
                    opt_prompt = "[SEAM_MODE] " + opt_prompt
                # Prepend full window part descriptions to give narrative intent
                if window_summaries:
                    opt_prompt = (
                        "Window context (full descriptions):\n" + "\n".join(window_summaries) + "\n\n" + opt_prompt
                    )
                optimized_track, _tokens = generate_optimization_data(
                    config, window_bars, track_to_optimize, role,
                    window_theme['label'], window_theme.get('label', ''),
                    historical_context, other_tracks, start,
                    user_optimization_prompt=opt_prompt
                )
                if optimized_track:
                    # Preserve metadata
                    optimized_track['instrument_name'] = inst_name
                    optimized_track['program_num'] = base_tr.get('program_num', 0)
                    optimized_track['role'] = role
                    optimized_tracks.append(optimized_track)
                else:
                    # Fallback to original track
                    optimized_tracks.append(track_to_optimize)

                # Immediately write back only this track into original themes for resilience
                try:
                    _split_window_back_into_themes([optimized_tracks[-1]], themes, start, actual_parts, theme_length_bars, beats_per_bar)
                except Exception:
                    pass

                # (Changed) No immediate per-track MIDI export; MIDIs will be written after the whole window finishes

                # Save per-track resumable progress
                try:
                    progress_payload_tr = {
                        'type': 'window_optimization',
                        'config': config,
                        'theme_length': theme_length_bars,
                        'window_bars': window_bars,
                        'current_window_start_index': start,
                        'current_track_in_window': track_idx + 1,
                        'themes': themes,
                        'user_optimization_prompt': user_optimization_prompt,
                        'timestamp': run_timestamp
                    }
                    save_progress(progress_payload_tr, script_dir, run_timestamp)
                except Exception:
                    pass

            # Write back into original themes by splitting the entire window (noop for already applied tracks)
            _split_window_back_into_themes(optimized_tracks, themes, start, actual_parts, theme_length_bars, beats_per_bar)

            # Save intermediate progress artifact for safety
            try:
                save_final_artifact(config, themes, theme_length_bars, [], script_dir, run_timestamp)
            except Exception:
                pass

            # Export MIDI once per part after the entire window is completed (finalized state for this window)
            try:
                part_len_beats = theme_length_bars * beats_per_bar
                for idx in range(start, end):
                    theme = themes[idx]
                    theme_data = {"tracks": theme.get('tracks', [])}
                    # Reuse generation filename logic
                    theme_label = theme.get('label', f"Part_{idx+1}")
                    base_name = generate_filename(config, script_dir, theme_length_bars, theme_label, idx, run_timestamp)
                    # Notes have been split back to part-relative timing; don't subtract a time offset here
                    create_part_midi_from_theme(theme_data, config, base_name, time_offset_beats=0, section_length_beats=part_len_beats)
                    # Ensure filesystem ordering by mod-time
                    try:
                        time.sleep(1)
                    except Exception:
                        pass
            except Exception as e:
                print(Fore.YELLOW + f"Warning: Could not export window part MIDIs: {e}" + Style.RESET_ALL)

            # Save resumable progress after each window
            try:
                progress_payload = {
                    'type': 'window_optimization',
                    'config': config,
                    'theme_length': theme_length_bars,
                    'window_bars': window_bars,
                    'current_window_start_index': start + actual_parts,
                    'themes': themes,
                    'user_optimization_prompt': user_optimization_prompt,
                    'timestamp': run_timestamp
                }
                save_progress(progress_payload, script_dir, run_timestamp)
            except Exception:
                pass

        return themes
    except Exception as e:
        print(Fore.RED + f"Windowed optimization failed: {e}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return themes

# --- Filename helpers ---
def _sanitize_filename_component(text: str) -> str:
    try:
        text = re.sub(r'[\\/*?:"<>|]', '', text)
        text = re.sub(r'\s+', '_', text).strip('_')
        return text[:60] if len(text) > 60 else text
    except Exception:
        return "untitled"

def build_final_song_basename(config: Dict, themes: List[Dict], timestamp: str, *, resumed: bool = False, opt_iteration: int | None = None) -> str:
    try:
        genre = _sanitize_filename_component(config.get("genre", "audio"))
        key = _sanitize_filename_component(config.get("key_scale", "").replace('#', 's'))
        bpm = str(round(float(config.get("bpm", 120))))
        num_parts = str(len(themes)) if themes else "0"
        first_label = _sanitize_filename_component((themes[0].get('label') if themes and isinstance(themes[0], dict) else themes[0].get('theme_label', 'A')) if themes else "start")
        last_label = _sanitize_filename_component((themes[-1].get('label') if themes and isinstance(themes[-1], dict) else themes[-1].get('theme_label', 'Z')) if themes else "end")
        base = f"Final_{genre}_{key}_{bpm}bpm_{num_parts}parts_{first_label}-to-{last_label}_{timestamp}"
        if resumed:
            base = f"{base}_resumed"
        if opt_iteration is not None:
            base = f"{base}_opt_{opt_iteration}"
        return base
    except Exception:
        safe_ts = _sanitize_filename_component(timestamp)
        return f"Final_Song_{safe_ts}"

def load_config(config_file):
    """Loads and validates the configuration from a YAML file using ruamel.yaml."""
    print(Fore.CYAN + "Loading configuration..." + Style.RESET_ALL)
    try:
        yaml = YAML()
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.load(f)

        if not config:
            raise ValueError("Config file is empty or invalid.")

        # Validate critical fields first
        if not config.get("api_key"):
            raise ValueError("API key is missing in configuration.")
            
        if not config.get("model_name"):
            raise ValueError("Model name is missing in configuration.")

        # A list of required fields for the configuration to be valid.
        required_fields = [
            "inspiration", "genre", "bpm", "key_scale",
            "api_key", "model_name", "instruments", "time_signature"
        ]
        
        # Set defaults for missing non-critical fields and validate
        if "use_call_and_response" not in config: config["use_call_and_response"] = 0
        elif config["use_call_and_response"] not in [0, 1]: raise ValueError("use_call_and_response must be 0 or 1.")
        
        if "number_of_iterations" not in config: config["number_of_iterations"] = 1
        else:
            try:
                if int(config["number_of_iterations"]) < 1: raise ValueError()
            except: raise ValueError("number_of_iterations must be an integer of 1 or greater.")
        
        if "temperature" not in config: config["temperature"] = 1.0
        else:
            try:
                if not 0.0 <= float(config["temperature"]) <= 2.0: raise ValueError()
            except: raise ValueError("temperature must be a number between 0.0 and 2.0.")

        if "context_window_size" not in config: config["context_window_size"] = -1
        else:
            try:
                if int(config["context_window_size"]) < -1: raise ValueError()
            except: raise ValueError("context_window_size must be an integer (-1 for dynamic, or 0 and greater).")

        if "max_output_tokens" not in config: config["max_output_tokens"] = 8192
        else:
            try:
                if int(config["max_output_tokens"]) < 1: raise ValueError()
            except: raise ValueError("max_output_tokens must be an integer of 1 or greater.")
        
        if "automation_settings" not in config: config["automation_settings"] = {}
        for key, default in [("use_pitch_bend", 0), ("use_cc_automation", 0), ("use_sustain_pedal", 0)]:
            if key not in config["automation_settings"]: config["automation_settings"][key] = default
            elif config["automation_settings"][key] not in [0, 1]:
                print(Fore.YELLOW + f"Warning: Invalid value for '{key}'. Defaulting to {default}.")
                config["automation_settings"][key] = default
        if "allowed_cc_numbers" not in config["automation_settings"]: config["automation_settings"]["allowed_cc_numbers"] = []

        for field in required_fields:
            if field not in config:
                if field == "time_signature":
                    config["time_signature"] = {"beats_per_bar": 4, "beat_value": 4}
                else:
                    raise ValueError(f"Required field '{field}' is missing in the configuration.")

        # Parse key_scale into root_note and scale_type
        key_parts = config["key_scale"].lower().split()
        note_map = {
            "c": 60, "c#": 61, "db": 61, "d": 62, "d#": 63, "eb": 63, "e": 64, 
            "f": 65, "f#": 66, "gb": 66, "g": 67, "g#": 68, "ab": 68, "a": 69, 
            "a#": 70, "bb": 70, "b": 71
        }
        
        if len(key_parts) >= 2:
            root = key_parts[0]
            config["root_note"] = note_map.get(root, 60)
            config["scale_type"] = " ".join(key_parts[1:])
        else:
            config["root_note"] = 60
            config["scale_type"] = "major"

        if isinstance(config.get("time_signature"), str) and "/" in config["time_signature"]:
            try:
                beats, value = config["time_signature"].split("/")
                config["time_signature"] = {"beats_per_bar": int(beats), "beat_value": int(value)}
            except ValueError:
                config["time_signature"] = {"beats_per_bar": 4, "beat_value": 4}


        print(Fore.GREEN + "Configuration loaded and validated successfully." + Style.RESET_ALL)
        return config
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Error loading configuration: {str(e)}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while loading the configuration: {e}")

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

def _expand_pattern_blocks(pattern_blocks: List[Dict], length_bars: int, beats_per_bar: int) -> List[Dict]:
    """
    Expands a compact pattern representation into a list of note objects.
    Expected block schema (flexible, fields optional):
    - length_bars (int): length of this block in bars (default 1)
    - subdivision (int): steps per bar (e.g., 16, 32, 64; default 16)
    - bar_repeats (int): how many times this block repeats in sequence (default 1)
    - transpose (int): semitone transpose for this block (default 0)
    - octave_shift (int): octave shift in 12-semitone steps (default 0)
    - steps (list): each step definition:
        {"pitch": int, "velocity": int, "gate": float (0..1),
         one of: "mask": "0101..." or "indices": [int, ...]}

    Notes will be clamped into the total part length (length_bars * beats_per_bar).
    """
    try:
        notes: List[Dict] = []
        total_beats = float(length_bars * beats_per_bar)
        current_bar_offset = 0.0

        if not isinstance(pattern_blocks, list):
            return notes

        for block in pattern_blocks:
            if not isinstance(block, dict):
                continue
            length_bars_blk = int(block.get("length_bars", 1))
            subdivision = int(block.get("subdivision", 16))
            bar_repeats = int(block.get("bar_repeats", 1))
            transpose = int(block.get("transpose", 0)) + int(block.get("octave_shift", 0)) * 12

            step_count = max(1, length_bars_blk * max(1, subdivision))
            step_duration = float(beats_per_bar) / max(1, subdivision)

            steps = block.get("steps", [])
            if not isinstance(steps, list):
                steps = []

            # Place this block sequentially; repeat as requested
            for rep in range(max(1, bar_repeats)):
                bar_base = (current_bar_offset + rep * length_bars_blk) * float(beats_per_bar)

                for stepdef in steps:
                    if not isinstance(stepdef, dict):
                        continue
                    base_pitch = int(stepdef.get("pitch", 60)) + transpose
                    base_pitch = max(0, min(127, base_pitch))
                    velocity = int(stepdef.get("velocity", 100))
                    velocity = max(1, min(127, velocity))
                    gate = stepdef.get("gate", 0.5)
                    try:
                        gate = float(gate)
                    except Exception:
                        gate = 0.5
                    if gate <= 0:
                        gate = 0.5

                    indices: List[int] = []
                    if isinstance(stepdef.get("indices"), list):
                        try:
                            indices = [int(i) for i in stepdef.get("indices", [])]
                        except Exception:
                            indices = []
                    elif isinstance(stepdef.get("mask"), str):
                        mask = stepdef.get("mask", "")
                        step_len = min(len(mask), step_count)
                        for idx in range(step_len):
                            if mask[idx] == '1':
                                indices.append(idx)

                    # Create notes for each index within this block
                    for idx in indices:
                        if not isinstance(idx, int):
                            continue
                        if idx < 0 or idx >= step_count:
                            continue
                        start_beat = bar_base + float(idx) * step_duration
                        if start_beat >= total_beats:
                            continue
                        duration = max(0.01, step_duration * gate)
                        if start_beat + duration > total_beats:
                            duration = max(0.0, total_beats - start_beat)
                        if duration <= 0:
                            continue
                        notes.append({
                            "pitch": base_pitch,
                            "start_beat": float(start_beat),
                            "duration_beats": float(duration),
                            "velocity": velocity
                        })

            # Advance placement cursor by the total repeated length of this block
            current_bar_offset += float(length_bars_blk * max(1, bar_repeats))
            if current_bar_offset * beats_per_bar >= total_beats:
                break

        return notes
    except Exception:
        # On any unexpected error, return empty list and let caller fallback
        return []

def create_theme_prompt(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int, dialogue_role: str, theme_label: str, theme_description: str, previous_themes_full_history: List[Dict], current_theme_index: int):
    """
    Creates a universal, music-intelligent prompt that is tailored to generating a new theme based on previous ones.
    This version dynamically manages the context size and includes instructions for MIDI automation.
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
        notes_cap = 0
        try:
            notes_cap = int(config.get('_notes_context_cap', MAX_NOTES_IN_CONTEXT))
        except Exception:
            notes_cap = MAX_NOTES_IN_CONTEXT
        context_prompt_part = "**Inside the current theme, you have already written these parts. Compose a new part that fits with them:**\n"
        for track in context_tracks:
            # Use a more compact representation for context to save tokens and reduce JSON failures
            try:
                notes = track.get('notes', [])
                if isinstance(notes, list) and len(notes) > max(1, notes_cap):
                    head = notes[:max(1, notes_cap)//2]
                    tail = notes[-max(1, notes_cap)//2:]
                    notes = head + tail
            except Exception:
                notes = track.get('notes', [])
            notes_as_str = json.dumps(notes, separators=(',', ':'))
            context_prompt_part += f"- **{track['instrument_name']}** (Role: {track['role']}):\n```json\n{notes_as_str}\n```\n"
        context_prompt_part += "\n"
    
    # Main Task description based on theme
    theme_task_instruction = ""
    timing_rule = f"5.  **Timing is Absolute:** 'start_beat' is the absolute position from the beginning of the {length}-bar clip.\n"

    if current_theme_index == 0:
        theme_task_instruction = (
            f"**Your Task: Compose the First Musical Theme**\n"
            f"This is the very first section of the song. Your goal is to establish the main musical ideas.\n"
            f"**Theme Name/Label:** {theme_label}\n"
            f"**Creative Direction for this Theme:** {theme_description}\n"
            f"**CRITICAL AUTOMATION TASK:** Your primary goal is to translate any automation cues from the creative direction above (like 'pitch bend up', 'filter sweep') into the precise JSON format specified in the 'Advanced MIDI Automation' section. This is not optional; it is the most important part of your task. Faithfully convert the described musical effects into the corresponding JSON structures.\n"
        )
    else:
        total_previous_beats = current_theme_index * total_beats_per_theme
        theme_task_instruction = (
            f"**Your Task: Compose a New, Contrasting Theme starting from beat {total_previous_beats}**\n"
            f"You must create a new musical section that logically follows the previous themes, but has a distinct character.\n"
            f"**Theme Name/Label for this NEW Theme:** {theme_label}\n"
            f"**Creative Direction for this NEW Theme:** {theme_description}\n"
            "Analyze the previous themes and create something that complements them while bringing a fresh energy or emotion.\n"
            f"**CRITICAL AUTOMATION TASK:** Your primary goal is to translate any automation cues from the creative direction above (like 'pitch bend up', 'filter sweep') into the precise JSON format specified in the 'Advanced MIDI Automation' section. This is not optional; it is the most important part of your task. Faithfully convert the described musical effects into the corresponding JSON structures.\n"
        )
        timing_rule = f"5.  **Timing is Absolute:** 'start_beat' is the absolute position from the beginning of the *entire song composition so far*.\n"

    # Dialogue instructions (Call & Response)
    call_and_response_instructions = ""
    try:
        if str(dialogue_role).lower() == 'call':
            call_and_response_instructions = (
                "**Call & Response (this track is the CALL):**\n"
                "- Introduce a clear 1–2 bar motif.\n"
                "- Leave air afterwards (rests) to invite a response.\n"
                "- Avoid always starting phrases exactly on beat 1; try upbeat or beat 2/4 entries.\n"
                "- Keep the final 0.5–1 bar simpler to make room for the response.\n\n"
            )
        elif str(dialogue_role).lower() == 'response':
            call_and_response_instructions = (
                "**Call & Response (this track is the RESPONSE):**\n"
                "- Enter on off‑beats or slightly after the call; do not start on the exact same beat as the call.\n"
                "- Reference the last 2–4 notes of the call (rhythm or contour), but avoid unison and avoid the same register within ±2 semitones.\n"
                "- Complement rhythmically/harmonically; do not cover the call at the same time.\n"
                "- Keep density slightly lower than the call unless a lift is required.\n\n"
            )
        else:
            call_and_response_instructions = ""
    except Exception:
        call_and_response_instructions = ""

    # --- NEW: Define Polyphony and Key Rules ---
    POLYPHONIC_ROLES = {"harmony", "chords", "pads", "atmosphere", "texture", "guitar"}
    EXPRESSIVE_MONOPHONIC_ROLES = {"lead", "melody", "vocal"}
    if role in POLYPHONIC_ROLES:
        polyphony_rule = "2.  **Polyphonic:** Notes for this track CAN overlap."
    elif role in EXPRESSIVE_MONOPHONIC_ROLES:
        polyphony_rule = "2.  **Expressive Monophonic:** Notes should primarily be played one at a time, but short overlaps are permitted for expressive legato phrasing."
    else: 
        polyphony_rule = "2.  **Strictly Monophonic:** The notes in the JSON object's 'notes' array must NOT overlap in time."

    stay_in_key_rule = f"3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n"
    if role in ["drums", "percussion", "kick_and_snare"]:
        stay_in_key_rule = "3.  **Use Drum Map:** You must adhere to the provided Drum Map for all note pitches.\n"
    # --- END NEW ---


    # Role-specific instructions and rules
    role_instructions = get_role_instructions_for_generation(role, config)
    drum_map_instructions = ""
    if role in ["drums", "percussion", "kick_and_snare"]:
        drum_map_instructions = (
            "**Drum Map Guidance (Addictive Drums 2 Standard):**\n"
            "You MUST use the following MIDI notes for the corresponding drum sounds.\n"
            "- **Kick:** MIDI Note 36\n"
            "- **Snare (Center Hit):** MIDI Note 38\n"
            "- **Snare (Rimshot):** MIDI Note 40\n"
            "- **Hi-Hat (Closed):** MIDI Note 42\n"
            "- **Hi-Hat (Pedal Close):** MIDI Note 44\n"
            "- **Hi-Hat (Open):** MIDI Note 46\n"
            "- **Crash Cymbal 1:** MIDI Note 49\n"
            "- **Ride Cymbal 1:** MIDI Note 51\n"
            "- **High Tom:** MIDI Note 50\n"
            "- **Mid Tom:** MIDI Note 48\n"
            "- **Low Tom:** MIDI Note 45\n"
            "Velocity guidance: ghost snare < 45; closed HH vary 40–90; crash ≥ 100.\n\n"
        )
    
    # Polyphony and Key rules
    # ... (polyphony and key rules remain the same)
    
    # --- NEW: Automation Instructions ---
    automation_instructions = ""
    automation_settings = config.get("automation_settings", {})
    use_pitch_bend = automation_settings.get("use_pitch_bend", 0) == 1
    use_cc_automation = automation_settings.get("use_cc_automation", 0) == 1
    use_sustain_pedal = automation_settings.get("use_sustain_pedal", 0) == 1
    allowed_cc = automation_settings.get("allowed_cc_numbers", [])

    # --- Conditionally build the automation instructions ---
    
    # Header for the automation section
    if use_pitch_bend or use_cc_automation or use_sustain_pedal:
        automation_instructions += "**--- Advanced MIDI Automation ---**\n"
        automation_instructions += "To create expressive performances, you MUST use the following JSON structures. You will translate the creative direction into these precise formats.\n\n"

    # Note-based and Track-based automations (Pitch Bend and CC)
    if use_pitch_bend or use_cc_automation:
        # Note-based
        automation_instructions += "1.  **Note-Based Automations (Attached to a specific note):**\n"
        automation_instructions += "    - Add an `\"automations\"` object directly inside a note.\n"
        
        automation_example = []
        if use_pitch_bend:
            automation_example.append("""
        "pitch_bend": [{
            "type": "curve", "start_beat": 0.0, "end_beat": 0.5,
            "start_value": -4096, "end_value": 0, "bias": 0.5
        }]""")
        if use_cc_automation:
            automation_example.append("""
        "cc": [{
            "type": "curve", "cc": 74, "start_beat": 0.0, "end_beat": 4.0,
            "start_value": 60, "end_value": 127, "bias": 2.0
        }]""")
        
        automation_instructions += "    **JSON Structure for Note Automations:**\n"
        automation_instructions += "    ```json\n"
        automation_instructions += "    {\n"
        automation_instructions += "      \"pitch\": 60, ..., \n"
        automation_instructions += "      \"automations\": {\n"
        automation_instructions += "        " + ",\n        ".join(automation_example) + "\n"
        automation_instructions += "      }\n"
        automation_instructions += "    }\n"
        automation_instructions += "    ```\n"

        # Track-based
        automation_instructions += "2.  **Track-Based Automations (Independent of any note):**\n"
        automation_instructions += "    - Add a `\"track_automations\"` object at the top level of the JSON output, next to `\"notes\"`.\n"
        
        track_automation_example = []
        if use_pitch_bend:
            track_automation_example.append("""
        "pitch_bend": [{
            "type": "curve", "start_beat": 0.0, "end_beat": 16.0,
            "start_value": 0, "end_value": -8192, "bias": 1.0
        }]""")
        if use_cc_automation:
            track_automation_example.append("""
        "cc": [{
            "type": "curve", "cc": 11, "start_beat": 0.0, "end_beat": 16.0,
            "start_value": 80, "end_value": 127, "bias": 1.5
        }]""")

        automation_instructions += "    **JSON Structure for Track Automations:**\n"
        automation_instructions += "    ```json\n"
        automation_instructions += "    {\n"
        automation_instructions += "      \"notes\": [ ... ],\n"
        automation_instructions += "      \"track_automations\": {\n"
        automation_instructions += "        " + ",\n        ".join(track_automation_example) + "\n"
        automation_instructions += "      }\n"
        automation_instructions += "    }\n"
        automation_instructions += "    ```\n"

    # Sustain Pedal instructions
    if use_sustain_pedal:
        automation_instructions += "3.  **Sustain Pedal (CC 64):**\n"
        automation_instructions += "    - Add a `\"sustain_pedal\"` array at the top level of the JSON output.\n"
        automation_instructions += "    **JSON Structure for Sustain:**\n"
        automation_instructions += "    ```json\n"
        automation_instructions += "    {\n"
        automation_instructions += "      \"notes\": [ ... ],\n"
        automation_instructions += "      \"sustain_pedal\": [\n"
        automation_instructions += "        { \"beat\": 0.0, \"action\": \"down\" },\n"
        automation_instructions += "        { \"beat\": 15.5, \"action\": \"up\" }\n"
        automation_instructions += "      ]\n"
        automation_instructions += "    }\n"
        automation_instructions += "    ```\n"
    
    # General curve creation instructions (if any automation is on)
    if use_pitch_bend or use_cc_automation:
        automation_instructions += "**--- How to Create Different Automation Shapes ---**\n"
        automation_instructions += "You MUST use the `\"curve\"` object for ALL pitch and CC automations:\n"
        automation_instructions += "1.  **Linear Ramp:** Use `\"bias\": 1.0`.\n"
        automation_instructions += "2.  **Ease-In Curve (slow start, fast end):** Use `\"bias\" > 1.0` (e.g., 2.0).\n"
        automation_instructions += "3.  **Ease-Out Curve (fast start, slow end):** Use `\"bias\" < 1.0` (e.g., 0.5).\n"
        automation_instructions += "4.  **S-Curve (smooth ease-in and out):** Use `\"shape\": \"s_curve\"`.\n\n"
        
    # Rules and guidelines section
    if use_pitch_bend or use_cc_automation or use_sustain_pedal:
        automation_instructions += "**Automation Rules & Guidelines:**\n"
        if use_pitch_bend:
            automation_instructions += "- **Pitch Bend:** `value` ranges from -8192 (down) to 8191 (up).\n"
        if use_cc_automation:
            automation_instructions += f"- **Control Change (CC):** `value` is 0-127. Allowed CCs: {allowed_cc}.\n"
        if use_sustain_pedal:
             automation_instructions += "- **Sustain Pedal:** Use 'down' and 'up' actions. You MUST ensure that for every 'down' event, there is a corresponding 'up' event later in the part to release the pedal.\n"
        if use_pitch_bend or use_cc_automation:
            automation_instructions += "- **Timing:** Note-based automation is relative to the note's start. Track-based is relative to the section's start.\n"
        automation_instructions += "- **Automatic Reset:** For any temporary automation (like a single pitch bend or a short filter sweep), you MUST add a second automation event immediately following it to reset the value to its neutral state (e.g., pitch bend back to 0, modulation back to 0). This prevents automation values from getting 'stuck'.\n"


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
        f"{automation_instructions}"
        f"**--- UNIVERSAL PRINCIPLES OF GOOD MUSIC ---**\n"
        f"1. Reprise & Transform: Reprise the main motif in key parts with transformation (inversion, octave shift, rhythm augmentation).\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON object with a top-level key \"notes\". The \"notes\" array contains note objects with these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n'
        f'- **"automations"**: (Optional) An object containing automation data for this note.\n\n'
        f"Optional compact format for dense rhythms: You may include a 'pattern_blocks' array to represent many fast steps (e.g., 32nd/64th). Each block may have: 'length_bars', 'subdivision', 'bar_repeats', optional 'transpose' or 'octave_shift', and 'steps' with either 'mask' (e.g., '1010..') or 'indices' [0,2,4]. Ensure that expanded notes do not exceed the section length.\n\n"
        f"**IMPORTANT RULES:**\n"
        f'1.  **JSON OBJECT ONLY:** Your entire response MUST be only the raw JSON object, starting with "{" and ending with "}".\n'
        # ... (other rules remain the same)
    )

    # ... (The rest of the function for smart context calculation and final prompt assembly remains largely the same)
    
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
    safe_previous_themes, context_start_index = get_dynamic_context(previous_themes_full_history, character_budget=history_budget)

    # --- Part 3: Assemble the FINAL prompt ---
    previous_themes_prompt_part = ""
    if previous_themes_full_history: # Check if there was any history to begin with
        total_previous_themes = len(previous_themes_full_history)
        used_themes_count = len(safe_previous_themes)
        
        # Determine the source of limitation (window size vs character budget)
        cws_cfg = config.get("context_window_size", -1)
        limit_source = ""
        if cws_cfg > 0:
            # historical_source already reflects the window slice of up to cws_cfg items
            # Case A: fewer available than requested window size
            if total_previous_themes < cws_cfg:
                limit_source = f"limited by availability (only {total_previous_themes} previous themes exist, window={cws_cfg})"
            # Case B: window provided enough, but we still used fewer => char limit
            elif used_themes_count < total_previous_themes:
                limit_source = f"due to character limit (MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS})"
            # Case C: exactly the window size used
            else:
                limit_source = f"due to 'context_window_size={cws_cfg}'"
        elif cws_cfg == -1:
            if used_themes_count < total_previous_themes:
                limit_source = f"due to character limit (MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS})"
            else:
                limit_source = "using full available history"

        # Compute indices relative to the original full history length
        original_total_hist = len(previous_themes_full_history) if 'previous_themes_full_history' in locals() else total_previous_themes
        offset_into_original = original_total_hist - total_previous_themes
        # Correct original indices based on the current theme index and the size of the provided history slice
        # We have only a slice of the full history here; its original start index is:
        # (current_theme_index - len(previous_themes_full_history))
        original_slice_start = current_theme_index - total_previous_themes
        start_theme_num = original_slice_start + context_start_index + 1
        end_theme_num = start_theme_num + used_themes_count - 1
        if used_themes_count > 0:
            print(Fore.CYAN + f"Context Info: Using {used_themes_count}/{total_previous_themes} previous themes (original indices {start_theme_num}-{end_theme_num}) for context ({limit_source})." + Style.RESET_ALL)
            try:
                globals()['LAST_CONTEXT_COUNT'] = int(used_themes_count)
                globals()['PLANNED_CONTEXT_COUNT'] = int(used_themes_count)
            except Exception:
                pass

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
        f"{automation_instructions}"
        f"**--- UNIVERSAL PRINCIPLES OF GOOD MUSIC ---**\n"
        f"1. **Structure & Evolution:** Your composition should have a clear structure. A good musical part tells a story over the full {length} bars by introducing a core idea ('motif') and then developing it through variation, repetition, and contrast. Avoid mindless, robotic repetition.\n"
        f"2. **Clarity through Space:** Do not create a constant wall of sound. Use rests effectively. The musical role of a part determines how it should use space. Your role-specific instructions provide guidance on this.\n"
        f"3. **Dynamic Phrasing:** Use a wide range of velocity to create accents and shape the energy of the phrase. A static volume is boring and unnatural.\n"
        f"4.  **Tension & Release:** Build musical tension through dynamics, rhythmic complexity, or harmony, and resolve it at key moments (e.g., at the end of 4, 8, or 16 bar phrases) to create a satisfying arc.\n"
        f"5.  **Ensemble Playing:** Think like a member of a band. Your performance must complement the other parts. Pay attention to the phrasing of other instruments and find pockets of space to add your musical statement without cluttering the arrangement.\n"
        f"6.  **Micro-timing for Groove:** To add a human feel, you can subtly shift notes off the strict grid. Slightly anticipating a beat (pushing) can add urgency, while slightly delaying it (pulling) can create a more relaxed feel. This is especially effective for non-kick/snare elements.\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Generate the musical data as a single, valid JSON object with ONLY these top-level keys: `notes` (required), `track_automations` (optional), `sustain_pedal` (optional), `pattern_blocks` (optional). No other top-level keys. No prose, no markdown.\n\n"
        f"Each note object MUST have these keys:\n"
        f'- **"pitch"**: MIDI note number (integer 0-127).\n'
        f'- **"start_beat"**: The beat on which the note begins (float).\n'
        f'- **"duration_beats"**: The note\'s length in beats (float).\n'
        f'- **"velocity"**: MIDI velocity (integer 1-127).\n'
        f'- **"automations"**: (Optional) An object containing automation data for this note.\n\n'
        f"**IMPORTANT RULES:**\n"
        f'1.  **JSON OBJECT ONLY:** The entire response MUST be only the raw JSON object (no markdown, no commentary).\n'
        f"{polyphony_rule}\n"
        f"{stay_in_key_rule}"
        f"{timing_rule}"
        f'5.  **Valid JSON Syntax:** The output must be a perfectly valid JSON object.\n'
        f'6.  **Handling Silence:** If the instrument is intentionally silent for the entire section, return ONLY: `{{"notes": [{{"pitch": 0, "start_beat": 0, "duration_beats": 0, "velocity": 0}}]}}`. Do not return any other keys.\n'
        f'7.  **Density Cap:** Keep the total number of notes for this part reasonable (≤ 400) to avoid excessive density.\n'
        f'8.  **Sorting & Formatting:** Use dot decimals (e.g., 1.5), non-negative beats, and sort notes by `start_beat` ascending.\n\n'
        f"Now, generate the JSON object for the **{instrument_name}** track for the theme described as '{theme_description}'.\n"
    )
    
    return prompt


def generate_instrument_track_data(config: Dict, length: int, instrument_name: str, program_num: int, context_tracks: List[Dict], role: str, current_track_index: int, total_tracks: int, dialogue_role: str, theme_label: str, theme_description: str, previous_themes_full_history: List[Dict], current_theme_index: int) -> Tuple[Dict, int]:
    """
    Generates musical data for a single instrument track using the generative AI model, adapted for themes.
    Returns a tuple of (track_data, token_count).
    """
    global CURRENT_KEY_INDEX, SESSION_MODEL_OVERRIDE
    prompt = create_theme_prompt(config, length, instrument_name, program_num, context_tracks, role, current_track_index, total_tracks, dialogue_role, theme_label, theme_description, previous_themes_full_history, current_theme_index)
    # Local model override for this step (interactive switch on repeated failures)
    local_model_name = SESSION_MODEL_OVERRIDE or config["model_name"]
    try:
        origin = "session default" if SESSION_MODEL_OVERRIDE else "config default"
        print(Fore.CYAN + f"Model for this step: {local_model_name} ({origin}). Press 1/2/3 to switch for THIS step only; press 0 to set current as session default." + Style.RESET_ALL)
    except Exception:
        pass
    json_failure_count = 0
    failure_for_escalation_count = 0
    # MAX_TOKENS phase counters
    max_tokens_fail_flash = 0
    max_tokens_fail_pro = 0
    def _escalate_if_needed():
        nonlocal local_model_name, failure_for_escalation_count
        try:
            if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and failure_for_escalation_count >= AUTO_ESCALATE_THRESHOLD:
                local_model_name = 'gemini-2.5-pro'
                print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {failure_for_escalation_count} failures." + Style.RESET_ALL)
        except Exception:
            pass
    
    # Reset runtime hotkey guards/state for this step and show hint
    global PROMPTED_CUSTOM_THIS_STEP, REQUESTED_SWITCH_MODEL, REQUEST_SET_SESSION_DEFAULT, ABORT_CURRENT_STEP, DEFER_CURRENT_TRACK, REDUCE_CONTEXT_THIS_STEP, REDUCE_CONTEXT_HALVES, LAST_CONTEXT_COUNT
    PROMPTED_CUSTOM_THIS_STEP = False
    REQUESTED_SWITCH_MODEL = None
    REQUEST_SET_SESSION_DEFAULT = False
    ABORT_CURRENT_STEP = False
    DEFER_CURRENT_TRACK = False
    REDUCE_CONTEXT_THIS_STEP = False
    REDUCE_CONTEXT_HALVES = 0
    LAST_CONTEXT_COUNT = 0
    # Show hotkey hint once before we start attempts
    print_hotkey_hint(config, context=f"Generate: {instrument_name}")
    # Start background hotkey monitor once per process
    global HOTKEY_MONITOR_STARTED
    if not HOTKEY_MONITOR_STARTED:
        try:
            t = threading.Thread(target=_hotkey_monitor_loop, args=(config,), daemon=True)
            t.start(); HOTKEY_MONITOR_STARTED = True
        except Exception:
            HOTKEY_MONITOR_STARTED = True
    
    # Track full failure cycles for this track (for degrade/deferral control)
    full_failure_count = 0
    while True: # Loop to allow retrying after complete failure
        max_retries = 3
        start_key_index = CURRENT_KEY_INDEX
        keys_have_rotated_fully = False
        quota_rotation_count = 0  # Counts full key-rotation exhaustion cycles to drive long backoff
        attempt_count = 0

        # Non-blocking model switch during waits (Windows only)
        def _wait_with_optional_switch(wait_time: float):
            nonlocal local_model_name
            try:
                # Offer one-key model switch without blocking
                if sys.platform == "win32":
                    print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (config.custom_model_name) to switch model for THIS track; waiting..." + Style.RESET_ALL)
                    end_t = time.time() + max(0.0, wait_time)
                    while time.time() < end_t:
                        if msvcrt.kbhit():
                            ch = msvcrt.getch().decode().lower()
                            if ch == '1':
                                local_model_name = 'gemini-2.5-pro'; print(Fore.YELLOW + "Switching to gemini-2.5-pro (this track)." + Style.RESET_ALL); break
                            if ch == '2':
                                local_model_name = 'gemini-2.5-flash'; print(Fore.YELLOW + "Switching to gemini-2.5-flash (this track)." + Style.RESET_ALL); break
                            if ch == '3':
                                custom = config.get('custom_model_name')
                                if custom:
                                    local_model_name = custom; print(Fore.YELLOW + f"Switching to {custom} (this track)." + Style.RESET_ALL); break
                        time.sleep(0.25)
                else:
                    time.sleep(wait_time)
            except Exception:
                time.sleep(wait_time)
        while attempt_count < max_retries:
            try:
                print(Fore.BLUE + f"Attempt {attempt_count + 1}/{max_retries}: Generating part for {instrument_name} ({role})..." + Style.RESET_ALL)
                
                generation_config = {
                    "temperature": config["temperature"],
                    "response_mime_type": "application/json",
                    "max_output_tokens": config.get("max_output_tokens", 8192)
                }

                # Poll persistent hotkey (Windows) to change model instantly
                local_model_name = _poll_model_switch(local_model_name, config)
                model = genai.GenerativeModel(
                    model_name=local_model_name,
                    generation_config=generation_config
                )

                # Define safety settings to avoid blocking responses unnecessarily
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]

                # Apply automation-only degrade after 4 full failures
                effective_prompt = prompt
                if full_failure_count >= 4:
                    effective_prompt = re.sub(r"\*\*\-\-\- Advanced MIDI Automation[\s\S]*?\n\n", "", effective_prompt)
                # Add strict output hint
                effective_prompt += "\nOutput: Return a single JSON object with a 'notes' key only; no prose.\n"

                # Asynchronous poll for hotkeys while waiting: short slices with checks
                # We chunk one long call into a loop that polls and bails fast if a switch is requested.
                # Gemini SDK doesn't expose a non-blocking call; we simulate cooperative checks around the call sites.
                if ABORT_CURRENT_STEP or DEFER_CURRENT_TRACK:
                    # Apply requested switch immediately before issuing the call
                    if REQUESTED_SWITCH_MODEL:
                        local_model_name = REQUESTED_SWITCH_MODEL
                        model = genai.GenerativeModel(
                            model_name=local_model_name,
                            generation_config=generation_config
                        )
                    if REQUEST_SET_SESSION_DEFAULT:
                        SESSION_MODEL_OVERRIDE = local_model_name
                        REQUEST_SET_SESSION_DEFAULT = False
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + "Deferring current track on user request (hotkey 'd')." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        return None, 0
                    ABORT_CURRENT_STEP = False
                # Apply on-demand context halving (hotkey 'h')
                if REDUCE_CONTEXT_THIS_STEP:
                    try:
                        reduced_cfg = json.loads(json.dumps(config))
                        prev_ctx = previous_themes_full_history
                        if isinstance(prev_ctx, list) and prev_ctx:
                            original = len(prev_ctx)
                            LAST_CONTEXT_COUNT = original
                            halves = max(1, int(REDUCE_CONTEXT_HALVES))
                            target = original
                            for _ in range(halves):
                                target = max(1, target // 2)
                            reduced_cfg["context_window_size"] = target
                            print(Fore.CYAN + f"Applying halve context (x{halves}) for this step: {original} → {target} parts." + Style.RESET_ALL)
                            print(Fore.CYAN + f"Context in use now: {target} previous theme(s)." + Style.RESET_ALL)
                            prompt = create_theme_prompt(reduced_cfg, length, instrument_name, program_num, context_tracks, role, current_track_index, total_tracks, dialogue_role, theme_label, theme_description, previous_themes_full_history[-target:], current_theme_index)
                    except Exception:
                        pass
                    REDUCE_CONTEXT_THIS_STEP = False
                    REDUCE_CONTEXT_HALVES = 0
                response = model.generate_content(effective_prompt, safety_settings=safety_settings)
                
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

                    print(Fore.RED + f"Error on attempt {attempt_count + 1}: Generation failed or was incomplete." + Style.RESET_ALL)
                    print(Fore.YELLOW + f"Reason: {finish_reason_name}" + Style.RESET_ALL)
                    
                    # Also check for safety blocking information
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                         print(Fore.YELLOW + f"Block Reason: {response.prompt_feedback.block_reason.name}" + Style.RESET_ALL)
                    # Non-counting errors: MAX_TOKENS (but tracked for phase/deferral)
                    if str(finish_reason_name).upper().find("MAX_TOKENS") != -1:
                        if 'flash' in (local_model_name or ''):
                            max_tokens_fail_flash += 1
                            if max_tokens_fail_flash >= 6:
                                if AUTO_ESCALATE_TO_PRO:
                                    local_model_name = 'gemini-2.5-pro'
                                    max_tokens_fail_pro = 0
                                    print(Fore.CYAN + "Auto-escalate after 6 MAX_TOKENS on flash → switching to pro for this track." + Style.RESET_ALL)
                                else:
                                    print(Fore.YELLOW + "Deferring this track due to repeated MAX_TOKENS on flash (6 attempts)." + Style.RESET_ALL)
                                    return None, 0
                        elif 'pro' in (local_model_name or ''):
                            max_tokens_fail_pro += 1
                            if max_tokens_fail_pro >= 6:
                                print(Fore.YELLOW + "Deferring this track due to repeated MAX_TOKENS on pro (6 attempts)." + Style.RESET_ALL)
                                return None, 0
                        print(Fore.YELLOW + "Not counting this attempt due to MAX_TOKENS. Retrying..." + Style.RESET_ALL)
                        failure_for_escalation_count += 1
                        _escalate_if_needed()
                        # After two MAX_TOKENS, reduce historical context by half and retry
                        try:
                            if (('flash' in (local_model_name or '')) and max_tokens_fail_flash >= 2) or (('pro' in (local_model_name or '')) and max_tokens_fail_pro >= 2):
                                reduced_cfg = json.loads(json.dumps(config))
                                prev_ctx = previous_themes_full_history
                                if isinstance(prev_ctx, list) and prev_ctx:
                                    half = max(1, len(prev_ctx)//2)
                                    reduced_cfg["context_window_size"] = half
                                    print(Fore.CYAN + f"Reducing historical context window to {half} due to repeated MAX_TOKENS." + Style.RESET_ALL)
                                    # Replace prompt with smaller context on next loop
                                    prompt = create_theme_prompt(reduced_cfg, length, instrument_name, program_num, context_tracks, role, current_track_index, total_tracks, dialogue_role, theme_label, theme_description, previous_themes_full_history[-half:], current_theme_index)
                        except Exception:
                            pass
                        # Offer a quick, non-blocking model switch for this track
                        _wait_with_optional_switch(3)
                        continue
                    # Safety blocks and other content-related reasons count as attempts
                    attempt_count += 1
                    # Apply backoff for countable errors below
                    base = 3
                    wait_time = base * (2 ** max(0, attempt_count - 1))
                    jitter = random.uniform(0, 1.5)
                    wait_time = min(30, wait_time + jitter)
                    print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                    _interruptible_backoff(wait_time, config, context_label=f"{instrument_name}")
                    continue

                # 2. Now that we know the response is valid, safely access the text and parse JSON
                # Check if user requested a switch during generation window
                if ABORT_CURRENT_STEP or DEFER_CURRENT_TRACK:
                    # Apply requested switch and restart this attempt cleanly
                    if REQUESTED_SWITCH_MODEL:
                        local_model_name = REQUESTED_SWITCH_MODEL
                    if REQUEST_SET_SESSION_DEFAULT:
                        SESSION_MODEL_OVERRIDE = local_model_name
                        REQUEST_SET_SESSION_DEFAULT = False
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + "Deferring current track on user request (hotkey 'd')." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        return None, 0
                    print(Fore.CYAN + f"Restarting step with model: {local_model_name}" + Style.RESET_ALL)
                    ABORT_CURRENT_STEP = False
                    continue

                response_text = _extract_text_from_response(response)
                if not response_text:
                    print(Fore.YELLOW + f"Warning on attempt {attempt_count + 1}: Empty or invalid response payload for {instrument_name}." + Style.RESET_ALL)
                    # Counts as content-related error
                    attempt_count += 1
                    base = 3
                    wait_time = base * (2 ** max(0, attempt_count - 1))
                    jitter = random.uniform(0, 1.5)
                    wait_time = min(30, wait_time + jitter)
                    print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                    _interruptible_backoff(wait_time, config, context_label=f"{instrument_name}")
                    continue

                # --- NEW: Token Usage Reporting ---
                total_token_count = 0
                if hasattr(response, 'usage_metadata'):
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    total_token_count = response.usage_metadata.total_token_count
                    char_per_token = len(prompt) / prompt_tokens if prompt_tokens > 0 else 0
                    print(Fore.CYAN + f"Token Usage: Prompt: {prompt_tokens:,} | Output: {output_tokens:,} | Total: {total_token_count:,} "
                                      f"(Prompt: {len(prompt):,} chars, ~{char_per_token:.2f} chars/token)" + Style.RESET_ALL)
                
                if not response_text.strip():
                    print(Fore.YELLOW + f"Warning on attempt {attempt_count + 1}: Model returned an empty response for {instrument_name}." + Style.RESET_ALL)
                    # Counts as content-related error
                    failure_for_escalation_count += 1
                    _escalate_if_needed()
                    attempt_count += 1
                    base = 3
                    wait_time = base * (2 ** max(0, attempt_count - 1))
                    jitter = random.uniform(0, 1.5)
                    wait_time = min(30, wait_time + jitter)
                    print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                    time.sleep(wait_time)
                    continue

                # 3. Parse the JSON response
                # The model should return a JSON object with a "notes" key and an optional "sustain_pedal" key
                json_payload = _extract_json_object(response_text)
                if not json_payload:
                    json_failure_count += 1
                    print(Fore.YELLOW + f"Warning on attempt {attempt_count + 1}: Could not extract JSON object for {instrument_name}." + Style.RESET_ALL)
                    failure_for_escalation_count += 1
                    _escalate_if_needed()
                    # Offer one-time model switch for this track after repeated JSON failures
                    if json_failure_count == 2:
                        try:
                            print(Fore.CYAN + "\nModel appears to struggle with strict JSON. Switch model for THIS track only?" + Style.RESET_ALL)
                            print("  1) gemini-2.5-pro\n  2) gemini-2.5-flash\n  3) custom\n  4) keep current")
                            sel = input(Fore.GREEN + "> " + Style.RESET_ALL).strip()
                            if sel == '1':
                                local_model_name = 'gemini-2.5-pro'
                                attempt_count = 0
                                continue
                            elif sel == '2':
                                local_model_name = 'gemini-2.5-flash'
                                attempt_count = 0
                                continue
                            elif sel == '3':
                                custom = input(Fore.GREEN + "Enter model name: " + Style.RESET_ALL).strip()
                                if custom:
                                    local_model_name = custom
                                    attempt_count = 0
                                    continue
                        except Exception:
                            pass
                    # Counts as content-related error
                    attempt_count += 1
                    base = 3
                    wait_time = base * (2 ** max(0, attempt_count - 1))
                    jitter = random.uniform(0, 1.5)
                    wait_time = min(30, wait_time + jitter)
                    print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                    time.sleep(wait_time)
                    continue
                response_data = json.loads(json_payload)

                if isinstance(response_data, dict) and "notes" not in response_data and isinstance(response_data.get("pattern_blocks"), list):
                    # Accept compact format without explicit notes array
                    response_data["notes"] = []
                if not isinstance(response_data, dict) or "notes" not in response_data:
                    raise TypeError("The generated data is not a valid JSON object with a 'notes' key.")

                notes_list = response_data["notes"]
                # Optional compact pattern support
                try:
                    pattern_blocks = response_data.get("pattern_blocks")
                    if isinstance(pattern_blocks, list):
                        expanded = _expand_pattern_blocks(pattern_blocks, length, config["time_signature"]["beats_per_bar"])
                        if expanded:
                            if isinstance(notes_list, list):
                                notes_list = notes_list + expanded
                            else:
                                notes_list = expanded
                except Exception:
                    pass
                sustain_events = response_data.get("sustain_pedal", []) # Get sustain events or an empty list

                # --- NEW: Check for special silence signal ---
                if isinstance(notes_list, list) and len(notes_list) == 1:
                    note = notes_list[0]
                    if note.get("pitch") == 0 and note.get("start_beat") == 0 and note.get("duration_beats") == 0 and note.get("velocity") == 0:
                        print(Fore.GREEN + f"Recognized intentional silence for {instrument_name}. The track will be empty." + Style.RESET_ALL)
                        return {
                            "instrument_name": instrument_name,
                            "program_num": program_num,
                            "role": role,
                            "notes": [], # Return a valid track with no notes
                            "sustain_pedal": []
                        }, total_token_count

                if not isinstance(notes_list, list):
                    raise TypeError("The 'notes' field is not a valid list.")
                if not isinstance(sustain_events, list):
                    raise TypeError("The 'sustain_pedal' field is not a valid list.")

                # --- Data Validation ---
                validated_notes = []
                for note in notes_list:
                    if not all(k in note for k in ["pitch", "start_beat", "duration_beats", "velocity"]):
                         print(Fore.YELLOW + f"Warning: Skipping invalid note object: {note}" + Style.RESET_ALL)
                         continue
                    validated_notes.append(note)
                
                # --- Sustain Validation ---
                validated_sustain = []
                for event in sustain_events:
                    if not all(k in event for k in ["beat", "action"]):
                        print(Fore.YELLOW + f"Warning: Skipping invalid sustain event: {event}" + Style.RESET_ALL)
                        continue
                    validated_sustain.append(event)


                print(Fore.GREEN + f"Successfully generated part for {instrument_name}." + Style.RESET_ALL)
                return {
                    "instrument_name": instrument_name,
                    "program_num": program_num,
                    "role": role,
                    "notes": validated_notes,
                    "sustain_pedal": validated_sustain
                }, total_token_count

            except (json.JSONDecodeError, TypeError) as e:
                print(Fore.YELLOW + f"Warning on attempt {attempt_count + 1}: Data validation failed for {instrument_name}. Reason: {str(e)}" + Style.RESET_ALL)
                # We already checked for blocking, so we just show the text if parsing fails.
                if "response_text" in locals():
                    print(Fore.YELLOW + "Model response was:\n" + response_text + Style.RESET_ALL)
                # Counts as content-related error
                json_failure_count += 1
                failure_for_escalation_count += 1
                _escalate_if_needed()
                attempt_count += 1
                base = 3
                wait_time = base * (2 ** max(0, attempt_count - 1))
                jitter = random.uniform(0, 1.5)
                wait_time = min(30, wait_time + jitter)
                print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                _wait_with_optional_switch(wait_time)
                continue
            except Exception as e:
                error_message = str(e).lower()
                if "429" in error_message and "quota" in error_message:
                    qtype = _classify_quota_error(error_message)
                    print(Fore.YELLOW + f"Warning on attempt {attempt_count + 1}: API quota exceeded for key #{CURRENT_KEY_INDEX + 1} ({qtype})." + Style.RESET_ALL)
                    # Apply cooldown based on detected window
                    # For daily quotas: retry hourly to probe reset windows
                    if qtype == "per-day":
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                    elif qtype == "per-hour":
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                    else:  # per-minute, rate-limit, unknown
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_MINUTE_COOLDOWN_SECONDS)
                    
                    if len(API_KEYS) > 1 and not keys_have_rotated_fully:
                        nxt = _next_available_key()
                        if nxt is not None:
                            CURRENT_KEY_INDEX = nxt
                            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                            print(Fore.CYAN + f"Retrying immediately with available key #{CURRENT_KEY_INDEX + 1}...")
                            continue
                        else:
                            print(Fore.RED + "All available API keys are cooling down or have exceeded their quota.")
                            keys_have_rotated_fully = True
                    elif len(API_KEYS) <= 1:
                        print(Fore.RED + "The only available API key has exceeded its quota.")
                    # 429 does not count as an attempt; after all keys are exhausted: long backoff up to 1h
                    quota_rotation_count += 1
                    base = 3
                    # if all keys cooling down and we have per-minute, show countdown to first available
                    if _all_keys_cooling_down():
                        wait_time = max(5.0, _seconds_until_first_available())
                        print(Fore.CYAN + f"All keys cooling down. Next available in ~{wait_time:.1f}s." + Style.RESET_ALL)
                    else:
                        wait_time = base * (2 ** max(0, quota_rotation_count - 1))
                    jitter = random.uniform(0, 5.0)
                    wait_time = min(3600, wait_time + jitter)
                    print(Fore.YELLOW + f"All available API keys exhausted. Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                    _wait_with_optional_switch(wait_time)
                    continue
                # 5xx/Timeouts/Deadline: do not count attempts, but consider for auto-escalation
                if any(code in error_message for code in [" 500", " 502", " 503", " 504"]) or "deadline" in error_message or "timeout" in error_message:
                    print(Fore.YELLOW + f"Non-counting server error on attempt {attempt_count + 1}: {str(e)}" + Style.RESET_ALL)
                    failure_for_escalation_count += 1
                    _escalate_if_needed()
                    continue
                else:
                    print(Fore.RED + f"An unexpected error occurred on attempt {attempt_count + 1}: {str(e)}" + Style.RESET_ALL)
                    # Unexpected errors count
                    attempt_count += 1
                    base = 3
                    wait_time = base * (2 ** max(0, attempt_count - 1))
                    jitter = random.uniform(0, 1.5)
                    wait_time = min(30, wait_time + jitter)
                    print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                    time.sleep(wait_time)
                    continue

            # No additional backoff/count here; cases handled above
        
        # This part is reached after all retries fail.
        full_failure_count += 1
        if full_failure_count >= 10:
            print(Fore.RED + f"Giving up for now on {instrument_name} after {full_failure_count} full failure cycles; deferring." + Style.RESET_ALL)
            return None, 0
        print(Fore.RED + f"Failed to generate a valid part for {instrument_name} after {max_retries} attempts." + Style.RESET_ALL)
        
        print(Fore.CYAN + "Automatic retry in 3 seconds..." + Style.RESET_ALL)
        print(Fore.YELLOW + "Press 'y' to retry now, 'n' to cancel, or 'w' to pause the timer." + Style.RESET_ALL)

        user_action = None
        # Countdown loop
        for i in range(3, 0, -1):
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
            return None, 0 # Exit function
            
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
                    return None, 0
                else:
                    print(Fore.YELLOW + "Invalid input. Please enter 'y' or 'n'." + Style.RESET_ALL)


def create_song_optimization(config: Dict, theme_length: int, themes_to_optimize: List[Dict], script_dir: str, opt_iteration_num: int, run_timestamp: str, user_optimization_prompt: str = "", resume_data=None) -> List[Dict]:
    """
    Orchestrates the optimization of a whole song's themes.
    It iterates through each track of each theme, generates an optimization,
    and replaces the original track data with the new optimized data.
    """
    total_optimization_tokens = 0
    optimized_themes = themes_to_optimize[:]  # Create a mutable copy

    # The effective prompt to be used for generation, may be overridden by resume_data
    effective_prompt = user_optimization_prompt

    start_theme_index = 0
    start_track_index = 0

    if resume_data:
        print(f"{Fore.YELLOW}Resuming optimization based on progress file...{Style.RESET_ALL}")
        start_theme_index = resume_data.get('current_theme_index', 0)
        # CRITICAL FIX: The saved index is the one that was IN PROGRESS.
        # We start directly at this index to retry it.
        start_track_index = resume_data.get('current_track_index', 0)
        total_optimization_tokens = resume_data.get('total_optimization_tokens', 0)

        # CRITICAL: When resuming, ALWAYS use the prompt that was saved with the progress.
        # This ensures the optimization continues with the original user intent.
        if 'user_optimization_prompt' in resume_data:
            effective_prompt = resume_data['user_optimization_prompt']
            print(f"Using saved optimization prompt: '{effective_prompt}'")
        # Restore deferred queue for the theme we resume in (if present)
        resume_deferred_tracks = resume_data.get('deferred_tracks', []) or []
    
    print() # Spacer for readability

    try:
                                             # Process themes one by one; each theme has its own deferred-retry queue
        for theme_index in range(start_theme_index, len(optimized_themes)):
            theme_to_optimize = optimized_themes[theme_index]
            theme_label = theme_to_optimize.get('label', f"Theme {chr(65+theme_index)}")
            print(f"\n--- Optimizing Theme {theme_index + 1}/{len(optimized_themes)}: '{Fore.CYAN}{theme_label}{Style.RESET_ALL}' ---")
            
            tracks = theme_to_optimize.get('tracks', [])
            optimized_tracks_for_theme = tracks[:] # Start with a copy
            theme_deferred_tracks = []  # (track_index, failure_count)
            # If resuming inside this theme, restore its deferred queue
            try:
                if 'resume_deferred_tracks' in locals() and theme_index == start_theme_index and resume_deferred_tracks:
                    theme_deferred_tracks = list(resume_deferred_tracks)
                    print(Fore.CYAN + f"Restored {len(theme_deferred_tracks)} deferred track(s) for this theme from resume." + Style.RESET_ALL)
            except Exception:
                pass
            
            # Define the historical context for the AI (all previously processed themes)
            historical_context = optimized_themes[:theme_index]
            
            # When we resume, we need to reset the track_index for the *next* theme
            current_loop_start_track = start_track_index if theme_index == start_theme_index else 0
            
            for track_index in range(current_loop_start_track, len(tracks)):
                track_to_optimize = tracks[track_index]
                instrument_name = get_instrument_name(track_to_optimize)
                role = track_to_optimize.get('role', 'Unknown Role')

                # Preserve intentionally silent tracks
                if not track_to_optimize.get('notes'):
                    print(f"\n{Fore.YELLOW}--- Skipping Track: {Style.BRIGHT}{Fore.GREEN}{instrument_name}{Style.NORMAL} (Preserving intentional silence) ---{Style.RESET_ALL}")
                    continue

                print(f"\n--- Optimizing Track: {Fore.GREEN}{instrument_name}{Style.RESET_ALL} (Role: {role}) ---")

                # Define the inner-theme context for the AI (all other tracks in the current theme)
                inner_context = [t for idx, t in enumerate(optimized_tracks_for_theme) if idx != track_index]

                # Generate optimization
                (optimized_track_data, tokens_used) = generate_optimization_data(
                            config,
                            theme_length,
                            track_to_optimize,
                            role,
                            theme_label,
                            theme_to_optimize.get('description', ''),
                            historical_context,
                            inner_context,
                            theme_index,
                            user_optimization_prompt=effective_prompt
                )

                total_optimization_tokens += tokens_used

                if optimized_track_data:
                    # Ensure we keep essential track metadata while applying optimized musical data
                    base_track = tracks[track_index]
                    merged_track = {
                        "instrument_name": get_instrument_name(base_track),
                        "program_num": base_track.get("program_num", 0),
                        "role": base_track.get("role", "complementary"),
                        # Always accept the newly optimized musical content
                        "notes": optimized_track_data.get("notes", base_track.get("notes", []))
                    }
                    # Optional top-level automations from optimized data
                    if "sustain_pedal" in optimized_track_data:
                        merged_track["sustain_pedal"] = optimized_track_data.get("sustain_pedal", [])
                    # Normalize/merge track-level automations
                    track_autos = optimized_track_data.get("track_automations", {})
                    # Backward-compat: accept simple top-level 'cc' list as track automation
                    if "cc" in optimized_track_data and "cc" not in track_autos:
                        track_autos["cc"] = optimized_track_data.get("cc", [])
                    if track_autos:
                        merged_track["track_automations"] = track_autos

                    if track_index < len(optimized_tracks_for_theme):
                        optimized_tracks_for_theme[track_index] = merged_track
                    else:
                        optimized_tracks_for_theme.append(merged_track)

                    # Update the theme data
                    theme_to_optimize['tracks'] = optimized_tracks_for_theme

                    # Save progress
                    progress_data_to_save = {
                        'type': 'optimization', 'config': config, 'theme_length': theme_length,
                        'themes_to_optimize': optimized_themes, 'opt_iteration_num': opt_iteration_num,
                        'final_optimized_themes': optimized_themes, 'current_theme_index': theme_index,
                        'current_track_index': track_index + 1,
                        'user_optimization_prompt': effective_prompt,
                        'total_optimization_tokens': total_optimization_tokens,
                        # For compatibility with generation progress readers
                        'total_tokens_used': total_optimization_tokens,
                        'theme_length': theme_length,
                        'theme_definitions': [t.get('definition', {}) for t in optimized_themes], # Save definitions for resume
                        'opt_iteration_num': opt_iteration_num,
                        'user_optimization_prompt': effective_prompt,
                        'last_generated_song_basename': get_progress_filename(config, run_timestamp) 
                    }
                    save_progress(progress_data_to_save, script_dir, run_timestamp)
                    print(Fore.CYAN + f"Cumulative optimization tokens so far: {total_optimization_tokens:,}" + Style.RESET_ALL)

                else:
                    # Track failed (e.g., MAX_TOKENS deferral): push to deferred queue for this theme
                    theme_deferred_tracks.append([track_index, 1])
                    # Persist progress including current deferred queue
                    try:
                        progress_data_to_save = {
                            'type': 'optimization', 'config': config, 'theme_length': theme_length,
                            'themes_to_optimize': optimized_themes, 'opt_iteration_num': opt_iteration_num,
                            'final_optimized_themes': optimized_themes, 'current_theme_index': theme_index,
                            'current_track_index': track_index + 1,
                            'user_optimization_prompt': effective_prompt,
                            'total_optimization_tokens': total_optimization_tokens,
                            'total_tokens_used': total_optimization_tokens,
                            'deferred_tracks': theme_deferred_tracks,
                            'timestamp': run_timestamp
                        }
                        save_progress(progress_data_to_save, script_dir, run_timestamp)
                    except Exception:
                        pass
            # Retry deferred tracks for this theme until success or cap
            round_counter = 0
            while theme_deferred_tracks and round_counter < 10:
                round_counter += 1
                track_index, fail_count = theme_deferred_tracks.pop(0)
                if track_index >= len(tracks):
                    continue
                track_to_optimize = tracks[track_index]
                instrument_name = get_instrument_name(track_to_optimize)
                role = track_to_optimize.get('role', 'Unknown Role')
                print(Fore.MAGENTA + f"\nRetry deferred optimization (round {round_counter}): {instrument_name} in theme {theme_index+1}" + Style.RESET_ALL)
                inner_context = [t for idx, t in enumerate(tracks) if idx != track_index]
                # Halve historical theme context size each round for optimization as well
                try:
                    cws_base = theme_index  # number of previous themes available
                    cws_override = max(1, cws_base // (2 ** max(0, round_counter-1))) if cws_base > 0 else 0
                    temp_cfg = json.loads(json.dumps(config))
                    if cws_override > 0:
                        temp_cfg["context_window_size"] = cws_override
                    else:
                        temp_cfg["context_window_size"] = 0
                    prev_ctx_decayed = optimized_themes[:theme_index][-cws_override:] if cws_override > 0 else []
                except Exception:
                    temp_cfg = config
                    prev_ctx_decayed = optimized_themes[:theme_index]
                (optimized_track_data, tokens_used) = generate_optimization_data(
                    temp_cfg, theme_length, track_to_optimize, role, theme_to_optimize.get('label', f"Theme {theme_index+1}"),
                    theme_to_optimize.get('description', ''), prev_ctx_decayed, inner_context, theme_index,
                    user_optimization_prompt=effective_prompt
                )
                total_optimization_tokens += tokens_used
                if optimized_track_data:
                    base_track = tracks[track_index]
                    merged_track = {
                        "instrument_name": get_instrument_name(base_track),
                        "program_num": base_track.get("program_num", 0),
                        "role": base_track.get("role", "complementary"),
                        "notes": optimized_track_data.get("notes", base_track.get("notes", []))
                    }
                    if "sustain_pedal" in optimized_track_data:
                        merged_track["sustain_pedal"] = optimized_track_data.get("sustain_pedal", [])
                    track_autos = optimized_track_data.get("track_automations", {})
                    if "cc" in optimized_track_data and "cc" not in track_autos:
                        track_autos["cc"] = optimized_track_data.get("cc", [])
                    if track_autos:
                        merged_track["track_automations"] = track_autos
                    theme_to_optimize['tracks'][track_index] = merged_track
                else:
                    fail_count += 1
                    if fail_count < 10:
                        theme_deferred_tracks.append([track_index, fail_count])
                    else:
                        print(Fore.RED + f"Giving up on track '{instrument_name}' in theme {theme_index+1} after 10 failure cycles." + Style.RESET_ALL)
                print(Fore.CYAN + f"Cumulative optimization tokens so far: {total_optimization_tokens:,}" + Style.RESET_ALL)

            # After finishing retries for this theme, clear deferred queue in saved progress
            try:
                progress_data_to_save = {
                    'type': 'optimization', 'config': config, 'theme_length': theme_length,
                    'themes_to_optimize': optimized_themes, 'opt_iteration_num': opt_iteration_num,
                    'final_optimized_themes': optimized_themes, 'current_theme_index': theme_index,
                    'current_track_index': 0,
                    'user_optimization_prompt': effective_prompt,
                    'total_optimization_tokens': total_optimization_tokens,
                    'total_tokens_used': total_optimization_tokens,
                    'deferred_tracks': [],
                    'timestamp': run_timestamp
                }
                save_progress(progress_data_to_save, script_dir, run_timestamp)
            except Exception:
                pass

            # Export this theme's final part MIDI now that all its retries are done
            try:
                theme_part_data = {"tracks": theme_to_optimize.get('tracks', [])}
                base_part_path = generate_filename(config, script_dir, theme_length, theme_label, theme_index, run_timestamp)
                if base_part_path.lower().endswith('.mid'):
                    part_out_path = re.sub(r"\.mid$", f"_opt_{opt_iteration_num}.mid", base_part_path, flags=re.IGNORECASE)
                else:
                    part_out_path = f"{base_part_path}_opt_{opt_iteration_num}.mid"
                # Clamp exact section length at write time
                part_len_beats = theme_length * config["time_signature"]["beats_per_bar"]
                create_part_midi_from_theme(theme_part_data, config, part_out_path, time_offset_beats=0, section_length_beats=part_len_beats)
            except Exception as e:
                print(Fore.YELLOW + f"Warning: Could not create optimized part MIDI for '{theme_label}': {e}" + Style.RESET_ALL)

        # After all themes and tracks are processed, print accurate total
        final_token_count_str = f"Total tokens used for this optimization run: {total_optimization_tokens:,}"
        print(Fore.GREEN + final_token_count_str + Style.RESET_ALL)
        # Also persist the final number one more time for resume readers
        try:
            save_progress({
                'type': 'optimization', 'config': config, 'theme_length': theme_length,
                'themes_to_optimize': optimized_themes, 'opt_iteration_num': opt_iteration_num,
                'current_theme_index': len(optimized_themes)-1, 'current_track_index': 0,
                'user_optimization_prompt': effective_prompt,
                'total_optimization_tokens': total_optimization_tokens,
                'total_tokens_used': total_optimization_tokens,
                'timestamp': run_timestamp
            }, script_dir, run_timestamp)
        except Exception:
            pass

        # Create the final optimized MIDI file (always)
        try:
            final_base = build_final_song_basename(config, optimized_themes, run_timestamp, resumed=True, opt_iteration=opt_iteration_num)
            create_midi_from_json(merge_themes_to_song_data(optimized_themes, config, theme_length), config, os.path.join(script_dir, f"{final_base}.mid"))
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Could not create final optimized MIDI: {e}" + Style.RESET_ALL)

        # Return the optimized themes
        return optimized_themes
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n--- Optimization interrupted by user. Progress has been saved. ---" + Style.RESET_ALL)
        print("You can resume this job from the main menu.")
        return None
    except Exception as e:
        print(Fore.RED + f"\n\n--- An unexpected error occurred during optimization: {e} ---" + Style.RESET_ALL)
        print("Progress has been saved. You can try to resume this job from the main menu.")
        import traceback
        traceback.print_exc()
        return None

def create_optimization_prompt(config: Dict, length: int, track_to_optimize: Dict | None, role: str, theme_label: str, theme_detailed_description: str, historical_themes_context: List[Dict], inner_context_tracks: List[Dict],   current_theme_index: int, user_optimization_prompt: str = "") -> str:
    """
    Creates a prompt for optimizing a single track within a themed song structure.
    It now normalizes the context themes' timestamps to be relative.
    """
    # Defensive defaults
    if track_to_optimize is None or not isinstance(track_to_optimize, dict):
        track_to_optimize = {"instrument_name": "track", "program_num": 0, "role": "complementary", "notes": []}
    user_optimization_prompt = user_optimization_prompt or ""
    scale_notes = get_scale_notes(config["root_note"], config["scale_type"])
    
    # --- CRITICAL: Compress the JSON for the prompt to save space ---
    original_part_str = json.dumps(track_to_optimize, separators=(',', ':'))
    
    basic_instructions = (
        f"**Genre:** {config['genre']}\n"
        f"**Tempo:** {config['bpm']} BPM\n"
        f"**Time Signature:** {config['time_signature']['beats_per_bar']}/{config['time_signature']['beat_value']}\n"
        f"**Key/Scale:** {config['key_scale'].title()} (Available notes: {scale_notes})\n"
        f"**Instrument:** {get_instrument_name(track_to_optimize)} (MIDI Program: {track_to_optimize.get('program_num', 0)})\n"
    )

    # --- Context from tracks WITHIN the CURRENT theme ---
    inner_context_prompt_part = ""
    if inner_context_tracks:
        try:
            notes_cap = int(config.get('_notes_context_cap', MAX_NOTES_IN_CONTEXT))
        except Exception:
            notes_cap = MAX_NOTES_IN_CONTEXT
        inner_context_prompt_part = "**Context from the Current Song Section:**\nWithin this section, you have already optimized these parts. Make your new part fit perfectly with them.\n"
        for track in inner_context_tracks:
            try:
                notes = track.get('notes', [])
                if isinstance(notes, list) and len(notes) > max(1, notes_cap):
                    head = notes[:max(1, notes_cap)//2]
                    tail = notes[-max(1, notes_cap)//2:]
                    notes = head + tail
            except Exception:
                notes = track.get('notes', [])
            notes_as_str = json.dumps(notes, separators=(',', ':'))
            inner_context_prompt_part += f"- **{get_instrument_name(track)}** (Role: {track['role']}):\n```json\n{notes_as_str}\n```\n"
        inner_context_prompt_part += "\n"

    original_part_prompt = (
        "**This is the original part you need to optimize:**\n"
        f"```json\n{original_part_str}\n```\n"
    )

    optimization_instruction = (
        "**Your Task: Make Minimal, Targeted Improvements (No Rewrite)**\n"
        f"You are working on the section **'{theme_label}'**. Your job is to **preserve the identity** of the original part and apply **small, surgical fixes only**.\n"
        "- Fix obvious problems (timing clashes, muddiness, unplayable overlaps per role rules).\n"
        "- Improve feel and clarity with tiny changes (velocities, micro‑timing, short fills), but **do not create new motifs or rewrite phrases**.\n"
        "- Prefer removing or slightly adjusting notes over adding many new notes.\n"
        f"- Keep structure, phrasing and coverage intact across the full **{length} bars**.\n"
        "\n**Change Budget (Hard Limits):**\n"
        "1) Replace, add or delete at most ~10–15% of notes.\n"
        "2) Do not change harmony/key, and do not alter the main motif.\n"
        "3) Only add subtle automation if it clearly solves a musical issue.\n"
    )
    
    # --- NEW: Dynamically create the primary goal based on user input ---
    optimization_goal_prompt = ""
    if user_optimization_prompt:
        # User has a specific goal
        optimization_goal_prompt = (
            f"**--- Your Primary Optimization Goal ---**\n"
            f"A user has provided specific feedback. Your main priority is to modify the track to meet this request:\n"
            f"**User Request:** '{user_optimization_prompt}'\n\n"
            f"**--- Original Context (For Reference) ---**\n"
            f"To guide your changes, here was the original description for this part. The User Request above takes priority if they conflict.\n"
            f"**Original Part Description:** '{theme_detailed_description}'\n\n"
        )
    else:
        # User did not provide a goal, so the original description IS the goal
        optimization_goal_prompt = (
            f"**--- Your Primary Optimization Goal ---**\n"
            f"A user has requested a general enhancement. Your main priority is to improve the track based on its original creative description:\n"
            f"**Original Part Description:** '{theme_detailed_description}'\n\n"
        )

    # --- NEW: Advanced Automation Instructions (Copied from generation prompt logic) ---
    automation_instructions = ""
    automation_settings = config.get("automation_settings", {})
    use_pitch_bend = automation_settings.get("use_pitch_bend", 0) == 1
    use_cc_automation = automation_settings.get("use_cc_automation", 0) == 1
    use_sustain_pedal = automation_settings.get("use_sustain_pedal", 0) == 1

    if use_pitch_bend or use_cc_automation or use_sustain_pedal:
        automation_instructions += "\n**--- Advanced MIDI Automation (Apply where musically appropriate) ---**\n"
        if use_pitch_bend:
            automation_instructions += (
                f'- **Pitch Bend (curves):** Add a `"pitch_bend"` array under `note.automations` with curve objects: `{{"type":"curve","start_beat":0.0,"end_beat":0.5,"start_value":0,"end_value":8191,"bias":1.0}}`. Range: −8192..8191.\n'
                f'  Example: `{{"pitch":60, "start_beat":0, "duration_beats":1, "velocity":100, "automations":{{"pitch_bend":[{{"type":"curve","start_beat":0.0,"end_beat":0.5,"start_value":0,"end_value":8191,"bias":1.0}}]}}}}`\n'
            )
        if use_cc_automation:
            allowed_ccs = ", ".join(map(str, automation_settings.get("allowed_cc_numbers", [])))
            automation_instructions += (
                f'- **CC (curves):** Use only CCs [{allowed_ccs}]. Track‑level under `track_automations.cc` or note‑level under `note.automations.cc` as curves: `{{"type":"curve","cc":74,"start_beat":0.0,"end_beat":4.0,"start_value":60,"end_value":127,"bias":1.0}}`.\n'
            )
        if use_sustain_pedal:
            automation_instructions += (
                f'- **Sustain Pedal (CC64):** Track‑level events `{{"beat":x,"action":"down|up"}}`. Example: `{{"sustain_pedal":[{{"beat":0.0,"action":"down"}},{{"beat":3.5,"action":"up"}}]}}`.\n\n'
            )
        automation_instructions += "**CRITICAL AUTOMATION TASK:** Use the curve schema consistently and return to neutral values (e.g., pitch bend → 0).\n\n"


    # --- Drum Map Instructions ---
    drum_map_instructions = ""
    if role == "drums":
        drum_map_instructions = (
            "**Drum Map Guidance (Addictive Drums 2 Standard):**\n"
            "You MUST use the following MIDI notes for the corresponding drum sounds.\n"
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
        polyphony_rule = "2.  **Strictly Monophonic:** The notes in the JSON object's 'notes' array must NOT overlap in time."

    stay_in_key_rule = f"3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n"
    if role in ["drums", "percussion", "kick_and_snare"]:
        stay_in_key_rule = "3.  **Use Drum Map:** You must adhere to the provided Drum Map for all note pitches.\n"

    role_instructions = get_role_instructions_for_optimization(role, config)

    # --- Smart Context Calculation ---
    base_prompt_size = len(
        basic_instructions + inner_context_prompt_part + original_part_prompt +
        optimization_instruction + optimization_goal_prompt + role_instructions +
        drum_map_instructions + polyphony_rule + stay_in_key_rule
    )
    
    # Respect fixed context window size if configured (>0)
    historical_source = historical_themes_context
    cws = config.get("context_window_size", -1)
    if isinstance(cws, int) and cws > 0:
        historical_source = historical_source[-cws:]
    elif isinstance(cws, int) and cws == 0:
        historical_source = []

    history_budget = MAX_CONTEXT_CHARS - base_prompt_size
    safe_context_themes, context_start_index = get_dynamic_context(historical_source, character_budget=history_budget)
    
    # --- Context Info Printout ---
    total_previous_themes = len(historical_source)
    used_themes_count = len(safe_context_themes)
    if total_previous_themes > 0:
        limit_source = ""
        # Determine the source of limitation (window size vs character budget)
        if cws > 0:
             if total_previous_themes < cws:
                 limit_source = f"limited by availability (only {total_previous_themes} previous themes exist, window={cws})"
             elif used_themes_count < total_previous_themes:
                 limit_source = f"due to character limit (MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS})"
             else:
                 limit_source = f"due to 'context_window_size={config['context_window_size']}'"
        elif cws == -1:
             if used_themes_count < total_previous_themes:
                 limit_source = f"due to character limit (MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS})"
             else:
                 limit_source = "using full available history"
        
        # Compute indices relative to the source slice within the original history
        original_total_hist = len(historical_themes_context) + (len(historical_source) - total_previous_themes)
        offset_into_original = original_total_hist - total_previous_themes
        start_theme_num = offset_into_original + context_start_index + 1
        end_theme_num = start_theme_num + used_themes_count - 1
        print(Fore.CYAN + f"Context Info: Using {used_themes_count}/{total_previous_themes} previous themes (original indices {start_theme_num} to {end_theme_num}) for optimization context ({limit_source})." + Style.RESET_ALL)
        try:
            globals()['LAST_CONTEXT_COUNT'] = int(used_themes_count)
            globals()['PLANNED_CONTEXT_COUNT'] = int(used_themes_count)
        except Exception:
            pass
    
    # --- Context from PREVIOUS themes (with normalized timestamps) ---
    previous_themes_prompt_part = ""
    theme_length_beats = length * config["time_signature"]["beats_per_bar"]
    if safe_context_themes:
        previous_themes_prompt_part = "**Context from Previous Song Sections:**\nThe song begins with the following part(s). Their timings are relative to the start of their own section. Use them as a reference for your optimization.\n"
        for i, theme in enumerate(safe_context_themes):
            theme_name = theme.get("description", f"Theme {chr(65 + context_start_index + i)}")
            # Calculate the absolute time offset for this specific context theme
            context_theme_index_for_offset = context_start_index + i
            time_offset_beats = context_theme_index_for_offset * theme_length_beats

            previous_themes_prompt_part += f"- **{theme_name}**:\n"
            for track in theme.get('tracks', []):
                # Normalize notes to be relative to the start of their own theme
                normalized_notes = []
                for note in track.get('notes', []):
                    try:
                        sb = float(note.get('start_beat'))
                    except Exception:
                        # Skip malformed notes lacking start_beat
                        continue
                    new_note = dict(note)
                    new_note['start_beat'] = max(0, round(sb - time_offset_beats, 4))
                    normalized_notes.append(new_note)

                notes_as_str = json.dumps(normalized_notes, separators=(',', ':'))
                previous_themes_prompt_part += f"  - **{track.get('instrument_name', 'Unknown Instrument')}** (Role: {track.get('role', 'complementary')}):\n  ```json\n  {notes_as_str}\n  ```\n"
        previous_themes_prompt_part += "\n"

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
        f"**Motif Coherence:** Reprise the main motif where musically appropriate, and prefer transformations (inversion, octave shift, rhythm augmentation) over new ideas.\n\n"
        f"{optimization_goal_prompt}"
        f"{role_instructions}\n"
        f"{drum_map_instructions}"
        f"--- Your Producer's Checklist for Optimization ---\n\n"
        f"1.  **Seam Awareness:** If this section joins consecutive parts, keep transitions natural. If already fine, **change nothing**. If slightly abrupt, use a **tiny** connective gesture; avoid large fills.\n\n"
        f"2.  **Continuity First:** Prefer continuity over contrast for optimization; respect genre and the part description.\n\n"
        f"3.  **Preserve Identity:** Keep motifs, register, and rhythm shape. **No full rewrites.**\n\n"
        f"4.  **Density:** If cluttered, remove or shorten a few low‑value notes. If too sparse or robotic, add **very few** supportive notes (≤ change budget).\n\n"
        f"5.  **Dynamics & Feel:** Tweak velocities and micro‑timing **subtly**. Avoid large timing shifts or drastic accents.\n\n"
        f"6.  **Ensemble Fit:** Leave space for other instruments; avoid stepping on primary elements.\n\n"

        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Your response MUST be a single, valid JSON object with ONLY these top-level keys: `notes` (required), `track_automations` (optional), `sustain_pedal` (optional), `pattern_blocks` (optional). No other top-level keys. No prose, no markdown.\n\n"
        f"Minimal example (structure only):\n"
        f"```json\n"
        f"{{\n  \"notes\": [{{\"pitch\": 60, \"start_beat\": 0.0, \"duration_beats\": 1.0, \"velocity\": 100}}],\n  \"track_automations\": {{\"cc\": [{{\"type\": \"curve\", \"cc\": 11, \"start_beat\": 0.0, \"end_beat\": 4.0, \"start_value\": 60, \"end_value\": 80, \"bias\": 1.0}}]}},\n  \"sustain_pedal\": [{{\"beat\": 0.0, \"action\": \"down\"}}, {{\"beat\": 3.5, \"action\": \"up\"}}]\n}}\n"
        f"```\n\n"
        f'1.  **Notes Array:** The JSON object MUST have a key named `"notes"` containing an array of note objects. Each note object MUST have these keys:\n'
        f'    - **"pitch"**: MIDI note number (integer 0-127).\n'
        f'    - **"start_beat"**: The beat on which the note begins (float, relative to the start of this part).\n'
        f'    - **"duration_beats"**: The note\'s length in beats (float).\n'
        f'    - **"velocity"**: MIDI velocity (integer 1-127).\n\n'
        f"**--- IMPORTANT RULES ---**\n"
        f'1.  **JSON OBJECT ONLY:** The entire response MUST be only the raw JSON object (no markdown, no commentary).\n'
        f"{polyphony_rule}\n"
        f"{stay_in_key_rule}"
        f"4.  **Timing is Relative:** All 'start_beat' values must be relative to the beginning of this {length}-bar section, NOT the whole song.\n"
        f"5.  **Be Creative:** Compose a high-quality, optimized part that is musically interesting and follows the creative direction.\n"
        f'6.  **Valid JSON Syntax:** The output must be a perfectly valid JSON object.\n'
        f'7.  **Handling Silence:** If the instrument is intentionally silent for the entire section, return ONLY: `{{"notes": [{{"pitch": 0, "start_beat": 0, "duration_beats": 0, "velocity": 0}}]}}`. Do not return any other keys.\n'
        f'8.  **Density Cap:** Keep the total number of notes for this part reasonable (≤ 400) to avoid excessive density.\n'
        f'9.  **Sorting & Formatting:** Use dot decimals (e.g., 1.5), non-negative beats, and sort notes by `start_beat` ascending.\n\n'
        f"Optional compact format for dense rhythms: You may include a 'pattern_blocks' array to represent many fast steps (e.g., 32nd/64th). Each block may have: 'length_bars', 'subdivision', 'bar_repeats', optional 'transpose' or 'octave_shift', and 'steps' with either 'mask' (e.g., '1010..') or 'indices' [0,2,4]. Ensure that expanded notes do not exceed the section length.\n\n"
        f"Now, generate the JSON object for the new, optimized version of the **{get_instrument_name(track_to_optimize)}** track for the section '{theme_label}'. The generated part MUST cover the full {length} bars.\n"
    )
    return prompt

def generate_optimization_data(config: Dict, length: int, track_to_optimize: Dict | None, role: str, theme_label: str, theme_detailed_description: str, historical_themes_context: List[Dict], inner_context_tracks: List[Dict], current_theme_index: int, user_optimization_prompt: str = "") -> Tuple[Dict, int]:
    """
    Generates an optimized version of a single track's notes using the generative model.
    """
    global CURRENT_KEY_INDEX, SESSION_MODEL_OVERRIDE
    # Defensive defaults
    if track_to_optimize is None or not isinstance(track_to_optimize, dict):
        track_to_optimize = {"instrument_name": "track", "program_num": 0, "role": role or "complementary", "notes": []}
    user_optimization_prompt = user_optimization_prompt or ""

    prompt = create_optimization_prompt(
        config, length, track_to_optimize, role, theme_label, theme_detailed_description,
        historical_themes_context, inner_context_tracks, current_theme_index, user_optimization_prompt
    )
    max_retries = 6
    local_model_name = SESSION_MODEL_OVERRIDE or config["model_name"]
    try:
        origin = "session default" if SESSION_MODEL_OVERRIDE else "config default"
        print(Fore.CYAN + f"Model for this step: {local_model_name} ({origin}). Press 1/2/3 to switch for THIS step only; press 0 to set current as session default." + Style.RESET_ALL)
    except Exception:
        pass
    json_failure_count = 0
    max_tokens_fail_flash = 0
    max_tokens_fail_pro = 0
    # Reset custom prompt guard for this step and show hint
    global PROMPTED_CUSTOM_THIS_STEP, REQUESTED_SWITCH_MODEL, REQUEST_SET_SESSION_DEFAULT, ABORT_CURRENT_STEP, DEFER_CURRENT_TRACK, REDUCE_CONTEXT_THIS_STEP, REDUCE_CONTEXT_HALVES, LAST_CONTEXT_COUNT
    PROMPTED_CUSTOM_THIS_STEP = False
    REQUESTED_SWITCH_MODEL = None
    REQUEST_SET_SESSION_DEFAULT = False
    ABORT_CURRENT_STEP = False
    DEFER_CURRENT_TRACK = False
    REDUCE_CONTEXT_THIS_STEP = False
    REDUCE_CONTEXT_HALVES = 0
    LAST_CONTEXT_COUNT = 0
    # Show hotkey hint before attempts
    print_hotkey_hint(config, context=f"Optimize: {track_to_optimize.get('instrument_name','track')}")
    global HOTKEY_MONITOR_STARTED
    if not HOTKEY_MONITOR_STARTED:
        try:
            t = threading.Thread(target=_hotkey_monitor_loop, args=(config,), daemon=True)
            t.start(); HOTKEY_MONITOR_STARTED = True
        except Exception:
            HOTKEY_MONITOR_STARTED = True

    for attempt in range(max_retries):
        start_key_index = CURRENT_KEY_INDEX
        quota_rotation_count = 0  # Counts full key-rotation exhaustion cycles to drive long backoff
        
        while True: # Inner loop for API key rotation
            try:
                instrument_name = (track_to_optimize or {}).get('instrument_name', 'track')
                print(Fore.BLUE + f"Attempt {attempt + 1}/{max_retries}: Generating optimization for {instrument_name}..." + Style.RESET_ALL)
                
                generation_config = {
                    "temperature": config["temperature"],
                    "response_mime_type": "application/json",
                    "max_output_tokens": config.get("max_output_tokens", 8192)
                }

                # Poll persistent hotkey (Windows) to change model instantly
                local_model_name = _poll_model_switch(local_model_name, config)
                model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]

                # Build effective prompt: always add strict output hint
                effective_prompt = prompt + "\nOutput: Return a single JSON object with a 'notes' key only; no prose.\n"
                # Degrade: remove automation section only after 4th failure (attempt index >= 4)
                if attempt >= 4:
                    effective_prompt = re.sub(r"\*\*\-\-\- Advanced MIDI Automation[\s\S]*?\n\n", "", effective_prompt)

                if ABORT_CURRENT_STEP or DEFER_CURRENT_TRACK:
                    if REQUESTED_SWITCH_MODEL:
                        local_model_name = REQUESTED_SWITCH_MODEL
                        model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                    if REQUEST_SET_SESSION_DEFAULT:
                        SESSION_MODEL_OVERRIDE = local_model_name
                        REQUEST_SET_SESSION_DEFAULT = False
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + "Deferring current optimization track on user request (hotkey 'd')." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        return None, 0
                    ABORT_CURRENT_STEP = False
                response = model.generate_content(effective_prompt, safety_settings=safety_settings, generation_config=generation_config)
                
                # Mid-call hotkey: if requested, restart this attempt with new model
                if ABORT_CURRENT_STEP or DEFER_CURRENT_TRACK:
                    if REQUESTED_SWITCH_MODEL:
                        local_model_name = REQUESTED_SWITCH_MODEL
                    if REQUEST_SET_SESSION_DEFAULT:
                        SESSION_MODEL_OVERRIDE = local_model_name
                        REQUEST_SET_SESSION_DEFAULT = False
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + "Deferring current optimization track on user request (hotkey 'd')." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        return None, 0
                    print(Fore.CYAN + f"Restarting step with model: {local_model_name}" + Style.RESET_ALL)
                    ABORT_CURRENT_STEP = False
                    continue

                response_text = _extract_text_from_response(response)
                if not response_text:
                    print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Empty or invalid response payload for {instrument_name}." + Style.RESET_ALL)
                    json_failure_count += 1
                    if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                        local_model_name = 'gemini-2.5-pro'
                        model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                        print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
                    continue

                # --- NEW: Token Usage Reporting ---
                total_token_count = 0
                if hasattr(response, 'usage_metadata'):
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    total_token_count = response.usage_metadata.total_token_count
                    char_per_token = len(prompt) / prompt_tokens if prompt_tokens > 0 else 0
                    print(Fore.CYAN + f"Token Usage: Prompt: {prompt_tokens:,} | Output: {output_tokens:,} | Total: {total_token_count:,} "
                                      f"(Prompt: {len(prompt):,} chars, ~{char_per_token:.2f} chars/token)" + Style.RESET_ALL)

                # --- Data Validation ---
                json_payload = _extract_json_object(response_text)
                if not json_payload:
                    json_failure_count += 1
                    print(Fore.YELLOW + f"Warning on attempt {attempt + 1}: Could not extract JSON object for {instrument_name}." + Style.RESET_ALL)
                    # Non-blocking quick switch (Windows): press 1/2/3 during next wait window
                    if json_failure_count == 2 and sys.platform == "win32":
                        print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (config.custom_model_name) to switch model for THIS track; continuing attempts..." + Style.RESET_ALL)
                    # Auto-escalate: if using flash and threshold exceeded
                    if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                        local_model_name = 'gemini-2.5-pro'
                        print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
                    # After two MAX_TOKENS or repeated truncation symptoms, reduce context window by half
                    try:
                        if (('flash' in (local_model_name or '')) and max_tokens_fail_flash >= 2) or (('pro' in (local_model_name or '')) and max_tokens_fail_pro >= 2):
                            reduced_cfg = json.loads(json.dumps(config))
                            source = historical_themes_context or []
                            if isinstance(source, list) and source:
                                half = max(1, len(source)//2)
                                reduced_cfg["context_window_size"] = half
                                print(Fore.CYAN + f"Reducing optimization context window to {half} due to repeated token issues." + Style.RESET_ALL)
                                prompt = create_optimization_prompt(
                                    reduced_cfg, length, track_to_optimize, role, theme_label, theme_detailed_description,
                                    source[-half:], inner_context_tracks, current_theme_index, user_optimization_prompt
                                )
                    except Exception:
                        pass
                    continue
                parsed_data = json.loads(json_payload)
                
                if isinstance(parsed_data, dict) and "notes" not in parsed_data and isinstance(parsed_data.get("pattern_blocks"), list):
                    parsed_data["notes"] = []
                if not isinstance(parsed_data, dict) or "notes" not in parsed_data:
                    raise ValueError("JSON is valid but does not contain the required 'notes' key.")
                
                # Optional compact pattern support
                try:
                    pattern_blocks = parsed_data.get("pattern_blocks")
                    if isinstance(pattern_blocks, list):
                        expanded = _expand_pattern_blocks(pattern_blocks, length, config["time_signature"]["beats_per_bar"])
                        if expanded:
                            if isinstance(parsed_data.get("notes"), list):
                                parsed_data["notes"].extend(expanded)
                            else:
                                parsed_data["notes"] = expanded
                except Exception:
                    pass

                # Harmonize intentional silence handling with generation path
                try:
                    notes_list = parsed_data.get("notes", [])
                    if isinstance(notes_list, list) and len(notes_list) == 1:
                        n0 = notes_list[0]
                        if isinstance(n0, dict) and n0.get("pitch") == 0 and n0.get("start_beat") == 0 and n0.get("duration_beats") == 0 and n0.get("velocity") == 0:
                            parsed_data["notes"] = []
                            if "sustain_pedal" not in parsed_data:
                                parsed_data["sustain_pedal"] = []
                except Exception:
                    pass
                
                total_token_count = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                return parsed_data, total_token_count

            except Exception as e:
                error_message = str(e).lower()
                if "429" in error_message and "quota" in error_message:
                    qtype = _classify_quota_error(error_message)
                    print(Fore.YELLOW + f"Warning: API quota exceeded for key #{CURRENT_KEY_INDEX + 1} ({qtype})." + Style.RESET_ALL)
                    # For daily quotas: retry hourly to probe reset windows
                    if qtype == "per-day":
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                    elif qtype == "per-hour":
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                    else:
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_MINUTE_COOLDOWN_SECONDS)
                    
                    if len(API_KEYS) > 1:
                        nxt = _next_available_key()
                        if nxt is not None:
                            CURRENT_KEY_INDEX = nxt
                            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                            print(Fore.CYAN + f"Retrying immediately with available key #{CURRENT_KEY_INDEX + 1}...")
                            continue # Retry the API call with the new key
                        else:
                            print(Fore.RED + "All available API keys are cooling down or have exceeded their quota.")
                            # Long backoff while waiting for first key to free up
                            quota_rotation_count += 1
                            base = 3
                            wait_time = max(5.0, _seconds_until_first_available())
                            jitter = random.uniform(0, 5.0)
                            wait_time = min(3600, wait_time + jitter)
                            print(Fore.CYAN + f"Next available key in ~{wait_time:.1f}s. Waiting before retry..." + Style.RESET_ALL)
                            time.sleep(wait_time)
                            continue
                    else:
                        print(Fore.RED + "The only available API key has exceeded its quota.")
                        # Long backoff for single-key setups
                        quota_rotation_count += 1
                        base = 3
                        wait_time = max(5.0, _seconds_until_first_available())
                        jitter = random.uniform(0, 5.0)
                        wait_time = min(3600, wait_time + jitter)
                        print(Fore.YELLOW + f"API key exhausted. Waiting for {wait_time:.1f} seconds before retrying..." + Style.RESET_ALL)
                        time.sleep(wait_time)
                        continue
                else:
                    # Detect MAX_TOKENS phase and defer if needed
                    if "max_tokens" in error_message:
                        if 'flash' in (local_model_name or ''):
                            max_tokens_fail_flash += 1
                            if max_tokens_fail_flash >= 6:
                                if AUTO_ESCALATE_TO_PRO:
                                    local_model_name = 'gemini-2.5-pro'
                                    max_tokens_fail_pro = 0
                                    print(Fore.CYAN + "Auto-escalate after 6 MAX_TOKENS on flash → switching to pro for this track (optimization)." + Style.RESET_ALL)
                                else:
                                    print(Fore.YELLOW + "Deferring this optimization track due to repeated MAX_TOKENS on flash (6 attempts)." + Style.RESET_ALL)
                                    return None, 0
                        elif 'pro' in (local_model_name or ''):
                            max_tokens_fail_pro += 1
                            if max_tokens_fail_pro >= 6:
                                print(Fore.YELLOW + "Deferring this optimization track due to repeated MAX_TOKENS on pro (6 attempts)." + Style.RESET_ALL)
                                return None, 0
                    print(Fore.RED + f"An unexpected error during optimization on attempt {attempt + 1}: {str(e)}" + Style.RESET_ALL)
                    # Offer a quick, non-blocking model switch for JSON parse/content issues
                    if sys.platform == "win32" and any(k in error_message for k in ["json", "expecting", "delimiter", "notes"]):
                        print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (config.custom_model_name) to switch model for THIS track; continuing attempts..." + Style.RESET_ALL)
                        try:
                            end_t = time.time() + 3
                            while time.time() < end_t:
                                if msvcrt.kbhit():
                                    ch = msvcrt.getch().decode().lower()
                                    if ch == '1': local_model_name = 'gemini-2.5-pro'; break
                                    if ch == '2': local_model_name = 'gemini-2.5-flash'; break
                                    if ch == '3':
                                        custom = config.get('custom_model_name')
                                        if custom: local_model_name = custom
                                        break
                                time.sleep(0.2)
                        except Exception:
                            pass
                    # Backoff and retry next outer attempt
                    break

        # If we broke from the inner loop, it means a full rotation failed or a non-quota error happened.
        # Now we wait before the next of the main retry attempts (exponential backoff with jitter).
        if attempt < max_retries - 1:
            base = 3
            wait_time = base * (2 ** attempt)
            jitter = random.uniform(0, 1.5)
            wait_time = min(30, wait_time + jitter)
            print(Fore.YELLOW + f"Waiting for {wait_time:.1f} seconds before retrying optimization..." + Style.RESET_ALL)
            time.sleep(wait_time)

    # If all 3 main retries fail, ask the user what to do.
    instrument_name = (track_to_optimize or {}).get('instrument_name', 'track')
    print(Fore.RED + f"Failed to generate a valid optimization for {instrument_name} after {max_retries} full attempts." + Style.RESET_ALL)
    
    print(Fore.CYAN + "Automatic retry in 60 seconds..." + Style.RESET_ALL)
    print(Fore.YELLOW + "Press 'y' to retry now, 'n' to cancel, or 'w' to pause the timer." + Style.RESET_ALL)

    user_action = None
    for i in range(60, 0, -1):
        if sys.platform == "win32" and msvcrt.kbhit():
            char = msvcrt.getch().decode().lower()
            if char in ['y', 'n', 'w']:
                user_action = char
                break
        print(f"  Retrying in {i} seconds...  ", end="\r")
        time.sleep(1)
    print("                               ", end="\r")

    # Decide action
    if user_action is None or user_action == 'y':
        print(Fore.CYAN + "Retrying now..." + Style.RESET_ALL)
        return generate_optimization_data(config, length, (track_to_optimize or {"instrument_name":"track","program_num":0,"role":(role or "complementary"),"notes":[]}), role, theme_label, theme_detailed_description, historical_themes_context, inner_context_tracks, current_theme_index, (user_optimization_prompt or ""))
    elif user_action == 'n':
        print(Fore.RED + f"Aborting optimization for '{instrument_name}'." + Style.RESET_ALL)
        return None, 0
    elif user_action == 'w':
        print(Fore.YELLOW + "Timer paused. Waiting for manual input." + Style.RESET_ALL)
        while True:
            manual_choice = input(Fore.YELLOW + f"Retry for '{instrument_name}'? (y/n): " + Style.RESET_ALL).strip().lower()
            if manual_choice in ['y', 'yes']:
                return generate_optimization_data(config, length, (track_to_optimize or {"instrument_name":"track","program_num":0,"role":(role or "complementary"),"notes":[]}), role, theme_label, theme_detailed_description, historical_themes_context, inner_context_tracks, current_theme_index, (user_optimization_prompt or ""))
            elif manual_choice in ['n', 'no']:
                print(Fore.RED + f"Aborting optimization for '{instrument_name}'." + Style.RESET_ALL)
                return None, 0
            else:
                print(Fore.YELLOW + "Invalid input. Please enter 'y' or 'n'." + Style.RESET_ALL)

    return None, 0 # Default exit

# --- AUTOMATION ENHANCEMENT MODE (NEW) ---
def create_automation_prompt(config: Dict, length: int, base_track: Dict, role: str, theme_label: str, theme_detailed_description: str, historical_themes_context: List[Dict], inner_context_tracks: List[Dict], current_theme_index: int, enhancement_mode: str = "auto") -> str:
    scale_notes = get_scale_notes(config["root_note"], config["scale_type"])

    # Compact original track JSON
    original_part_str = json.dumps({
        'instrument_name': get_instrument_name(base_track),
        'program_num': base_track.get('program_num', 0),
        'role': base_track.get('role', 'complementary'),
        'notes': base_track.get('notes', [])
    }, separators=(',', ':'))

    basic_instructions = (
        f"**Genre:** {config['genre']}\n"
        f"**Tempo:** {config['bpm']} BPM\n"
        f"**Time Signature:** {config['time_signature']['beats_per_bar']}/{config['time_signature']['beat_value']}\n"
        f"**Key/Scale:** {config['key_scale'].title()} (Available notes: {scale_notes})\n"
        f"**Instrument:** {get_instrument_name(base_track)} (MIDI Program: {base_track.get('program_num', 0)})\n"
        f"**Section Length:** {length} bars\n"
    )

    # Inner context (compact)
    inner_context_prompt_part = ""
    if inner_context_tracks:
        inner_context_prompt_part = "**Context from the Current Song Section:**\nWithin this section, other tracks already exist. Make your automation musically fit them.\n"
        for track in inner_context_tracks:
            try:
                notes = track.get('notes', [])
                if isinstance(notes, list) and len(notes) > MAX_NOTES_IN_CONTEXT:
                    head = notes[:MAX_NOTES_IN_CONTEXT//2]; tail = notes[-MAX_NOTES_IN_CONTEXT//2:]
                    notes = head + tail
            except Exception:
                notes = track.get('notes', [])
            inner_context_prompt_part += f"- **{get_instrument_name(track)}** (Role: {track.get('role','complementary')}):\n```json\n{json.dumps(notes, separators=(',', ':'))}\n```\n"
        inner_context_prompt_part += "\n"

    # Historical themes (relative times)
    theme_length_beats = length * config["time_signature"]["beats_per_bar"]
    history_prompt = ""
    if historical_themes_context:
        history_prompt = "**Context from Previous Sections (relative to each section start):**\n"
        for i, theme in enumerate(historical_themes_context[-3:]):  # cap to last 3 for brevity
            history_prompt += f"- **{theme.get('description', f'Theme {i+1}')}:**\n"
            for track in theme.get('tracks', []):
                normalized = []
                for note in track.get('notes', []):
                    try:
                        sb = float(note.get('start_beat', 0))
                        new_note = dict(note); new_note['start_beat'] = max(0, round(sb % theme_length_beats, 4))
                        normalized.append(new_note)
                    except Exception:
                        continue
                history_prompt += f"  - **{get_instrument_name(track)}**:```json\n{json.dumps(normalized[:MAX_NOTES_IN_CONTEXT], separators=(',', ':'))}\n```\n"
        history_prompt += "\n"

    # Automation flags
    a = config.get('automation_settings', {})
    use_pb = a.get('use_pitch_bend', 0) == 1
    use_cc = a.get('use_cc_automation', 0) == 1
    use_sus = a.get('use_sustain_pedal', 0) == 1
    allowed_ccs = a.get('allowed_cc_numbers', [])

    # Detect existing automations in the base_track (to allow "improve" vs "add")
    has_track_pb = bool(base_track.get('track_automations', {}).get('pitch_bend')) if isinstance(base_track.get('track_automations'), dict) else False
    has_track_cc = bool(base_track.get('track_automations', {}).get('cc')) if isinstance(base_track.get('track_automations'), dict) else False
    has_sustain = bool(base_track.get('sustain_pedal'))
    # Note-level detection (counts only for brevity)
    note_pb_count = 0
    note_cc_count = 0
    try:
        for n in base_track.get('notes', [])[:200]:  # scan up to 200 notes for speed
            autos = n.get('automations', {}) if isinstance(n, dict) else {}
            if isinstance(autos, dict):
                if isinstance(autos.get('pitch_bend'), list) and autos.get('pitch_bend'):
                    note_pb_count += 1
                if isinstance(autos.get('cc'), list) and autos.get('cc'):
                    note_cc_count += 1
    except Exception:
        pass
    has_existing = any([has_track_pb, has_track_cc, has_sustain, note_pb_count > 0, note_cc_count > 0])

    # Build automation instructions only for enabled features
    auto_text = ""
    if use_pb or use_cc or use_sus:
        # Mode guidance
        mode_line = ""
        if enhancement_mode == 'improve' or (enhancement_mode == 'auto' and has_existing):
            mode_line = ("Improve and refine existing automation curves. You MAY also add new curves when musically justified. "
                         "Do not remove useful curves; remove only if they clearly harm the musical intent.\n")
        elif enhancement_mode == 'add' or (enhancement_mode == 'auto' and not has_existing):
            mode_line = ("Add new expressive automations. Prefer subtlety and musicality. Ensure proper resets.\n")
        auto_text += "**--- Automation Goals (Only enabled types) ---**\n" + mode_line
        if use_pb:
            auto_text += ("- **Pitch Bend:** Add expressive slides/vibrato where musical. Range −8192..8191. Always return to 0 after each phrase.\n"
                          "  You may split notes to reflect bends accurately.\n")
        if use_cc:
            auto_text += (f"- **Control Change:** Use only allowed CCs {allowed_ccs}. Prefer long, smooth curves for timbral movement.\n"
                          "  Always reset to a neutral value at the end of phrases.\n")
        if use_sus:
            auto_text += ("- **Sustain Pedal (CC 64):** Use paired events 'down'/'up' to create legato; adjust note lengths minimally as needed.\n")

    # Allowed minimal note edits
    note_edit_text = ("**Note Edits Allowed (Minimal):**\n"
                      "- Split/merge notes to support bends/legato.\n"
                      "- Slight length/position tweaks to fit automation phrasing.\n"
                      "- Do NOT rewrite the musical content drastically.\n\n")

    # Existing automation summary (shown if present and enabled)
    existing_summary_text = ""
    try:
        if has_existing:
            existing_summary_text += "**--- Existing Automation Summary ---**\n"
            if use_pb and has_track_pb:
                pb = base_track.get('track_automations', {}).get('pitch_bend', [])
                existing_summary_text += f"- Track Pitch Bend curves: {len(pb)}\n"
            if use_cc and has_track_cc:
                cc = base_track.get('track_automations', {}).get('cc', [])
                existing_summary_text += f"- Track CC curves: {len(cc)}\n"
            if use_sus and has_sustain:
                sus = base_track.get('sustain_pedal', [])
                existing_summary_text += f"- Sustain pedal events: {len(sus)}\n"
            if use_pb and note_pb_count:
                existing_summary_text += f"- Notes with note-level pitch_bend: {note_pb_count}\n"
            if use_cc and note_cc_count:
                existing_summary_text += f"- Notes with note-level CC: {note_cc_count}\n"
            existing_summary_text += "\n"
    except Exception:
        pass

    prompt = (
        f"You are an expert MIDI musician. Your task is to enhance expression by adding automations to a single track.\n\n"
        f"**--- MUSICAL CONTEXT ---**\n"
        f"{basic_instructions}\n"
        f"{history_prompt}"
        f"**--- CURRENT SECTION CONTEXT ---**\n"
        f"{inner_context_prompt_part}"
        f"**--- ORIGINAL TRACK ---**\n"
        f"```json\n{original_part_str}\n```\n\n"
        f"{existing_summary_text}"
        f"**--- YOUR TASK ---**\n"
        f"Enhance the track with expressive automation. Respect the musical intent and keep note edits minimal.\n\n"
        f"{auto_text}"
        f"{note_edit_text}"
        f"**Output Format (JSON):** A single object with keys: `notes` (required), optionally `automations` per note, `track_automations`, and `sustain_pedal` (if enabled).\n"
        f"- Notes have keys: pitch, start_beat (relative to this {length}-bar section), duration_beats, velocity.\n"
        f"- Keep JSON valid. No comments or prose.\n"
    )
    return prompt

def generate_automation_data(config: Dict, length: int, base_track: Dict, role: str, theme_label: str, theme_detailed_description: str, historical_themes_context: List[Dict], inner_context_tracks: List[Dict], current_theme_index: int, enhancement_mode: str = "auto") -> Tuple[Dict, int]:
    global CURRENT_KEY_INDEX, SESSION_MODEL_OVERRIDE
    # Reset custom prompt guard for this step and show hint
    global PROMPTED_CUSTOM_THIS_STEP, REQUESTED_SWITCH_MODEL, REQUEST_SET_SESSION_DEFAULT, ABORT_CURRENT_STEP, DEFER_CURRENT_TRACK
    PROMPTED_CUSTOM_THIS_STEP = False
    REQUESTED_SWITCH_MODEL = None
    REQUEST_SET_SESSION_DEFAULT = False
    ABORT_CURRENT_STEP = False
    DEFER_CURRENT_TRACK = False
    # Show hotkey hint before attempts
    print_hotkey_hint(config, context=f"Automation: {get_instrument_name(base_track)}")
    global HOTKEY_MONITOR_STARTED
    if not HOTKEY_MONITOR_STARTED:
        try:
            t = threading.Thread(target=_hotkey_monitor_loop, args=(config,), daemon=True)
            t.start(); HOTKEY_MONITOR_STARTED = True
        except Exception:
            HOTKEY_MONITOR_STARTED = True
    prompt = create_automation_prompt(config, length, base_track, role, theme_label, theme_detailed_description, historical_themes_context, inner_context_tracks, current_theme_index, enhancement_mode=enhancement_mode)
    max_retries = 6
    local_model_name = config["model_name"]
    json_failure_count = 0
    for attempt in range(max_retries):
        start_key_index = CURRENT_KEY_INDEX
        quota_rotation_count = 0
        while True:
            try:
                generation_config = {
                    "temperature": config["temperature"],
                    "response_mime_type": "application/json",
                    "max_output_tokens": config.get("max_output_tokens", 8192)
                }
                model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
                effective = prompt + "\nOutput: Return a single JSON object with a 'notes' key only; no prose.\n"
                if ABORT_CURRENT_STEP or DEFER_CURRENT_TRACK:
                    if REQUESTED_SWITCH_MODEL:
                        local_model_name = REQUESTED_SWITCH_MODEL
                        model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                    if REQUEST_SET_SESSION_DEFAULT:
                        SESSION_MODEL_OVERRIDE = local_model_name
                        REQUEST_SET_SESSION_DEFAULT = False
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + "Deferring current automation track on user request (hotkey 'd')." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        return None, 0
                    ABORT_CURRENT_STEP = False
                # Apply on-demand context halving (hotkey 'h')
                if REDUCE_CONTEXT_THIS_STEP:
                    try:
                        reduced_cfg = json.loads(json.dumps(config))
                        source = historical_themes_context or []
                        if isinstance(source, list) and source:
                            original = len(source)
                            LAST_CONTEXT_COUNT = original
                            halves = max(1, int(REDUCE_CONTEXT_HALVES))
                            target = original
                            for _ in range(halves):
                                target = max(1, target // 2)
                            reduced_cfg["context_window_size"] = target
                            print(Fore.CYAN + f"Applying halve optimization context (x{halves}) for this step: {original} → {target} parts." + Style.RESET_ALL)
                            print(Fore.CYAN + f"Context in use now: {target} previous theme(s)." + Style.RESET_ALL)
                            prompt = create_automation_prompt(
                                reduced_cfg, length, base_track, role, theme_label, theme_detailed_description,
                                source[-target:], inner_context_tracks, current_theme_index, enhancement_mode=enhancement_mode
                            )
                    except Exception:
                        pass
                    REDUCE_CONTEXT_THIS_STEP = False
                    REDUCE_CONTEXT_HALVES = 0
                response = model.generate_content(effective, safety_settings=safety_settings, generation_config=generation_config)

                # Mid-call hotkey: if requested, restart this attempt with new model
                if ABORT_CURRENT_STEP or DEFER_CURRENT_TRACK:
                    if REQUESTED_SWITCH_MODEL:
                        local_model_name = REQUESTED_SWITCH_MODEL
                    if REQUEST_SET_SESSION_DEFAULT:
                        SESSION_MODEL_OVERRIDE = local_model_name
                        REQUEST_SET_SESSION_DEFAULT = False
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + "Deferring current automation track on user request (hotkey 'd')." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        return None, 0
                    print(Fore.CYAN + f"Restarting step with model: {local_model_name}" + Style.RESET_ALL)
                    ABORT_CURRENT_STEP = False
                    continue

                resp_text = _extract_text_from_response(response)
                if not resp_text:
                    continue
                    # Count as a failure and possibly escalate (flash→pro)
                    json_failure_count += 1
                    if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                        local_model_name = 'gemini-2.5-pro'
                        model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                        print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
                json_payload = _extract_json_object(resp_text)
                if not json_payload:
                    json_failure_count += 1
                    if json_failure_count == 2 and sys.platform == "win32":
                        print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (config.custom_model_name) to switch model for THIS track; continuing attempts..." + Style.RESET_ALL)
                    if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                        local_model_name = 'gemini-2.5-pro'
                        model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                        print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
                    continue
                data = json.loads(json_payload)
                if not isinstance(data, dict) or "notes" not in data:
                    raise ValueError("JSON missing 'notes'")

                # Merge back into track (allow note changes)
                merged = {
                    "instrument_name": get_instrument_name(base_track),
                    "program_num": base_track.get("program_num", 0),
                    "role": role,
                    "notes": data.get("notes", base_track.get("notes", []))
                }
                # Sustain: replace if provided, else keep existing
                if "sustain_pedal" in data:
                    merged["sustain_pedal"] = data.get("sustain_pedal", [])
                elif base_track.get("sustain_pedal"):
                    merged["sustain_pedal"] = base_track.get("sustain_pedal", [])

                # Track-level automations: if new provided, use them; otherwise keep existing
                track_autos = data.get("track_automations", {}) or {}
                if not track_autos and isinstance(base_track.get("track_automations"), dict):
                    track_autos = base_track.get("track_automations", {})
                if track_autos:
                    merged["track_automations"] = track_autos
                return merged, getattr(response.usage_metadata, 'total_token_count', 0)

            except Exception as e:
                err = str(e).lower()
                if "429" in err and "quota" in err:
                    print(Fore.YELLOW + "Quota exceeded; rotating key..." + Style.RESET_ALL)
                    if len(API_KEYS) > 1:
                        new_key = get_next_api_key(); genai.configure(api_key=new_key); continue
                    base = 3; wait = min(3600, base * (2 ** quota_rotation_count) + random.uniform(0,5.0))
                    time.sleep(wait); quota_rotation_count += 1; continue
                # MAX_TOKENS quick non-blocking switch
                if "max_tokens" in err and sys.platform == "win32":
                    print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom to switch model (this track)." + Style.RESET_ALL)
                base = 3; wait = min(30, base * (2 ** attempt) + random.uniform(0,1.5)); time.sleep(wait); break
    return None, 0

def create_automation_enhancement(config: Dict, theme_length: int, themes_to_enhance: List[Dict], script_dir: str, run_timestamp: str, user_prompt: str = "", enhancement_mode: str = "auto") -> List[Dict]:
    # Guard: if all automation flags disabled, warn
    a = config.get('automation_settings', {})
    if a.get('use_pitch_bend',0)==0 and a.get('use_cc_automation',0)==0 and a.get('use_sustain_pedal',0)==0:
        print(Fore.YELLOW + "Automation settings are all disabled; no changes will be applied." + Style.RESET_ALL)
    optimized = themes_to_enhance[:]
    try:
        for theme_index, theme in enumerate(optimized):
            print(Fore.MAGENTA + f"\n--- Automation Enhancement: Theme {theme_index+1}/{len(optimized)}: '{theme.get('label','')}' ---" + Style.RESET_ALL)
            tracks = theme.get('tracks', [])
            inner_context = tracks[:]
            historical_context = optimized[:theme_index]
            for track_index, track in enumerate(tracks):
                name = get_instrument_name(track); role = track.get('role','complementary')
                if role in ["drums","percussion","kick_and_snare"]:
                    continue
                print(Fore.BLUE + f"Enhancing automations on {name} (Role: {role})" + Style.RESET_ALL)
                other = [t for i,t in enumerate(inner_context) if i!=track_index]
                new_track, _tok = generate_automation_data(config, theme_length, track, role, theme.get('label',''), theme.get('description',''), historical_context, other, theme_index, enhancement_mode=enhancement_mode)
                if new_track:
                    theme['tracks'][track_index] = new_track
                # Save resumable progress after each track
                try:
                    save_progress({
                        'type': 'automation_enhancement',
                        'config': config,
                        'theme_length': theme_length,
                        'themes': optimized,
                        'current_theme_index': theme_index,
                        'current_track_index': track_index + 1,
                        'user_optimization_prompt': user_prompt,
                        'timestamp': run_timestamp
                    }, script_dir, run_timestamp)
                except Exception:
                    pass
        # After enhancement, always write a combined final MIDI for this pass
        try:
            final_song_data = merge_themes_to_song_data(optimized, config, theme_length)
            base = build_final_song_basename(config, optimized, run_timestamp, resumed=True)
            final_path = os.path.join(script_dir, f"{base}_automation.mid")
            create_midi_from_json(final_song_data, config, final_path)
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Could not create final MIDI after automation enhancement: {e}" + Style.RESET_ALL)
        return optimized
    except Exception as e:
        print(Fore.RED + f"Automation enhancement failed: {e}" + Style.RESET_ALL)
        return None
def generate_one_theme(config, length: int, theme_def: dict, previous_themes: List[Dict], current_theme_index: int) -> Tuple[bool, Dict, int]:
    """Generates all the instrument tracks for a single theme."""
    
    print(Fore.CYAN + f"\n--- Generating Theme {current_theme_index + 1}/{len(config.get('theme_definitions', []))}: {theme_def['label']} ---" + Style.RESET_ALL)
    
    dialogue_role = "User" # Default
    total_tokens = 0
    generated_tracks = {}
    context_tracks_for_this_theme = []
    CALL_AND_RESPONSE_ROLES = {'bass', 'chords', 'arp', 'guitar', 'lead', 'melody', 'vocal'}
    call_has_been_made = False

    for i, instrument in enumerate(config['instruments']):
        
        # --- Call & Response Logic ---
        if config.get("use_call_and_response") == 1 and instrument['role'] in CALL_AND_RESPONSE_ROLES:
            if not call_has_been_made:
                dialogue_role = 'call'
                call_has_been_made = True
            else:
                dialogue_role = 'response'
        else:
            dialogue_role = 'none'

        print(
            f"\n{Fore.MAGENTA}--- Track {Style.BRIGHT}{Fore.YELLOW}{i + 1}/{len(config['instruments'])}{Style.RESET_ALL}{Fore.MAGENTA}"
            f": {Style.BRIGHT}{Fore.GREEN}{instrument['name']}{Style.NORMAL}"
            f" (Role: {instrument['role']})"
            f"{Style.RESET_ALL}"
        )
        
        # The generation function now returns a tuple (track_data, tokens_used) or (None, 0) on failure
        result_tuple = generate_instrument_track_data(
            config, length, instrument['name'], instrument.get('program_num', 0),
            context_tracks_for_this_theme,
            instrument['role'], i, len(config['instruments']),
            dialogue_role, theme_def['label'], theme_def['description'], previous_themes, current_theme_index
        )
        
        # --- FINAL, ROBUST CHECK ---
        if result_tuple is None or result_tuple[0] is None:
            print(Fore.RED + f"Failed to generate track for {instrument['name']}. Stopping generation for this theme." + Style.RESET_ALL)
            return False, None, 0
        
        # If the check passes, we can safely unpack
        track_data, tokens_used = result_tuple
        
        total_tokens += tokens_used
        generated_tracks[instrument['name']] = track_data
        context_tracks_for_this_theme.append({
            "instrument_name": instrument['name'],
            "role": instrument['role'],
            "notes": track_data['notes']
        })

    return True, {"theme_label": theme_def['label'], "tracks": generated_tracks}, total_tokens

def create_midi_from_json(song_data: Dict, config: Dict, output_file: str, time_offset_beats: float = 0.0, resolution: float = 0.1) -> bool:
    """
    Creates a MIDI file from the generated song data structure, including automations.
    Now includes a curve generation engine for smooth automation ramps.
    """
    try:
        bpm = config["bpm"]
        time_signature_beats = config["time_signature"]["beats_per_bar"]
        
        num_instrument_tracks = len(song_data["tracks"])

        # Read automation flags once
        automation_settings = config.get("automation_settings", {})
        allow_pitch_bend = automation_settings.get("use_pitch_bend", 0) == 1
        allow_cc = automation_settings.get("use_cc_automation", 0) == 1
        allow_sustain = automation_settings.get("use_sustain_pedal", 0) == 1

        # --- Soft Channel Capacity Check (Non-blocking Warning) ---
        try:
            NON_DRUM_ROLES = {"drums", "percussion", "kick_and_snare"}
            non_drum_count = sum(1 for t in song_data.get("tracks", []) if t.get("role", "complementary") not in NON_DRUM_ROLES)
            # There are 16 MIDI channels (0-15). Channel 9 (10th) is typically used for drums.
            # So there are 15 melodic channels available without channel sharing.
            if non_drum_count > 15:
                print(
                    Fore.YELLOW
                    + f"Warning: {non_drum_count} non-drum tracks in this part exceed 15 available melodic channels. "
                    + "Some tracks will share channels, and channel-wide automation (pitch bend/CC/sustain) may affect multiple tracks."
                    + Style.RESET_ALL
                )
        except Exception:
            # Do not block MIDI writing due to a warning failure
            pass
        midi_file = MIDIFile(num_instrument_tracks + 1, removeDuplicates=True, deinterleave=False)
        
        tempo_track = 0
        midi_file.addTempo(track=tempo_track, time=0, tempo=bpm)
        midi_file.addTimeSignature(track=tempo_track, time=0, numerator=time_signature_beats, denominator=4, clocks_per_tick=24)

        next_melodic_channel = 0
        for i, track_data in enumerate(song_data["tracks"]):
            track_name = get_instrument_name(track_data)
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
            
            # --- NEW: Process TRACK-LEVEL (Sound Design) Automations ---
            DRUM_ROLES = {"drums", "percussion", "kick_and_snare"}
            if "track_automations" in track_data and role not in DRUM_ROLES:
                # Process Pitch Bend
                if allow_pitch_bend and "pitch_bend" in track_data["track_automations"]:
                    for pb in track_data["track_automations"]["pitch_bend"]:
                        if pb.get("type") == "curve":
                            pb_start_beat = time_offset_beats + pb.get("start_beat", 0)
                            pb_end_beat = time_offset_beats + pb.get("end_beat", 0)
                            # ... (rest of the curve logic is identical to note-based)
                            pb_start_val = int(pb.get("start_value", 0))
                            pb_end_val = int(pb.get("end_value", 0))
                            pb_bias = float(pb.get("bias", 1.0))
                            duration = pb_end_beat - pb_start_beat
                            if duration <= 0: continue
                            num_steps = int(duration / resolution)
                            if num_steps == 0:
                                num_steps = 1
                            if num_steps > MAX_AUTOMATION_STEPS:
                                num_steps = MAX_AUTOMATION_STEPS
                            for step in range(num_steps + 1):
                                t = step / num_steps
                                # Avoid 0.0 ** negative and stabilize invalid bias
                                safe_bias = pb_bias if isinstance(pb_bias, (int, float)) and pb_bias > 0 else 1.0
                                progress = 0.0 if t <= 0.0 else t ** safe_bias
                                current_val = int(pb_start_val + (pb_end_val - pb_start_val) * progress)
                                current_time = pb_start_beat + t * duration
                                midi_file.addPitchWheelEvent(midi_track_num, channel, current_time, current_val)
                            # Enforce neutral reset (0) at curve end if needed
                            if pb_end_val != 0:
                                midi_file.addPitchWheelEvent(midi_track_num, channel, pb_end_beat, 0)
                # Process CC
                if allow_cc and "cc" in track_data["track_automations"]:
                    for cc in track_data["track_automations"]["cc"]:
                        if cc.get("type") == "curve":
                            cc_start_beat = time_offset_beats + cc.get("start_beat", 0)
                            cc_end_beat = time_offset_beats + cc.get("end_beat", 0)
                            # ... (rest of the curve logic is identical)
                            cc_start_val = int(cc.get("start_value", 0))
                            cc_end_val = int(cc.get("end_value", 0))
                            cc_bias = float(cc.get("bias", 1.0))
                            cc_num = int(cc.get("cc", 0))
                            duration = cc_end_beat - cc_start_beat
                            if duration <= 0: continue
                            num_steps = int(duration / resolution)
                            if num_steps == 0:
                                num_steps = 1
                            if num_steps > MAX_AUTOMATION_STEPS:
                                num_steps = MAX_AUTOMATION_STEPS
                            for step in range(num_steps + 1):
                                t = step / num_steps
                                # Avoid 0.0 ** negative and stabilize invalid bias
                                safe_bias = cc_bias if isinstance(cc_bias, (int, float)) and cc_bias > 0 else 1.0
                                progress = 0.0 if t <= 0.0 else t ** safe_bias
                                current_val = int(cc_start_val + (cc_end_val - cc_start_val) * progress)
                                current_time = cc_start_beat + t * duration
                                midi_file.addControllerEvent(midi_track_num, channel, current_time, cc_num, current_val)
                            # Enforce neutral reset (0) at curve end if needed
                            if cc_end_val != 0:
                                midi_file.addControllerEvent(midi_track_num, channel, cc_end_beat, cc_num, 0)
            
            # --- Sustain Pedal --- (skip for drums)
            if allow_sustain and "sustain_pedal" in track_data and role not in DRUM_ROLES:
                for sustain_event in track_data["sustain_pedal"]:
                    try:
                        sustain_time = float(sustain_event["beat"]) + time_offset_beats
                        sustain_action = sustain_event["action"].lower()
                        sustain_value = 127 if sustain_action == "down" else 0
                        midi_file.addControllerEvent(midi_track_num, channel, sustain_time, 64, sustain_value)
                    except (ValueError, TypeError, KeyError) as e:
                        print(Fore.YELLOW + f"Warning: Skipping invalid sustain event in track '{track_name}': {sustain_event}. Reason: {e}" + Style.RESET_ALL)
            
            for note in track_data["notes"]:
                try:
                    pitch = int(note["pitch"])
                    start_beat = float(note["start_beat"])
                    duration_beats = float(note["duration_beats"])
                    velocity = int(note["velocity"])
                    
                    if 0 <= pitch <= 127 and 1 <= velocity <= 127 and duration_beats > 0:
                        midi_file.addNote(
                            track=midi_track_num,
                            channel=channel,
                            pitch=pitch,
                            time=start_beat + time_offset_beats,
                            duration=duration_beats,
                            volume=velocity
                        )
                    
                    # --- NEW: Process Automations ---
                    if "automations" in note and (allow_pitch_bend or allow_cc):
                        # --- CURVE ENGINE FOR PITCH BEND ---
                        if allow_pitch_bend and "pitch_bend" in note["automations"]:
                            for pb in note["automations"]["pitch_bend"]:
                                if pb.get("type") == "curve":
                                    pb_start_beat = start_beat + time_offset_beats + pb.get("start_beat", 0)
                                    pb_end_beat = start_beat + time_offset_beats + pb.get("end_beat", 0)
                                    pb_start_val = int(pb.get("start_value", 0))
                                    pb_end_val = int(pb.get("end_value", 0))
                                    shape = pb.get("shape", "exponential") # Default to old bias method

                                    duration = pb_end_beat - pb_start_beat
                                    if duration <= 0: continue
                                    
                                    num_steps = int(duration / resolution)
                                    if num_steps == 0: num_steps = 1
                                    if num_steps > MAX_AUTOMATION_STEPS:
                                        num_steps = MAX_AUTOMATION_STEPS

                                    for step in range(num_steps + 1):
                                        t = step / num_steps
                                        
                                        progress = 0
                                        if shape == "s_curve":
                                            # Use a cosine-based S-curve for smooth ease-in and ease-out
                                            progress = (1 - math.cos(t * math.pi)) / 2
                                        else: # Default to exponential
                                            pb_bias = float(pb.get("bias", 1.0))
                                            # Avoid 0.0 ** negative and stabilize invalid bias
                                            safe_bias = pb_bias if pb_bias > 0 else 1.0
                                            progress = 0.0 if t <= 0.0 else t ** safe_bias

                                        current_val = int(pb_start_val + (pb_end_val - pb_start_val) * progress)
                                        current_time = pb_start_beat + t * duration
                                        midi_file.addPitchWheelEvent(midi_track_num, channel, current_time, current_val)
                                    # Enforce neutral reset (0) at curve end if needed
                                    if pb_end_val != 0:
                                        midi_file.addPitchWheelEvent(midi_track_num, channel, pb_end_beat, 0)
                                else: # Fallback for single points (legacy)
                                    pb_time = start_beat + time_offset_beats + pb.get("beat", 0)
                                    pb_value = int(pb.get("value", 0))
                                    midi_file.addPitchWheelEvent(midi_track_num, channel, pb_time, pb_value)

                        # --- CURVE ENGINE FOR CC ---
                        if allow_cc and "cc" in note["automations"]:
                            for cc in note["automations"]["cc"]:
                                if cc.get("type") == "curve":
                                    cc_start_beat = start_beat + time_offset_beats + cc.get("start_beat", 0)
                                    cc_end_beat = start_beat + time_offset_beats + cc.get("end_beat", 0)
                                    cc_start_val = int(cc.get("start_value", 0))
                                    cc_end_val = int(cc.get("end_value", 0))
                                    cc_num = int(cc.get("cc", 0))
                                    shape = cc.get("shape", "exponential")

                                    duration = cc_end_beat - cc_start_beat
                                    if duration <= 0: continue

                                    num_steps = int(duration / resolution)
                                    if num_steps == 0: num_steps = 1
                                    if num_steps > MAX_AUTOMATION_STEPS:
                                        num_steps = MAX_AUTOMATION_STEPS
                                    
                                    for step in range(num_steps + 1):
                                        t = step / num_steps
                                        
                                        progress = 0
                                        if shape == "s_curve":
                                            progress = (1 - math.cos(t * math.pi)) / 2
                                        else: # Default to exponential
                                            cc_bias = float(cc.get("bias", 1.0))
                                            # Avoid 0.0 ** negative and stabilize invalid bias
                                            safe_bias = cc_bias if cc_bias > 0 else 1.0
                                            progress = 0.0 if t <= 0.0 else t ** safe_bias

                                        current_val = int(cc_start_val + (cc_end_val - cc_start_val) * progress)
                                        current_time = cc_start_beat + t * duration
                                        midi_file.addControllerEvent(midi_track_num, channel, current_time, cc_num, current_val)
                                    # Enforce neutral reset (0) at curve end if needed
                                    if cc_end_val != 0:
                                        midi_file.addControllerEvent(midi_track_num, channel, cc_end_beat, cc_num, 0)
                                else: # Fallback for single points (legacy)
                                    cc_time = start_beat + time_offset_beats + cc.get("beat", 0)
                                    cc_num = int(cc.get("cc", 0))
                                    cc_val = int(cc.get("value", 0))
                                    midi_file.addControllerEvent(midi_track_num, channel, cc_time, cc_num, cc_val)

                except (ValueError, TypeError) as e:
                    print(Fore.YELLOW + f"Warning: Skipping invalid note/automation data in track '{track_name}': {note}. Reason: {e}" + Style.RESET_ALL)

        with open(output_file, "wb") as f:
            midi_file.writeFile(f)
            
        print(Fore.GREEN + f"\nSuccessfully created MIDI file: {output_file}" + Style.RESET_ALL)
        return True

    except Exception as e:
        print(Fore.RED + f"Error creating MIDI file: {str(e)}" + Style.RESET_ALL)
        return False

def _clamp_track_to_section_length(track: Dict, section_length_beats: float) -> Dict:
    """Returns a copy of track with all events clamped to [0, section_length_beats].
    - Notes beyond the section are dropped; notes crossing the end are shortened.
    - Sustain pedal intervals (start_beat/end_beat) are converted to down/up events.
    - Track-level automation curves are clipped to the section.
    """
    try:
        clamped = {
            "instrument_name": get_instrument_name(track),
            "program_num": track.get("program_num", 0),
            "role": track.get("role", "complementary"),
            "notes": []
        }

        # Clamp notes
        for note in track.get("notes", []):
            try:
                start = float(note.get("start_beat", 0))
                dur = float(note.get("duration_beats", 0))
                vel = int(note.get("velocity", 0))
                if start >= section_length_beats:
                    continue
                end_time = start + max(0.0, dur)
                if end_time > section_length_beats:
                    dur = max(0.0, section_length_beats - start)
                if dur <= 0 or vel < 1:
                    continue
                new_note = dict(note)
                new_note["start_beat"] = start
                new_note["duration_beats"] = dur
                clamped["notes"].append(new_note)
            except Exception:
                # Skip malformed notes silently
                continue

        # Convert/Clamp sustain events
        sustain_out = []
        for ev in track.get("sustain_pedal", []):
            try:
                if "action" in ev and "beat" in ev:
                    b = float(ev.get("beat", 0))
                    if 0 <= b <= section_length_beats:
                        sustain_out.append({"beat": b, "action": ev.get("action", "down").lower()})
                elif "start_beat" in ev and "end_beat" in ev:
                    sb = float(ev.get("start_beat", 0))
                    eb = float(ev.get("end_beat", 0))
                    if eb > sb:
                        down_b = max(0.0, sb)
                        up_b = min(section_length_beats, eb)
                        if down_b <= section_length_beats and up_b >= 0 and up_b > down_b:
                            sustain_out.append({"beat": down_b, "action": "down"})
                            sustain_out.append({"beat": up_b, "action": "up"})
            except Exception:
                continue
        if sustain_out:
            clamped["sustain_pedal"] = sustain_out

        # Track-level automations
        if "track_automations" in track and isinstance(track.get("track_automations"), dict):
            ta = {k: list(v) for k, v in track["track_automations"].items() if isinstance(v, list)}
            pb_out = []
            for pb in ta.get("pitch_bend", []):
                try:
                    sb = float(pb.get("start_beat", 0))
                    eb = float(pb.get("end_beat", 0))
                    if eb <= sb:
                        continue
                    sb_clamped = max(0.0, sb)
                    eb_clamped = min(section_length_beats, eb)
                    if eb_clamped <= sb_clamped:
                        continue
                    new_pb = dict(pb)
                    new_pb["start_beat"] = sb_clamped
                    new_pb["end_beat"] = eb_clamped
                    pb_out.append(new_pb)
                except Exception:
                    continue
            cc_out = []
            for cc in ta.get("cc", []):
                try:
                    sb = float(cc.get("start_beat", 0))
                    eb = float(cc.get("end_beat", 0))
                    if eb <= sb:
                        continue
                    sb_clamped = max(0.0, sb)
                    eb_clamped = min(section_length_beats, eb)
                    if eb_clamped <= sb_clamped:
                        continue
                    new_cc = dict(cc)
                    new_cc["start_beat"] = sb_clamped
                    new_cc["end_beat"] = eb_clamped
                    cc_out.append(new_cc)
                except Exception:
                    continue
            ta_out = {}
            if pb_out:
                ta_out["pitch_bend"] = pb_out
            if cc_out:
                ta_out["cc"] = cc_out
            if ta_out:
                clamped["track_automations"] = ta_out

        return clamped
    except Exception:
        return track

def create_part_midi_from_theme(theme_data: Dict, config: Dict, output_file: str, time_offset_beats: float = 0.0, resolution: float = 0.1, section_length_beats: float | None = None) -> bool:
    """
    Creates a MIDI file for a single theme (part) from its track data, including automations.
    It subtracts the provided time_offset_beats from all notes and automation events to normalize them to start at beat 0.
    """
    try:
        # Create a deep copy to avoid modifying the original data structure
        normalized_theme_data = json.loads(json.dumps(theme_data))

        # Normalize all note and automation times by subtracting the offset
        for track in normalized_theme_data["tracks"]:
            # Normalize Notes and their attached automations
            if "notes" in track:
                for note in track["notes"]:
                    note["start_beat"] = max(0, float(note.get("start_beat", 0)) - time_offset_beats)
                    if "automations" in note:
                        if "pitch_bend" in note["automations"]:
                            for pb in note["automations"]["pitch_bend"]:
                                if pb.get("type") == "curve":
                                    pb["start_beat"] = max(0, pb.get("start_beat", 0) - time_offset_beats)
                                    pb["end_beat"] = max(0, pb.get("end_beat", 0) - time_offset_beats)
                                else: # Legacy single point
                                    pb["beat"] = max(0, pb.get("beat", 0) - time_offset_beats)
                        if "cc" in note["automations"]:
                             for cc in note["automations"]["cc"]:
                                if cc.get("type") == "curve":
                                    cc["start_beat"] = max(0, cc.get("start_beat", 0) - time_offset_beats)
                                    cc["end_beat"] = max(0, cc.get("end_beat", 0) - time_offset_beats)
                                else: # Legacy single point
                                    cc["beat"] = max(0, cc.get("beat", 0) - time_offset_beats)

            # Normalize Sustain Pedal Events
            if "sustain_pedal" in track:
                for event in track["sustain_pedal"]:
                    event["beat"] = max(0, float(event.get("beat", 0)) - time_offset_beats)
            
            # Normalize TRACK-LEVEL Automations
            if "track_automations" in track:
                if "pitch_bend" in track["track_automations"]:
                    for pb in track["track_automations"]["pitch_bend"]:
                        if pb.get("type") == "curve":
                            pb["start_beat"] = max(0, pb.get("start_beat", 0) - time_offset_beats)
                            pb["end_beat"] = max(0, pb.get("end_beat", 0) - time_offset_beats)
                if "cc" in track["track_automations"]:
                    for cc in track["track_automations"]["cc"]:
                        if cc.get("type") == "curve":
                            cc["start_beat"] = max(0, cc.get("start_beat", 0) - time_offset_beats)
                            cc["end_beat"] = max(0, cc.get("end_beat", 0) - time_offset_beats)

        # Clamp to section length if provided
        if isinstance(section_length_beats, (int, float)) and section_length_beats is not None:
            clamped_tracks = []
            for tr in normalized_theme_data.get("tracks", []):
                clamped_tracks.append(_clamp_track_to_section_length(tr, float(section_length_beats)))
            normalized_theme_data["tracks"] = clamped_tracks

        # Call the main MIDI creation function with the now fully normalized data and a zero offset
        return create_midi_from_json(normalized_theme_data, config, output_file, time_offset_beats=0, resolution=resolution)

    except Exception as e:
        print(Fore.RED + f"Error creating MIDI part file '{output_file}': {str(e)}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return False

def get_excel_column_name(n):
    """Converts a zero-based integer to an Excel-style column name (A, B, ..., Z, AA, AB, ...)."""
    name = ""
    while n >= 0:
        name = chr(n % 26 + 65) + name
        n = n // 26 - 1
    return name

def generate_filename(config: Dict, base_dir: str, length_bars: int, theme_label: str, theme_index: int, timestamp: str) -> str:
    """
    Generates a descriptive and valid filename for a theme, supporting more than 26 parts.
    """
    try:
        genre = config.get("genre", "audio").replace(" ", "_").replace("/", "-")
        key = config.get("key_scale", "").replace(" ", "").replace("#", "s")
        bpm = round(float(config.get("bpm", 120)))
        
        # NEW: Use a robust naming system for theme parts (A, B, ..., Z, AA, AB, ...)
        theme_part_name = get_excel_column_name(theme_index)

        # Sanitize parts for filename
        genre = re.sub(r'[\\*?:"<>|]', "", genre)
        key = re.sub(r'[\\*?:"<>|]', "", key)
        sanitized_label = re.sub(r'[\s/\\:*?"<>|]+', '_', theme_label)

        # Construct the new, descriptive filename
        # Format: PartName_Label_Genre_Key_Length_BPM_Timestamp.mid
        new_name = f"{theme_part_name}_Part_{sanitized_label}_{genre}_{key}_{length_bars}bars_{bpm}bpm_{timestamp}.mid"
        
        return os.path.join(base_dir, new_name)
    except Exception as e:
        theme_part_name = get_excel_column_name(theme_index)
        print(Fore.YELLOW + f"Could not generate dynamic filename. Using default. Reason: {e}" + Style.RESET_ALL)
        return os.path.join(base_dir, f"theme_{theme_part_name}_{timestamp}.mid")

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

def main():
    """
    Main function to run the music theme generation process.
    """
    # Initialize Colorama for console color support
    init(autoreset=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    settings_file = os.path.join(script_dir, "song_settings.json")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Song Generator: Create or resume music generation.")
    parser.add_argument('--run', action='store_true', help="Run generation automatically with last settings.")
    parser.add_argument('--optimize', action='store_true', help="Optimize the last generated song.")
    parser.add_argument('--resume-file', type=str, help="Path to a progress file to resume directly.")
    parser.add_argument('--resume', type=str, help="Alias for --resume-file (compatibility with older callers).")
    parser.add_argument('user_opt_prompt', nargs='*', help="Optional user prompt for optimization.")

    args = parser.parse_args()
    
    # --- Load Base Config and API Key first ---
    try:
        config = load_config(CONFIG_FILE)
        if not initialize_api_keys(config):
            print(Fore.YELLOW + "Warning: No valid API key found. API calls will fail." + Style.RESET_ALL)
        else:
            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
    except (ValueError, FileNotFoundError) as e:
        print(Fore.RED + f"A critical error occurred on startup: {str(e)}" + Style.RESET_ALL)
        return

    # --- Validate and apply performance settings ---
    if "context_window_size" not in config:
        config["context_window_size"] = -1
    if "max_output_tokens" not in config:
        config["max_output_tokens"] = 8192

    # --- Direct Action Mode: --run ---
    if args.run:
        print_header("AUTO-GENERATION MODE")
        generated_themes, final_song_data, final_song_basename = None, None, None
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            length = settings.get('length')
            theme_definitions = settings.get('theme_definitions')
            if not length or not theme_definitions:
                raise ValueError("'length' or 'theme_definitions' missing from song_settings.json")

            print(f"Loaded {len(theme_definitions)} parts, each {length} bars long.")
            
            run_timestamp = time.strftime("%Y%m%d-%H%M%S")
            generated_themes, total_tokens = generate_all_themes_and_save_parts(config, length, theme_definitions, script_dir, run_timestamp)

            if generated_themes:
                print(Fore.GREEN + "\n--- Auto-Generation Complete! ---" + Style.RESET_ALL)
                print(Fore.CYAN + f"Total tokens used for this run: {total_tokens:,}" + Style.RESET_ALL)
                final_song_data, final_song_basename = combine_and_save_final_song(config, generated_themes, script_dir, run_timestamp)
                try:
                    prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                    if os.path.exists(prog_file): os.remove(prog_file)
                    print(Fore.GREEN + "Generation finished. Progress file removed." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
            else:
                 print(Fore.RED + "Auto-generation failed.")

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(Fore.RED + f"Could not start automatic generation: {e}")
            print(Fore.YELLOW + "Please ensure 'song_settings.json' is valid or run the interactive setup.")
        
        # After the run, proceed to the interactive menu with the results
        with open(settings_file, 'r') as f:
            previous_settings = json.load(f)
        interactive_main_menu(config, previous_settings, script_dir, generated_themes, final_song_data, final_song_basename)
        return # Exit after the interactive menu is closed by the user

    # --- Direct Action Mode: --resume-file / --resume ---
    if args.resume_file or args.resume:
        resume_path = args.resume_file or args.resume
        print(Fore.CYAN + f"\n--- Direct Resume Mode: {os.path.basename(resume_path)} ---" + Style.RESET_ALL)
        # Pass the results of the resume directly into the interactive menu
        resume_result = handle_resume(resume_path, script_dir)
        if not resume_result:
            return
        try:
            if len(resume_result) == 4:
                resumed_themes, resumed_song_data, resumed_basename, resumed_settings = resume_result
                interactive_main_menu(config, resumed_settings, script_dir, resumed_themes, resumed_song_data, resumed_basename)
            elif len(resume_result) == 3:
                resumed_themes, resumed_song_data, resumed_basename = resume_result
                interactive_main_menu(config, None, script_dir, resumed_themes, resumed_song_data, resumed_basename)
            else:
                # Fallback: attempt to unpack first three
                resumed_themes, resumed_song_data, resumed_basename = resume_result[:3]
                interactive_main_menu(config, None, script_dir, resumed_themes, resumed_song_data, resumed_basename)
        except Exception as e:
            print(Fore.RED + f"Resume handling failed: {e}" + Style.RESET_ALL)
            return
    
    # --- Interactive Mode (Default if no direct action is specified) ---
    print(Fore.MAGENTA + "="*60)
    print(Style.BRIGHT + "        ⚙️ Song Generator - Interactive Mode ⚙️")
    print(Fore.MAGENTA + "="*60 + Style.RESET_ALL)
    
    print(f"\n{Style.BRIGHT}{Fore.CYAN}Welcome to the Generation Engine!{Style.RESET_ALL}")
    print("This script reads your configuration and brings your music to life.")
    print("From here, you can:")
    print("  - Generate music using your last defined structure.")
    print("  - Define a new song structure from scratch.")
    print("  - Optimize a previously generated song for better results.\n")
    
    print(f"{Fore.YELLOW}Note: The 'Generate New Song' option will overwrite 'song_settings.json'.{Style.RESET_ALL}\n")

    try:
        # Config is already loaded
        with open(settings_file, 'r') as f:
            previous_settings = json.load(f)
        print(Fore.CYAN + "Loaded previous song settings from file." + Style.RESET_ALL)
    except (FileNotFoundError, json.JSONDecodeError):
        previous_settings = {}
    
    interactive_main_menu(config, previous_settings, script_dir)

def handle_resume(resume_file_path, script_dir):
    """Handles the logic for resuming a generation or optimization from a file."""
    progress_data = load_progress(resume_file_path)
    if not progress_data:
        print(Fore.RED + f"Failed to load progress data from {resume_file_path}" + Style.RESET_ALL)
        return None, None, None, None

    config = progress_data.get('config')
    if not config:
        print(Fore.RED + "Config data not found in progress file. Cannot resume." + Style.RESET_ALL)
        return None, None, None, None

    try:
        current_config = load_config(CONFIG_FILE)
        # Define all settings that should be refreshed from the live config on resume
        keys_to_update = [
            "api_key", "model_name", "temperature",
            "context_window_size", "max_output_tokens",
            "automation_settings", "use_call_and_response"
        ]

        print(Fore.CYAN + "\nUpdating runtime settings from current 'config.yaml'..." + Style.RESET_ALL)

        updated_any = False
        for key in keys_to_update:
            if key in current_config and config.get(key) != current_config[key]:
                old_val = config.get(key, 'Not Set')
                new_val = current_config[key]
                print(f"  - Updating '{key}': '{old_val}' -> '{new_val}'")
                config[key] = new_val
                updated_any = True

        if not updated_any:
            print("  - No runtime settings have changed in 'config.yaml'. Using settings from saved session.")

        print()
        # FIX: Re-initialize API keys with the (potentially) updated config from the current file
        if initialize_api_keys(config):
            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
            print(Fore.CYAN + f"Resume mode re-initialized with API key #{CURRENT_KEY_INDEX + 1}.")
        else:
            print(Fore.RED + "Could not initialize API keys from config during resume. API calls will likely fail.")

    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not reload current settings. Using all settings from progress file. Reason: {e}" + Style.RESET_ALL)
        # Fallback: Try to initialize keys from the config stored in the progress file
        if initialize_api_keys(config):
            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
        else:
            print(Fore.RED + "CRITICAL: Could not initialize API keys from progress file either. Aborting.")
            return None, None, None, None

    run_timestamp = progress_data.get('timestamp')
    if not run_timestamp:
        print(Fore.YELLOW + "Timestamp missing. Attempting to extract from filename..." + Style.RESET_ALL)
        match = re.search(r'(\d{8}-\d{6})', os.path.basename(resume_file_path))
        if match:
            run_timestamp = match.group(1)
            print(Fore.GREEN + f"Extracted timestamp: {run_timestamp}" + Style.RESET_ALL)
        else:
            print(Fore.RED + "Could not extract timestamp. Cannot resume." + Style.RESET_ALL)
            return None, None, None, None

    if 'generation' in progress_data.get('type', ''):
        length, defs = progress_data['length'], progress_data['theme_definitions']
        generated_themes, total_tokens = generate_all_themes_and_save_parts(config, length, defs, script_dir, run_timestamp, progress_data)

        if generated_themes:
            time.sleep(2)
            final_song_data, final_song_basename = combine_and_save_final_song(config, generated_themes, script_dir, run_timestamp)

            if final_song_data:
                print(Fore.GREEN + "\n--- Resumed Generation Complete! ---" + Style.RESET_ALL)
                print(Fore.CYAN + f"Total tokens used for this run: {total_tokens:,}" + Style.RESET_ALL)

                try:  # Clean up progress file
                    prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                    if os.path.exists(prog_file):
                        os.remove(prog_file)
                        print(Fore.GREEN + "Progress file removed." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)

                # Return the results so the main loop can offer optimization
                settings = {'length': progress_data['length'], 'theme_definitions': progress_data['theme_definitions']}
                return generated_themes, final_song_data, final_song_basename, settings
            else:
                print(Fore.RED + "Could not combine resumed parts into final song." + Style.RESET_ALL)
        else:
            print(Fore.RED + "Resumed generation failed to produce themes." + Style.RESET_ALL)

    # (removed duplicate 'window_optimization' branch)
    elif 'automation_enhancement' in progress_data.get('type', ''):
        print_header("Resume Automation Enhancement")
        try:
            theme_len = int(progress_data.get('theme_length', DEFAULT_LENGTH))
            themes_to_use = progress_data.get('themes', [])
            if not themes_to_use:
                print(Fore.RED + "No themes stored in progress. Cannot resume automation enhancement." + Style.RESET_ALL)
                return None, None, None, None
            print(Fore.CYAN + f"Resuming automation enhancement at theme {progress_data.get('current_theme_index',0)+1}, track {progress_data.get('current_track_index',0)+1}." + Style.RESET_ALL)
            resumed = create_automation_enhancement(config, theme_len, themes_to_use, script_dir, run_timestamp, progress_data.get('user_optimization_prompt',''))
            if resumed:
                save_final_artifact(config, resumed, theme_len, progress_data.get('theme_definitions', []), script_dir, run_timestamp)
                settings = {'length': theme_len, 'theme_definitions': progress_data.get('theme_definitions', [])}
                final_base = build_final_song_basename(config, resumed, run_timestamp, resumed=True)
                return resumed, merge_themes_to_song_data(resumed, config, theme_len), final_base, settings
            else:
                print(Fore.RED + "Resumed automation enhancement did not produce themes." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Automation enhancement resume failed: {e}" + Style.RESET_ALL)
    elif 'optimization' in progress_data.get('type', ''):
        print_header("Resume Optimization")
        # Backward/compatibility: some files may miss 'themes_to_optimize' (e.g., windowed schema)
        if 'themes_to_optimize' not in progress_data and 'themes' in progress_data:
            # Treat as windowed optimization fallback
            try:
                theme_len = int(progress_data.get('theme_length', DEFAULT_LENGTH))
                window_bars = int(progress_data.get('window_bars', 16))
                resume_start_index = int(progress_data.get('current_window_start_index', 0))
                user_opt_prompt = progress_data.get('user_optimization_prompt', "")
                themes_to_use = progress_data.get('themes', [])
                print(Fore.CYAN + "Detected windowed-optimization style progress. Redirecting resume..." + Style.RESET_ALL)
                resumed_themes = create_windowed_optimization(
                    config, themes_to_use, theme_len, window_bars, script_dir, run_timestamp,
                    user_optimization_prompt=user_opt_prompt, resume_start_index=resume_start_index
                )
                if resumed_themes:
                    save_final_artifact(config, resumed_themes, theme_len, progress_data.get('theme_definitions', []), script_dir, run_timestamp)
                    settings = {'length': theme_len, 'theme_definitions': progress_data.get('theme_definitions', [])}
                    final_base = build_final_song_basename(config, resumed_themes, run_timestamp, resumed=True)
                    return resumed_themes, merge_themes_to_song_data(resumed_themes, config, theme_len), final_base, settings
            except Exception:
                pass
        theme_len, themes_to_opt, opt_iter = progress_data['theme_length'], progress_data['themes_to_optimize'], progress_data['opt_iteration_num']

        print("Resuming the process to refine the last generated song, track by track.")
        user_opt_prompt = progress_data.get('user_optimization_prompt', "")
        if user_opt_prompt:
            print(Fore.CYAN + f"\nApplying user's creative direction: '{user_opt_prompt}'" + Style.RESET_ALL)
        else:
            print(Fore.CYAN + "\nRunning a general enhancement pass." + Style.RESET_ALL)

        optimized_themes = create_song_optimization(
            config, theme_len, themes_to_opt, script_dir, opt_iter,
            run_timestamp, user_opt_prompt, progress_data
        )

        if optimized_themes:
            print(Fore.GREEN + "\n--- Resumed Optimization Complete! ---" + Style.RESET_ALL)
            # Prefer live accumulated tokens if available; fall back to any saved counters
            final_tokens = 0
            try:
                # Some runs saved 'total_optimization_tokens', others 'total_tokens_used'
                final_tokens = progress_data.get('total_optimization_tokens')
                if final_tokens is None:
                    final_tokens = progress_data.get('total_tokens_used', 0)
            except Exception:
                final_tokens = 0
            print(Fore.CYAN + f"Total tokens used for this run: {final_tokens:,}" + Style.RESET_ALL)
            # Also return the result to potentially continue optimizing
            settings = {'length': progress_data['theme_length'], 'theme_definitions': progress_data.get('theme_definitions', [])}
            # Remove progress file now that optimization completed
            try:
                prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                if os.path.exists(prog_file):
                    os.remove(prog_file)
                    print(Fore.GREEN + "Progress file removed." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
            # Build a consistent final basename
            final_base = build_final_song_basename(config, optimized_themes, run_timestamp, resumed=True, opt_iteration=opt_iter)
            return optimized_themes, merge_themes_to_song_data(optimized_themes, config, theme_len), final_base, settings
        else:
            print(Fore.RED + "Resumed optimization failed to produce themes." + Style.RESET_ALL)
    elif 'window_optimization' in progress_data.get('type', ''):
        print_header("Resume Windowed Optimization")
        try:
            theme_len = int(progress_data.get('theme_length', DEFAULT_LENGTH))
            window_bars = int(progress_data.get('window_bars', 16))
            resume_start_index = int(progress_data.get('current_window_start_index', 0))
            user_opt_prompt = progress_data.get('user_optimization_prompt', "")
            themes_to_use = progress_data.get('themes', [])
            if not themes_to_use:
                print(Fore.RED + "No themes stored in progress. Cannot resume windowed optimization." + Style.RESET_ALL)
                return None, None, None, None

            print(Fore.CYAN + f"Resuming windowed optimization at window starting part index {resume_start_index+1} ({window_bars} bars)." + Style.RESET_ALL)
            # Ensure API keys are set (already done above); run windowed optimizer
            resumed_themes = create_windowed_optimization(
                config, themes_to_use, theme_len, window_bars, script_dir, run_timestamp,
                user_optimization_prompt=user_opt_prompt, resume_start_index=resume_start_index
            )
            if resumed_themes:
                # Save new artifact and return to menu with result; don't auto-start optimization follow-ups
                save_final_artifact(config, resumed_themes, theme_len, progress_data.get('theme_definitions', []), script_dir, run_timestamp)
                print(Fore.GREEN + "\n--- Resumed Windowed Optimization Complete! ---" + Style.RESET_ALL)
                # Return 4-tuple so the menu can proceed as with generation path
                settings = {'length': theme_len, 'theme_definitions': progress_data.get('theme_definitions', [])}
                final_base = build_final_song_basename(config, resumed_themes, run_timestamp, resumed=True)
                return resumed_themes, merge_themes_to_song_data(resumed_themes, config, theme_len), final_base, settings
            else:
                print(Fore.RED + "Resumed windowed optimization did not produce themes." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Windowed resume failed: {e}" + Style.RESET_ALL)

    # Return None if any path fails
    return None, None, None, None

def interactive_main_menu(config, previous_settings, script_dir, initial_themes=None, initial_song_data=None, initial_basename=None):
    """
    The main interactive menu for the song generator.
    Allows the user to generate, optimize, or resume song creation.
    """
    last_generated_themes = initial_themes
    last_generated_song_data = initial_song_data
    final_song_basename = initial_basename
    
    # Persist the settings across the session in the menu
    session_settings = previous_settings

    while True:
        print_header("Song Generator Menu")
        
        menu_options = {}
        next_option = 1

        if session_settings:
            menu_options[str(next_option)] = ('generate_again', "Generate Again (using last settings)")
            next_option += 1
        
        menu_options[str(next_option)] = ('generate_new', "Generate New Song (define new parts)")
        next_option += 1

        if last_generated_themes:
            menu_options[str(next_option)] = ('optimize', "Optimize Last Generated Song")
            next_option += 1

        # New: Optimize existing artifacts
        artifacts = find_final_artifacts(script_dir)
        if artifacts:
            menu_options[str(next_option)] = ('optimize_artifact', "Optimize Existing Song (from artifacts)")
            next_option += 1
        menu_options[str(next_option)] = ('advanced_opt', "Advanced Optimization Options")
        next_option += 1

        progress_files = find_progress_files(script_dir)
        if progress_files:
            menu_options[str(next_option)] = ('resume', "Resume In-Progress Job")
            next_option += 1
        
        menu_options[str(next_option)] = ('quit', "Quit")

        for key, (_, text) in menu_options.items():
            print(f"{Fore.YELLOW}{key}.{Style.RESET_ALL} {text}")
        # Show hotkey hint at the menu, too
        print_hotkey_hint(config, context="Menu")
        
        user_choice_key = input(f"\n{Style.BRIGHT}{Fore.GREEN}Choose an option: {Style.RESET_ALL}").strip()
        
        action = menu_options.get(user_choice_key, (None, None))[0]

        try:
            if action == 'quit':
                print(Fore.CYAN + "Exiting. Goodbye!" + Style.RESET_ALL)
                clean_old_progress_files(script_dir)
                break
            
            elif action == 'resume':
                print_header("Resume In-Progress Job")
                if progress_files:
                    for i, pfile in enumerate(progress_files[:10]):
                        print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {summarize_progress_file(pfile)}")
                
                choice_idx = -1
                while not (0 <= choice_idx < len(progress_files[:10])):
                    try:
                        choice_str = get_user_input(f"Choose file to resume (1-{len(progress_files[:10])}):", "1")
                        choice_idx = int(choice_str) - 1
                    except ValueError:
                        pass
                
                selected_progress_file = progress_files[choice_idx]
                resumed_themes, resumed_song_data, resumed_basename, resumed_settings = handle_resume(selected_progress_file, script_dir)
                if resumed_themes:
                    # Update all state variables after a successful resume
                    last_generated_themes = resumed_themes
                    last_generated_song_data = resumed_song_data
                    final_song_basename = resumed_basename
                    if resumed_settings:
                        session_settings = resumed_settings
                continue

            elif action in ['generate_again', 'generate_new']:
                # CRITICAL: Clear previous song state before starting a new generation
                last_generated_themes = None
                last_generated_song_data = None
                final_song_basename = None

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

                    settings_file = os.path.join(script_dir, "song_settings.json")
                    print(f"\n{Fore.YELLOW}Note: This will save your new song structure to '{os.path.basename(settings_file)}'.{Style.RESET_ALL}")
                    previous_settings = {'length': length, 'theme_definitions': defs}
                    with open(settings_file, 'w') as f: 
                        json.dump(previous_settings, f, indent=4)
                    print(Fore.GREEN + "New song structure saved successfully." + Style.RESET_ALL)

                elif not previous_settings:
                    print(Fore.YELLOW + "No previous settings found. Choose 'Generate New Song' first." + Style.RESET_ALL); continue

                length, defs = previous_settings['length'], previous_settings['theme_definitions']
                run_timestamp = time.strftime("%Y%m%d-%H%M%S")
                
                generated_themes, total_tokens = generate_all_themes_and_save_parts(config, length, defs, script_dir, run_timestamp)
                
                if generated_themes:
                    print(Fore.GREEN + "\n--- Generation Complete! ---" + Style.RESET_ALL)
                    print(Fore.CYAN + f"Total tokens used for this run: {total_tokens:,}" + Style.RESET_ALL)
                    
                    # Combine parts and save the final song file
                    final_song_data, final_song_basename_val = combine_and_save_final_song(config, generated_themes, script_dir, run_timestamp)
                    
                    if final_song_data:
                        # This block updates the session state to enable the "Optimize" option
                        print(Fore.CYAN + "Updating session state with generated song..." + Style.RESET_ALL)
                        last_generated_song_data = final_song_data
                        last_generated_themes = generated_themes
                        final_song_basename = final_song_basename_val
                        
                        # Clean up progress file now that generation is successful
                        try:
                            prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                            if os.path.exists(prog_file):
                                os.remove(prog_file)
                            print(Fore.GREEN + "Generation finished. Progress file removed." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Failed to combine parts into a final song. Optimization will not be available." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Generation process failed to produce any themes. Optimization will not be available." + Style.RESET_ALL)

            elif action == 'optimize':
                if not last_generated_song_data:
                    print(Fore.YELLOW + "No song has been generated yet in this session to optimize." + Style.RESET_ALL); continue

                print_header("Optimize Song")
                print("This process will read the last generated song and apply your creative instructions to refine it, track by track.")
                print(f"{Style.DIM}You can make it 'more aggressive', 'less repetitive', 'add more complexity to the drums', etc.{Style.RESET_ALL}")
                user_opt_prompt = input(f"{Fore.CYAN}\nEnter an optional English prompt for this optimization (or press Enter to skip):\n> {Style.RESET_ALL}").strip()

                match = re.search(r'(\d{8}-\d{6})', final_song_basename)
                if not match:
                    print(Fore.RED + f"Could not get timestamp from '{final_song_basename}'. Cannot link progress." + Style.RESET_ALL); continue
                run_timestamp = match.group(1)
                
                themes_to_opt = last_generated_themes
                theme_len = session_settings.get('length', DEFAULT_LENGTH)
                
                # Normalization should happen inside the optimization loop if needed, not before.
                # This was a likely source of errors.
                # normalized_themes = normalize_themes(themes_to_opt, theme_len, config)
                # themes_to_opt = normalized_themes

                opt_iter = get_next_available_file_number(os.path.join(script_dir, final_song_basename + ".mid"))
                
                print(Fore.CYAN + f"\n--- Starting Optimization (Version {opt_iter}) ---" + Style.RESET_ALL)
                
                optimized_themes = create_song_optimization(
                    config, theme_len, themes_to_opt, script_dir, 
                    opt_iter, run_timestamp, user_opt_prompt
                )
                
                if optimized_themes:
                    time.sleep(2)
                    base_name = re.sub(r'_opt_\d+$', '', final_song_basename)
                    opt_fname = os.path.join(script_dir, f"{base_name}_opt_{opt_iter}.mid")
                    
                    final_song_data = merge_themes_to_song_data(optimized_themes, config, theme_len)
                    create_midi_from_json(final_song_data, config, opt_fname)
                    
                    last_generated_themes = optimized_themes
                    last_generated_song_data = final_song_data
                    
                    # Save meta artifact for this optimized result so it can be selected later
                    try:
                        defs_for_artifact = session_settings.get('theme_definitions', []) if session_settings else []
                        save_final_artifact(config, optimized_themes, theme_len, defs_for_artifact, script_dir, run_timestamp)
                    except Exception:
                        pass

                    # Cleanup progress file (best-effort)
                    try:
                        prog_file = os.path.join(script_dir, get_progress_filename(config, run_timestamp))
                        if os.path.exists(prog_file):
                            os.remove(prog_file)
                    except Exception as e:
                        print(Fore.YELLOW + f"Could not remove progress file: {e}" + Style.RESET_ALL)

                    print(Fore.GREEN + "\nOptimization cycle complete. Returning to menu." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Optimization failed. Returning to menu." + Style.RESET_ALL)
            
            elif action == 'optimize_artifact':
                print_header("Optimize Existing Song (Artifacts)")
                artifacts = find_final_artifacts(script_dir)
                if not artifacts:
                    print(Fore.YELLOW + "No artifacts found." + Style.RESET_ALL)
                    continue
                for i, ap in enumerate(artifacts[:10]):
                    print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {summarize_artifact(ap)}")
                try:
                    sel = input(f"{Fore.GREEN}Choose artifact to optimize (1-{min(10,len(artifacts))}): {Style.RESET_ALL}").strip()
                    idx = int(sel) - 1
                except Exception:
                    print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL)
                    continue
                if not (0 <= idx < len(artifacts[:10])):
                    print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL)
                    continue
                artifact = load_final_artifact(artifacts[idx])
                if not artifact:
                    continue
                defs = artifact.get('theme_definitions', [])
                themes_to_opt = artifact.get('themes', [])
                art_length = artifact.get('length', previous_settings.get('length', DEFAULT_LENGTH))
                if not themes_to_opt:
                    print(Fore.YELLOW + "Artifact has no themes to optimize." + Style.RESET_ALL)
                    continue
                user_opt_prompt = input(f"{Fore.CYAN}\nEnter an optional English prompt for this optimization (or press Enter to skip):\n> {Style.RESET_ALL}").strip()
                run_timestamp = time.strftime("%Y%m%d-%H%M%S")
                opt_iter = get_next_available_file_number(os.path.join(script_dir, build_final_song_basename(config, themes_to_opt, run_timestamp) + ".mid"))
                print(Fore.CYAN + f"\n--- Starting Optimization (Version {opt_iter}) ---" + Style.RESET_ALL)
                optimized_themes = create_song_optimization(
                    config, art_length, themes_to_opt, script_dir,
                    opt_iter, run_timestamp, user_opt_prompt
                )
                if optimized_themes:
                    print(Fore.GREEN + "\nOptimization complete." + Style.RESET_ALL)
                    # Save new artifact so it can be selected later (do not remove old; keep history)
                    save_final_artifact(config, optimized_themes, art_length, defs, script_dir, run_timestamp)
                else:
                    print(Fore.RED + "Optimization failed." + Style.RESET_ALL)

            elif action == 'advanced_opt':
                print_header("Advanced Optimization Options")
                artifacts = find_final_artifacts(script_dir)
                print("Select the base song: last in-session result or choose from saved artifacts.")

                base_themes = None
                base_defs = []
                theme_len = None

                if last_generated_themes and artifacts:
                    print("  Base options: 1) In-session song  2) Choose artifact")
                    bc = input(f"{Fore.GREEN}Choose base [1/2, default 1]: {Style.RESET_ALL}").strip()
                    if bc == '2':
                        # List artifacts and let the user choose one
                        for i, ap in enumerate(artifacts[:10]):
                            print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {summarize_artifact(ap)}")
                        try:
                            sel = input(f"{Fore.GREEN}Choose artifact (1-{min(10,len(artifacts))}): {Style.RESET_ALL}").strip()
                            idx = int(sel) - 1
                        except Exception:
                            idx = 0
                        if not (0 <= idx < len(artifacts[:10])):
                            idx = 0
                        artifact = load_final_artifact(artifacts[idx])
                        if artifact:
                            base_themes = artifact.get('themes', [])
                            base_defs = artifact.get('theme_definitions', [])
                            theme_len = int(artifact.get('length', previous_settings.get('length', DEFAULT_LENGTH) if previous_settings else DEFAULT_LENGTH))
                    else:
                        base_themes = last_generated_themes
                        base_defs = previous_settings.get('theme_definitions', []) if previous_settings else []
                        theme_len = previous_settings.get('length', DEFAULT_LENGTH) if previous_settings else DEFAULT_LENGTH
                elif last_generated_themes:
                    # Only in-session available
                    base_themes = last_generated_themes
                    base_defs = previous_settings.get('theme_definitions', []) if previous_settings else []
                    theme_len = previous_settings.get('length', DEFAULT_LENGTH) if previous_settings else DEFAULT_LENGTH
                elif artifacts:
                    # Only artifacts available: force selection
                    print("No in-session song available. Choose an artifact:")
                    for i, ap in enumerate(artifacts[:10]):
                        print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {summarize_artifact(ap)}")
                    try:
                        sel = input(f"{Fore.GREEN}Choose artifact (1-{min(10,len(artifacts))}): {Style.RESET_ALL}").strip()
                        idx = int(sel) - 1
                    except Exception:
                        idx = 0
                    if not (0 <= idx < len(artifacts[:10])):
                        idx = 0
                    artifact = load_final_artifact(artifacts[idx])
                    if artifact:
                        base_themes = artifact.get('themes', [])
                        base_defs = artifact.get('theme_definitions', [])
                        theme_len = int(artifact.get('length', previous_settings.get('length', DEFAULT_LENGTH) if previous_settings else DEFAULT_LENGTH))

                if not base_themes:
                    print(Fore.YELLOW + "No song available to optimize." + Style.RESET_ALL)
                    continue

                run_timestamp = time.strftime("%Y%m%d-%H%M%S")
                if theme_len is None:
                    theme_len = previous_settings.get('length', DEFAULT_LENGTH) if previous_settings else DEFAULT_LENGTH
                # Work on a deep copy to avoid accidental shared references
                themes_copy = json.loads(json.dumps(base_themes))
                # Interactive loop: choose which advanced optimization to run
                while True:
                    print(f"\n{Fore.GREEN}Choose an advanced optimization:{Style.RESET_ALL}")
                    print("  1) 16-bar Window Optimization")
                    print("     - Optimizes across 2 consecutive parts (if part length is 8 bars); focuses on longer phrases.")
                    print("  2) 32-bar Window Optimization")
                    print("     - Optimizes across 4 consecutive parts; stronger macro‑phrasing with higher context.")
                    print("  3) Automation Enhancement")
                    print("     - Adds expressive Pitch Bend/CC/Sustain (as enabled in config). Minimal note edits allowed.")
                    print("  4) Done")
                    choice = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip()
                    if choice in ['', '4', 'q', 'Q']:
                        break
                    if choice not in ['1', '2', '3', '16', '32']:
                        print(Fore.YELLOW + "Invalid choice." + Style.RESET_ALL)
                        continue
                    if choice == '3':
                        # Automation Enhancement flow
                        user_prompt = input(f"{Fore.CYAN}\nOptional prompt for Automation Enhancement (or Enter):\n> {Style.RESET_ALL}").strip()
                        # Mode selection: auto/improve/add
                        print("Choose mode: 1) Auto-detect  2) Improve existing (may add)  3) Add new")
                        mode_in = input(f"{Fore.GREEN}Mode [1/2/3, default 1]: {Style.RESET_ALL}").strip()
                        mode_map = {'1': 'auto', '2': 'improve', '3': 'add'}
                        enh_mode = mode_map.get(mode_in, 'auto')
                        themes_copy = create_automation_enhancement(config, theme_len, themes_copy, script_dir, run_timestamp, user_prompt, enhancement_mode=enh_mode)
                        save_final_artifact(config, themes_copy, theme_len, base_defs, script_dir, run_timestamp)
                        print(Fore.GREEN + "Automation enhancement pass done. Artifact saved." + Style.RESET_ALL)
                        continue

                    # Windowed optimization flow (16/32 bars)
                    window_bars = 16 if choice in ['1', '16'] else 32

                    # Ask for an optional prompt now with tailored hint
                    prompt_hint = ("32-bar window" if window_bars == 32 else "16-bar window")
                    user_opt_prompt = input(f"{Fore.CYAN}\nOptional prompt for {prompt_hint} (or Enter):\n> {Style.RESET_ALL}").strip()

                    # Resume support: detect progress for this window size and match current song
                    resume_files = find_progress_files(script_dir)
                    resume_start_index = 0
                    found_progress = False
                    matched_progress = None
                    try:
                        base_labels = [t.get('label') for t in themes_copy]
                        base_parts = len(base_labels)
                        base_cfg = { 'genre': config.get('genre'), 'bpm': config.get('bpm'), 'key_scale': config.get('key_scale') }
                        for rf in resume_files:
                            pdata = _load_progress_silent(rf)
                            if not pdata: continue
                            if pdata.get('type') != 'window_optimization': continue
                            if pdata.get('window_bars') != window_bars: continue
                            pthemes = pdata.get('themes', [])
                            plabels = [t.get('label') for t in pthemes]
                            pcfg = pdata.get('config', {})
                            same_song = (len(plabels) == base_parts and plabels == base_labels and 
                                         str(pcfg.get('genre')) == str(base_cfg['genre']) and 
                                         str(pcfg.get('bpm')) == str(base_cfg['bpm']) and 
                                         str(pcfg.get('key_scale')) == str(base_cfg['key_scale']))
                            if same_song:
                                matched_progress = pdata
                                break
                        if matched_progress:
                            try:
                                resume_start_index = int(matched_progress.get('current_window_start_index', 0))
                            except Exception:
                                resume_start_index = 0
                            ts = matched_progress.get('timestamp', 'unknown time')
                            print(Fore.CYAN + f"Resume available for this window (start part index {resume_start_index+1}, saved {ts})." + Style.RESET_ALL)
                            ans = input(f"{Fore.GREEN}Resume from there (r) or start from part 1 (s)? [s]: {Style.RESET_ALL}").strip().lower()
                            if ans == 'r':
                                found_progress = True
                            else:
                                resume_start_index = 0
                                found_progress = False
                    except Exception:
                        resume_start_index = 0
                        found_progress = False

                    themes_copy = create_windowed_optimization(
                        config, themes_copy, theme_len, window_bars, script_dir, run_timestamp,
                        user_optimization_prompt=user_opt_prompt, resume_start_index=resume_start_index if found_progress else 0
                    )

                    # Save result as artifact after each pass
                    save_final_artifact(config, themes_copy, theme_len, base_defs, script_dir, run_timestamp)
                    label = f"{window_bars}-bar Window"
                    print(Fore.GREEN + f"{label} optimization pass done. Artifact saved." + Style.RESET_ALL)
            
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
                min_start_beat = min(n['start_beat'] for n in second_theme_notes if isinstance(n, dict) and 'start_beat' in n)
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
    Merges themes into a final song and corrects timing:
    - Detects per theme/track whether notes are relative (0..part) or absolute (offset..offset+part)
    - Applies appropriate offsets, clamps notes to part boundaries and discards out-of-range notes
    - Transfers sustain pedal events and track automations with offsets and clamping
    """
    merged_tracks: Dict[str, Dict] = {}
    instrument_order = [inst['name'] for inst in config['instruments']]
    theme_length_beats = theme_length_bars * config["time_signature"]["beats_per_bar"]

    def is_absolute_timing(notes: List[Dict], theme_index: int) -> bool:
        if not notes:
            return False
        try:
            offset = theme_index * theme_length_beats
            min_start = min(float(n.get('start_beat', 0)) for n in notes if isinstance(n, dict) and 'start_beat' in n)
            # Wenn die kleinste Startzeit im oder nach dem erwarteten Offset liegt, interpretieren wir als absolut
            return min_start >= max(0.0, offset - 1e-6)
        except Exception:
            return False

    def add_note_clamped(dest_list: List[Dict], note: Dict, abs_start: float, duration: float, part_start: float, part_end: float):
        end_time = abs_start + max(0.0, duration)
        if abs_start >= part_end:
            return
        if end_time > part_end:
            duration = max(0.0, part_end - abs_start)
        if duration <= 0:
            return
        new_note = dict(note)
        new_note['start_beat'] = abs_start
        new_note['duration_beats'] = duration
        dest_list.append(new_note)

    for theme_index, theme in enumerate(themes):
        part_start = theme_index * theme_length_beats
        part_end = part_start + theme_length_beats
        for track in theme.get('tracks', []):
            name = get_instrument_name(track)
            role = track.get('role', 'complementary')
            if name not in merged_tracks:
                merged_tracks[name] = {
                    'instrument_name': name,
                    'program_num': track.get('program_num', 0),
                    'role': role,
                    'notes': []
                }

            dest = merged_tracks[name]

            # 1) Merge notes
            notes = track.get('notes', []) or []
            abs_mode = is_absolute_timing(notes, theme_index)
            for note in notes:
                try:
                    start = float(note.get('start_beat', 0))
                    dur = float(note.get('duration_beats', 0))
                    if abs_mode:
                        abs_start = start
                    else:
                        abs_start = start + part_start
                    add_note_clamped(dest['notes'], note, abs_start, dur, part_start, part_end)
                except Exception:
                    continue

            # 2) Transfer sustain pedal events (event or interval form)
            sustain_src = track.get('sustain_pedal', []) or []
            if sustain_src:
                dest.setdefault('sustain_pedal', [])
                # Heuristic: if no notes present, infer timing mode based on sustain events
                abs_mode_sus = abs_mode
                if notes == [] and sustain_src:
                    try:
                        min_b = 1e9
                        for ev in sustain_src:
                            if 'beat' in ev:
                                min_b = min(min_b, float(ev.get('beat', 0)))
                            elif 'start_beat' in ev:
                                min_b = min(min_b, float(ev.get('start_beat', 0)))
                        abs_mode_sus = min_b >= max(0.0, part_start - 1e-6)
                    except Exception:
                        abs_mode_sus = False
                for ev in sustain_src:
                    try:
                        if 'beat' in ev:
                            b = float(ev.get('beat', 0))
                            abs_b = b if abs_mode_sus else b + part_start
                            if part_start <= abs_b <= part_end:
                                dest['sustain_pedal'].append({
                                    'beat': abs_b,
                                    'action': str(ev.get('action', 'down')).lower()
                                })
                        elif 'start_beat' in ev and 'end_beat' in ev:
                            sb = float(ev.get('start_beat', 0))
                            eb = float(ev.get('end_beat', 0))
                            if eb > sb:
                                abs_sb = sb if abs_mode_sus else sb + part_start
                                abs_eb = eb if abs_mode_sus else eb + part_start
                                # Clamp to part boundaries
                                abs_sb = max(part_start, abs_sb)
                                abs_eb = min(part_end, abs_eb)
                                if abs_eb > abs_sb:
                                    dest['sustain_pedal'].append({'beat': abs_sb, 'action': 'down'})
                                    dest['sustain_pedal'].append({'beat': abs_eb, 'action': 'up'})
                    except Exception:
                        continue

            # 3) Transfer track automations (curves)
            ta = track.get('track_automations', {}) or {}
            if isinstance(ta, dict) and (ta.get('pitch_bend') or ta.get('cc')):
                dest.setdefault('track_automations', {})
                # Pitch bend curves
                if isinstance(ta.get('pitch_bend'), list):
                    out_list = dest['track_automations'].setdefault('pitch_bend', [])
                    for pb in ta['pitch_bend']:
                        try:
                            sb = float(pb.get('start_beat', 0))
                            eb = float(pb.get('end_beat', 0))
                            if eb <= sb:
                                continue
                            abs_sb = sb if abs_mode else sb + part_start
                            abs_eb = eb if abs_mode else eb + part_start
                            # clamp
                            abs_sb = max(part_start, abs_sb)
                            abs_eb = min(part_end, abs_eb)
                            if abs_eb <= abs_sb:
                                continue
                            new_pb = dict(pb)
                            new_pb['start_beat'] = abs_sb
                            new_pb['end_beat'] = abs_eb
                            out_list.append(new_pb)
                        except Exception:
                            continue
                # CC curves
                if isinstance(ta.get('cc'), list):
                    out_list_cc = dest['track_automations'].setdefault('cc', [])
                    for cc in ta['cc']:
                        try:
                            sb = float(cc.get('start_beat', 0))
                            eb = float(cc.get('end_beat', 0))
                            if eb <= sb:
                                continue
                            abs_sb = sb if abs_mode else sb + part_start
                            abs_eb = eb if abs_mode else eb + part_start
                            abs_sb = max(part_start, abs_sb)
                            abs_eb = min(part_end, abs_eb)
                            if abs_eb <= abs_sb:
                                continue
                            new_cc = dict(cc)
                            new_cc['start_beat'] = abs_sb
                            new_cc['end_beat'] = abs_eb
                            out_list_cc.append(new_cc)
                        except Exception:
                            continue

    # Originale Instrument-Reihenfolge bewahren
    final_tracks_sorted = [merged_tracks[name] for name in instrument_order if name in merged_tracks]

    return {
        'bpm': config['bpm'],
        'time_signature': config['time_signature'],
        'key_scale': config['key_scale'],
        'tracks': final_tracks_sorted
    }

def get_context_for_theme(all_themes_data: List[Dict], current_theme_index: int, config: Dict) -> List[Dict]:
    """
    Determines the correct slice of previous themes to use as context
    based on the context_window_size setting in the config.
    """
    context_window_config = config.get("context_window_size", -1)
    
    if context_window_config == 0:
        return [] # No context
    elif context_window_config > 0:
        # Fixed window mode: pass a slice of the history
        start_index = max(0, current_theme_index - context_window_config)
        return all_themes_data[start_index:current_theme_index]
    else: # Dynamic mode (-1)
        # Pass the full history; the prompt generator will shrink it based on character limits
        return all_themes_data[:current_theme_index]

def generate_all_themes_and_save_parts(config, length, theme_definitions, script_dir, timestamp, resume_data=None) -> Tuple[List[Dict], int]:
    """Generates all themes, saves progress track-by-track, and saves a MIDI file for each completed theme."""
    print(Fore.CYAN + "\n--- Stage 1: Generating all individual song parts... ---" + Style.RESET_ALL)
    
    # --- Resume Logic ---
    all_themes_data = []
    start_theme_index = 0
    start_track_index = 0
    total_tokens_used = 0
    used_labels = {}

    if resume_data:
        all_themes_data = resume_data.get('all_themes_data', [])
        start_theme_index = resume_data.get('current_theme_index', 0)
        start_track_index = resume_data.get('current_track_index', 0)
        total_tokens_used = resume_data.get('total_tokens_used', 0)
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
            
            # Use the helper to get the correct context; ensure fixed window is respected inside the function
            previous_themes_context = get_context_for_theme(all_themes_data, i, config)

            print(Fore.BLUE + f"\n--- Generating Theme {Style.BRIGHT}{Fore.YELLOW}{i+1}/{len(theme_definitions)}{Style.RESET_ALL}{Fore.BLUE}: '{Style.BRIGHT}{theme_def['label']}{Style.NORMAL}' ---" + Style.RESET_ALL)
            print(f"{Style.DIM}{Fore.WHITE}Blueprint: {theme_def['description']}{Style.RESET_ALL}")

            # --- Track Loop (within the current theme) ---
            track_start_index_for_this_theme = start_track_index if i == start_theme_index else 0
            
            call_has_been_made = False
            CALL_AND_RESPONSE_ROLES = {'bass', 'chords', 'arp', 'guitar', 'lead', 'melody', 'vocal'}
            
            deferred_queue = []  # list of track indices to retry after first pass
            global DEFER_CURRENT_TRACK
            for j in range(track_start_index_for_this_theme, len(config["instruments"])):
                instrument = config["instruments"][j]
                instrument_name, program_num, role = instrument["name"], instrument["program_num"], instrument.get("role", "complementary")
                
                dialogue_role = 'none'
                if config.get("use_call_and_response") == 1 and role in CALL_AND_RESPONSE_ROLES:
                    if not call_has_been_made:
                        dialogue_role = 'call'
                        call_has_been_made = True
                    else:
                        dialogue_role = 'response'
                
                print(f"\n{Fore.MAGENTA}--- Track {Style.BRIGHT}{Fore.YELLOW}{j + 1}/{len(config['instruments'])}{Style.RESET_ALL}{Fore.MAGENTA}: {Style.BRIGHT}{Fore.GREEN}{instrument_name}{Style.RESET_ALL}")
                
                track_data, tokens_used = generate_instrument_track_data(
                    config, length, instrument_name, program_num, 
                    context_tracks_for_current_theme, role, j, len(config['instruments']), dialogue_role,
                    theme_def['label'], theme_def['description'], previous_themes_context,
                    current_theme_index=i
                )

                if track_data:
                    total_tokens_used += tokens_used
                    print(Fore.CYAN + f"Cumulative song tokens so far: {total_tokens_used:,}" + Style.RESET_ALL)
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
                        'total_themes': len(theme_definitions),
                        'total_tokens_used': total_tokens_used
                    }
                    save_progress(progress_data, script_dir, timestamp)
                else:
                    # If was user-deferred, we push it to queue explicitly
                    if DEFER_CURRENT_TRACK:
                        print(Fore.MAGENTA + f"Track '{instrument_name}' deferred; will retry after other tracks." + Style.RESET_ALL)
                        DEFER_CURRENT_TRACK = False
                        deferred_queue.append(j)
                        continue
                    print(Fore.YELLOW + f"Deferring track '{instrument_name}' for later retry within this theme." + Style.RESET_ALL)
                    deferred_queue.append(j)

            # Retry deferred tracks up to 10 rounds or until all succeed
            rounds = 0
            while deferred_queue and rounds < 10:
                rounds += 1
                j = deferred_queue.pop(0)
                instrument = config["instruments"][j]
                instrument_name, program_num, role = instrument["name"], instrument["program_num"], instrument.get("role", "complementary")
                print(Fore.MAGENTA + f"\nRetry deferred track (round {rounds}): {instrument_name}" + Style.RESET_ALL)
                # Halve historical theme context size each round (do not change notes cap)
                try:
                    cws_base = i  # number of previous themes available
                    cws_override = max(1, cws_base // (2 ** max(0, rounds-1))) if cws_base > 0 else 0
                    temp_cfg = json.loads(json.dumps(config))
                    if cws_override > 0:
                        temp_cfg["context_window_size"] = cws_override
                    else:
                        temp_cfg["context_window_size"] = 0
                    prev_ctx_decayed = get_context_for_theme(all_themes_data, i, temp_cfg)
                except Exception:
                    prev_ctx_decayed = previous_themes_context
                    temp_cfg = config
                track_data, tokens_used = generate_instrument_track_data(
                    temp_cfg, length, instrument_name, program_num,
                    context_tracks_for_current_theme, role, j, len(config['instruments']), dialogue_role,
                    theme_def['label'], theme_def['description'], prev_ctx_decayed,
                    current_theme_index=i
                )
                if track_data:
                    total_tokens_used += tokens_used
                    print(Fore.CYAN + f"Cumulative song tokens so far: {total_tokens_used:,}" + Style.RESET_ALL)
                    context_tracks_for_current_theme.append(track_data)
                else:
                    deferred_queue.append(j)
            if deferred_queue:
                print(Fore.RED + f"Still failing tracks after {rounds} rounds: {len(deferred_queue)}. Aborting generation for this theme." + Style.RESET_ALL)
                return None, total_tokens_used
            
            # --- After a theme's tracks are all generated, create its MIDI file ---
            print(Fore.GREEN + f"\n--- Theme '{theme_def['label']}' generated successfully! Saving part file... ---" + Style.RESET_ALL)
            
            # Use the original, user-provided label for the filename generation,
            # the sanitization will happen inside generate_filename.
            output_filename = generate_filename(config, script_dir, length, theme_def['label'], i, timestamp)
            current_theme_data['original_filename'] = os.path.basename(output_filename)
            
            # Create the MIDI part file for the completed theme, passing the correct time offset
            time_offset_for_this_theme = i * length * config["time_signature"]["beats_per_bar"]
            # Clamp to exact part length
            part_len_beats = length * config["time_signature"]["beats_per_bar"]
            if not create_part_midi_from_theme(current_theme_data, config, output_filename, time_offset_for_this_theme, section_length_beats=part_len_beats):
                 print(Fore.RED + f"Could not create part MIDI for {theme_def['label']}. Check for errors above." + Style.RESET_ALL)
            
            # Reset start_track_index for the next theme
            start_track_index = 0

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n--- Generation interrupted by user. Progress has been saved. ---" + Style.RESET_ALL)
        return None, total_tokens_used
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred during generation: {e}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, 0

    return all_themes_data, total_tokens_used

def combine_and_save_final_song(config, generated_themes, script_dir, timestamp):
    """Merges generated themes into a final song and saves it to a MIDI file (incl. Automationen)."""
    if not generated_themes:
        print(Fore.YELLOW + "No themes were generated, cannot create a final song." + Style.RESET_ALL)
        return None, None

    print(Fore.CYAN + "\n--- Stage 2: Combining all parts into the final song... ---" + Style.RESET_ALL)

    try:
        # Verwende die robuste Merge-Logik inkl. Sustain & Track-Automationen
        try:
            with open(os.path.join(script_dir, "song_settings.json"), 'r') as f:
                s = json.load(f)
                length_bars = int(s.get('length', DEFAULT_LENGTH)) if isinstance(s.get('length'), int) else DEFAULT_LENGTH
        except Exception:
            length_bars = DEFAULT_LENGTH

        final_song_data = merge_themes_to_song_data(generated_themes, config, length_bars)

        final_base = build_final_song_basename(config, generated_themes, timestamp)
        final_filename = os.path.join(script_dir, f"{final_base}.mid")

        # small delay to ensure clean file sorting by date
        time.sleep(3)

        create_midi_from_json(final_song_data, config, final_filename)

        # Artefakt speichern (wie zuvor)
        try:
            with open(os.path.join(script_dir, "song_settings.json"), 'r') as f:
                s = json.load(f)
                defs = s.get('theme_definitions', [])
        except Exception:
            defs = []
        save_final_artifact(config, generated_themes, length_bars, defs, script_dir, timestamp)

        return final_song_data, final_base

    except Exception as e:
        print(Fore.RED + f"Failed to create the final combined MIDI file. Reason: {e}" + Style.RESET_ALL)
        return None, None



def get_role_instructions_for_generation(role: str, config: Dict) -> str:
    """
    Returns ENHANCED, role-specific instructions for generation, conditionally
    encouraging proactive automation based on config settings.
    """
    # Get automation settings from config
    automation_settings = config.get("automation_settings", {})
    use_pitch_bend = automation_settings.get("use_pitch_bend", 0) == 1
    use_cc_automation = automation_settings.get("use_cc_automation", 0) == 1
    use_sustain_pedal = automation_settings.get("use_sustain_pedal", 0) == 1

    # 1. Define base role descriptions (universally applicable)
    role_map = {
        "drums": "**Your Role: The Rhythmic Foundation**\nCreate a strong, clear rhythmic backbone that defines the song's energy. Use varying velocities to create a human-like groove.",
        "kick_and_snare": f"**Your Role: The Core Beat**\nCreate the main kick and snare pattern for {config['genre']}. This is the fundamental pulse of the track.",
        "percussion": "**Your Role: Rhythmic Texture**\nAdd secondary percussion that complements the main drums and adds rhythmic interest.",
        "bass": "**Your Role: The Groove Foundation**\nCreate a rhythmic bassline that locks with the kick and provides a clear harmonic foundation.",
        "sub_bass": "**Your Role: The Low-End Anchor**\nProvide a clean, powerful low-end foundation using very low notes. The rhythm should be simple and lock in with the kick drum.",
        "pads": "**Your Role: Harmonic Atmosphere**\nProvide the main harmonic foundation with sustained chords and atmospheric textures.",
        "atmosphere": "**Your Role: Sonic Environment**\nCreate evolving sonic textures and soundscapes that define the mood of the track.",
        "lead": "**Your Role: The Main Hook (Lead)**\nCreate the primary, most memorable, and catchy melodic hook of the song.",
        "melody": "**Your Role: The Supporting Melody**\nCreate a secondary or counter-melody that complements the lead or fills space.",
        "chords": "**Your Role: Harmonic Structure**\nDefine the chord progression with clear harmonic movement, using either rhythmic stabs or sustained chords.",
        "arp": "**Your Role: Rhythmic Harmony**\nCreate a hypnotic arpeggio pattern using chord notes in a repetitive, evolving rhythm.",
        "guitar": "**Your Role: Guitar**\nCreate a suitable guitar part, which could be rhythmic chords or a melodic line.",
        "vocal": "**Your Role: Vocal Line**\nCreate a vocal-like melody or rhythmic chop that acts as a lead or supporting element.",
        "fx": "**Your Role: Sound Effects**\nCreate transitional sound effects that help move from one section to another.",
        "riser": "**Your Role: Tension Builder**\nCreate a sound that builds tension, usually by rising in pitch or intensity, leading into a new section."
    }

    # Get the base instruction for the role
    instructions = [role_map.get(role, f"**Your Role: {role.title()}**\nCreate a complementary part that enhances the overall composition.")]

    # 2. Conditionally build a list of proactive automation advice
    automation_prompts = []
    
    # Pitch Bend Advice
    if role in ["bass", "lead", "melody", "guitar", "vocal"] and use_pitch_bend:
        automation_prompts.append("using subtle pitch bends to make it more expressive")
    
    # CC Automation Advice
    if role in ["bass", "pads", "atmosphere", "lead", "arp", "fx", "riser", "sub_bass", "chords"] and use_cc_automation:
        automation_prompts.append("using filter (CC74) or volume (CC11) automation to create movement")

    # Sustain Pedal Advice
    # Added "piano" to the list of roles that might not be in the default map but is a valid role.
    if role in ["pads", "atmosphere", "lead", "chords", "guitar", "melody", "piano"] and use_sustain_pedal:
        automation_prompts.append("using the sustain pedal for smoother phrasing")

    # 3. Append the advice to the main instruction string if any prompts were generated
    if automation_prompts:
        # Create a sentence from the list of prompts.
        prompt_string = " and ".join(automation_prompts)
        instructions.append(f"**Proactive Automation:** You are encouraged to enhance this part by {prompt_string}, even if not explicitly described in the creative direction.")
    
    return "\n".join(instructions)

def get_role_instructions_for_optimization(role: str, config: Dict) -> str:
    """
    Returns concise, conservative role guidance for optimization.
    The intent is minimal edits: fix issues and polish, not rewrite.
    """
    base_instructions = (
        "**Edit Budget:** Change at most ~10–15% of notes. Prefer deletions/shortenings over additions.\n"
        "**Automation:** Only add subtle automation if it clearly fixes an issue; otherwise keep existing."
    )

    if role == "drums":
        return (
            "**Your Role: The Rhythmic Foundation**\n"
            "Tighten groove minimally. Tweak velocities for feel, remove clutter. Avoid adding many new hits."
        )
    elif role == "kick_and_snare":
        return (
            "**Your Role: The Core Beat (Kick & Snare)**\n"
            f"Keep pattern intact; make small velocity balance tweaks for a punchier {config['genre']} groove."
        )
    elif role == "percussion":
        return (
            "**Your Role: Rhythmic Texture**\n"
            "Reduce masking, add tiny syncopation if needed. Prefer subtraction over new layers."
        )
    elif role == "bass":
        return (
            "**Your Role: The Rhythmic and Harmonic Anchor**\n"
            "Tighten timing with kick, tame boomy notes, and adjust a few velocities. Minimal note changes."
        )
    elif role == "sub_bass":
        return (
            "**Your Role: The Low-End Anchor**\n"
            "Keep it clean and supportive. Subtle level/duration tweaks only; avoid extra notes."
        )
    elif role in ["pads", "atmosphere"]:
        return (
            "**Your Role: The Emotional Core and Harmonic Glue**\n"
            "Smooth voice leading; reduce mud. Very subtle swells allowed; avoid big evolutions."
        )
    elif role == "lead":
        return (
            "**Your Role: The Storyteller and Main Hook**\n"
            "Keep motif intact. Nudge phrasing and dynamics slightly; avoid new licks or runs."
        )
    elif role == "melody":
        return (
            "**Your Role: The Supporting Melody**\n"
            "Fit under the lead. Minor timing/velocity tweaks; minimal added notes if necessary."
        )
    elif role == "chords":
        return (
            "**Your Role: The Harmonic Core**\n"
            "Tighten chord voicings and rhythm feel slightly. Keep progression and pattern intact."
        )
    elif role == "arp":
        return (
            "**Your Role: The Hypnotic Arpeggio**\n"
            "Preserve pattern. Slight velocity shaping and rare grace notes only if needed."
        )
    elif role == "guitar":
        return (
            "**Your Role: Guitar**\n"
            "Keep riff/voicings. Minor timing/velocity polish; avoid new lines or dense ornaments."
        )
    elif role == "vocal":
        return (
            "**Your Role: Vocal Line**\n"
            "Maintain melody shape. Subtle dynamics and phrasing tweaks; avoid new melismas."
        )
    elif role in ["fx", "riser"]:
        return (
            "**Your Role: Sound Effects & Transitions**\n"
            "Use existing gestures; only smooth obvious bumps. Avoid larger new sweeps/risers."
        )
    else:
        return (
            f"**Your Role: {role.title()}**\n"
            f"Polish minimally to better fit the mix. {base_instructions}"
        )

def get_progress_filename(config: Dict, run_timestamp: str) -> str:
    """Constructs a descriptive progress filename."""
    genre = config.get("genre", "audio").replace(" ", "_").replace("/", "-")
    bpm = round(float(config.get("bpm", 120)))
    genre = re.sub(r'[\\*?:"<>|]', "", genre) # Sanitize
    return f"progress_run_{genre}_{bpm}bpm_{run_timestamp}.json"

def save_progress(data: Dict, script_dir: str, run_timestamp: str) -> str:
    """Saves progress to a single, run-specific, overwritable JSON file."""
    # --- Robustness: Ensure timestamp is always in the data ---
    if 'timestamp' not in data:
        data['timestamp'] = run_timestamp

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

def _load_progress_silent(progress_path: str) -> Dict:
    """Loads progress JSON without printing (used for scanning)."""
    try:
        with open(progress_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def save_final_artifact(config: Dict, generated_themes: List[Dict], length_bars: int, theme_definitions: List[Dict], script_dir: str, run_timestamp: str) -> str:
    """Saves a reusable artifact of a finished generation for later optimization runs."""
    try:
        artifact = {
            'type': 'final',
            'timestamp': run_timestamp,
            'config': config,
            'length': length_bars,
            'theme_definitions': theme_definitions,
            'themes': generated_themes
        }
        path = os.path.join(script_dir, f"final_run_{run_timestamp}.json")
        with open(path, 'w') as f:
            json.dump(artifact, f, indent=2)
        print(Fore.GREEN + f"Final artifact saved to: {os.path.basename(path)}" + Style.RESET_ALL)
        return path
    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not save final artifact: {e}" + Style.RESET_ALL)
        return ""

def load_final_artifact(path: str) -> Dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(Fore.RED + f"Failed to load artifact: {e}" + Style.RESET_ALL)
        return None

def find_final_artifacts(script_dir: str) -> List[str]:
    pattern = os.path.join(script_dir, "final_run_*.json")
    files = glob.glob(pattern)
    return sorted(files, key=os.path.getmtime, reverse=True)

def summarize_artifact(path: str) -> str:
    try:
        data = load_final_artifact(path)
        if not data:
            return os.path.basename(path)
        cfg = data.get('config', {})
        genre = cfg.get('genre', 'Unknown')
        bpm = cfg.get('bpm', 'N/A')
        key = cfg.get('key_scale', '')
        themes = data.get('themes', [])
        num_parts = len(themes)
        first_label = (themes[0].get('label') if themes and isinstance(themes[0], dict) else None) or 'A'
        last_label = (themes[-1].get('label') if themes and isinstance(themes[-1], dict) else None) or 'Z'
        ts = data.get('timestamp') or time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        return f"{genre} {bpm}bpm | {num_parts} parts | Key {key} | {first_label}→{last_label} | {ts}"
    except Exception:
        return os.path.basename(path)

def summarize_progress_file(path: str) -> str:
    try:
        pdata = load_progress(path)
        if not pdata:
            return os.path.basename(path)
        ptype = pdata.get('type') or pdata.get('generation_type', 'unknown')
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        if 'generation' in ptype:
            return f"Gen: Theme {pdata.get('current_theme_index', 0)+1}, Track {pdata.get('current_track_index',0)+1} ({ts})"
        if 'window_optimization' in ptype:
            wb = pdata.get('window_bars', 'N/A')
            ws = pdata.get('current_window_start_index', 0)
            tr = pdata.get('current_track_in_window', 0)
            return f"WinOpt: {wb} bars, Start Part {ws+1}, Track {tr+1} ({ts})"
        if 'optimization' in ptype:
            return f"Opt: Theme {pdata.get('current_theme_index',0)+1}, Track {pdata.get('current_track_index',0)+1} ({ts})"
        return f"{ptype} ({ts})"
    except Exception:
        return os.path.basename(path)

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