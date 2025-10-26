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
AUTO_ESCALATE_TO_PRO = False      # If True and using flash, auto-switch to pro after N failures per track - DISABLED
AUTO_ESCALATE_THRESHOLD = 6
DEFER_CURRENT_TRACK = False       # If True, immediately defer the current track (skip and push to end)
HOTKEY_DEBOUNCE_SEC = 0.8         # Debounce window for hotkeys
_LAST_HOTKEY_TS = {'1': 0.0, '2': 0.0, '3': 0.0, '0': 0.0, 'a': 0.0, 'd': 0.0, 'h': 0.0}
REDUCE_CONTEXT_THIS_STEP = False  # If True, halve historical context for the current step
REDUCE_CONTEXT_HALVES = 0        # Number of times to halve context for the current step
LAST_CONTEXT_COUNT = 0           # Last known number of context themes (for hotkey preview)
PLANNED_CONTEXT_COUNT = 0        # Planned context size after pending halvings (preview)

# Lyrics per-part meta (optional side-channel from words generation)
LYRICS_PART_META: Dict[str, Dict] = {}

# --- JSON Schemas for structured LLM outputs ---
LYRICS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {"type": "array", "items": {"type": "string"}},
        "melisma_spans": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "phoneme_hints": {"type": ["array", "null"], "items": {"type": ["string", "null"]}},
        "hook_canonical": {"type": ["string", "null"]},
        "hook_token_ranges": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
        "line_breaks": {"type": ["array", "null"], "items": {"type": "integer"}},
        "chant_segments": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
        "phrase_windows": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
        "vowel_sustain_targets": {"type": ["array", "null"], "items": {"type": "integer"}},
        "mapping_feasibility_score": {"type": ["number", "null"]},
        "note_rewrite_request": {"type": ["boolean", "null"]},
        "note_rewrite_intent": {"type": ["string", "null"]},
        "self_check": {
            "type": ["object", "null"],
            "properties": {
                "notes_count": {"type": ["integer", "null"]},
                "spans_sum": {"type": ["integer", "null"]},
                "words_len": {"type": ["integer", "null"]},
                "ok": {"type": ["boolean", "null"]},
                "intentional_silence": {"type": ["boolean", "null"]}
            }
        }
    },
    "required": ["words", "melisma_spans"],
    "additionalProperties": True
}

ROLES_JSON_SCHEMA = {"type": "object"}
HINTS_JSON_SCHEMA = {"type": "object"}
PLAN_JSON_SCHEMA = {"type": "object"}

# --- Quota state tracking (to improve cooldown reset behavior) ---
KEY_QUOTA_TYPE: Dict[int, str] = {}  # index -> 'per-day' | 'per-hour' | 'per-minute' | 'rate-limit' | ...
LAST_PER_DAY_SEEN_TS: float = 0.0
LAST_PER_HOUR_SEEN_TS: float = 0.0
NEXT_HOURLY_PROBE_TS: float = 0.0

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
KEY_QUOTA_TYPE: Dict[int, str] = {}  # index -> last quota class seen
KEY_ROTATION_STRIDE = 1  # try all keys sequentially without skipping
PER_MINUTE_COOLDOWN_SECONDS = 60
PER_HOUR_COOLDOWN_SECONDS = 3600
PER_DAY_COOLDOWN_SECONDS = 86400

def _is_key_available(idx: int) -> bool:
    until = KEY_COOLDOWN_UNTIL.get(idx, 0)
    return time.time() >= until

def _set_key_cooldown(idx: int, seconds: float, *, force: bool = False) -> None:
    target_until = time.time() + max(1.0, seconds)
    if force:
        KEY_COOLDOWN_UNTIL[idx] = target_until
    else:
        KEY_COOLDOWN_UNTIL[idx] = max(KEY_COOLDOWN_UNTIL.get(idx, 0), target_until)

def _reset_all_cooldowns() -> None:
    """Reset all API key cooldowns - useful for fresh API keys."""
    global KEY_COOLDOWN_UNTIL, KEY_QUOTA_TYPE
    KEY_COOLDOWN_UNTIL.clear()
    KEY_QUOTA_TYPE.clear()
    print(Fore.GREEN + "All API key cooldowns reset." + Style.RESET_ALL)

def _next_available_key(start_idx: int | None = None) -> int | None:
    if not API_KEYS:
        return None
    n = len(API_KEYS)
    s = CURRENT_KEY_INDEX if start_idx is None else start_idx
    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
    for off in range(1, n+1):
        idx = (s + off*stride) % n
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
    try:
        if not API_KEYS:
            return False
        # Only consider keys we have seen quota for; if any key not per-day (or unknown), return False
        for i in range(len(API_KEYS)):
            if KEY_QUOTA_TYPE.get(i) != 'per-day':
                return False
        return True
    except Exception:
        return False

def _schedule_hourly_probe_if_needed() -> None:
    global NEXT_HOURLY_PROBE_TS
    try:
        now = time.time()
        # If no probe scheduled or already passed, schedule next in one hour from now
        if NEXT_HOURLY_PROBE_TS <= now:
            NEXT_HOURLY_PROBE_TS = now + PER_HOUR_COOLDOWN_SECONDS
    except Exception:
        NEXT_HOURLY_PROBE_TS = time.time() + PER_HOUR_COOLDOWN_SECONDS

def _seconds_until_hourly_probe() -> float:
    try:
        now = time.time()
        if NEXT_HOURLY_PROBE_TS <= now:
            return 1.0
        return max(1.0, NEXT_HOURLY_PROBE_TS - now)
    except Exception:
        return PER_HOUR_COOLDOWN_SECONDS

def _clear_all_cooldowns() -> None:
    try:
        for i in range(len(API_KEYS)):
            KEY_COOLDOWN_UNTIL[i] = 0
    except Exception:
        pass

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
    return (
        track_dict.get('instrument_name')
        or track_dict.get('instrument')
        or track_dict.get('name')
        or 'Unknown Instrument'
    )

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
# Export beautification defaults (not a hard rule, only UST output smoothing)
EXPORT_TINY_NOTE_SMOOTHING = True
EXPORT_TINY_NOTE_THRESH_BEATS = 0.25

# --- LYRICS GENERATION (helper) ---
def _generate_lyrics_syllables(config: Dict, genre: str, inspiration: str, track_name: str, bpm: int | float, ts: Dict, notes: List[Dict], section_label: str | None = None, section_description: str | None = None, context_tracks_basic: List[Dict] | None = None, cfg: Dict | None = None) -> List[str]:
    """
    Legacy syllable-per-note generator (kept for fallback). Prefer word-first spans.
    """
    try:
        import google.generativeai as genai_local
    except Exception:
        genai_local = None

    num_slots = len(notes)
    if num_slots <= 0:
        return []

    # Build compact note timing preview for the prompt (start, dur, pitch) and simple stress flags
    try:
        beats_per_bar = int(ts.get("beats_per_bar", 4))
        def _stress_for(start_beat: float) -> int:
            try:
                pos = start_beat % max(1, beats_per_bar)
                # simple heuristic: strong on beat 1 and mid-beat in even meters
                if abs(pos - 0) < 1e-3:
                    return 1
                if beats_per_bar % 2 == 0 and abs(pos - (beats_per_bar/2)) < 1e-3:
                    return 1
                return 0
            except Exception:
                return 0
        preview = []
        for n in sorted(notes, key=lambda x: float(x.get("start_beat", 0.0))):
            s = float(n.get("start_beat", 0.0))
            d = max(0.0, float(n.get("duration_beats", 0.0)))
            p = int(n.get("pitch", 60))
            preview.append({
                "start": round(s, 3),
                "dur": round(d, 3),
                "pitch": p,
                "stress": _stress_for(s)
            })
        preview = preview[:256]
    except Exception:
        preview = []

    # Prompt (legacy): ask for exactly num_slots tokens
    # Extract musical parameters from JSON (not config.yaml)
    language = str(cfg.get("lyrics_language", "English")) if cfg else "English"
    # Use passed key_scale parameter first, then fallback to cfg
    if key_scale and str(key_scale).strip():
        key_scale = str(key_scale).strip()
    else:
        key_scale = str(cfg.get("key_scale", "")).strip() if cfg else ""
    
    # Use the scale from JSON - no automatic "correction"
    print(f"[INFO] Using scale from JSON: {key_scale}")
    
    # ANTI-MICRO-NOTE: Force longer, more natural phrasing
    print(f"[ANTI-MICRO] Enforcing natural phrasing with scale: {key_scale}")
    vocab = {
        "context_instruments": (context_tracks_basic or []),
        "style_keywords": [genre, inspiration][:8]
    }
    prompt = (
        "You are a professional lyricist. Create singable, meaningful tokens aligned to the melody notes.\n"
        "Goal: genre-true, minimal, context-aware phrasing; avoid generic clichés. Keep guidance broadly musical (no hard stylistic constraints).\n"
        f"Global: Genre={genre}; Language={language}; Key/Scale={key_scale}; BPM={round(float(bpm))}; TimeSig={ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}.\n"
        f"Track: {track_name}.\n"
        + (f"Section: {section_label or ''}. Description: {section_description or ''}.\n" if (section_label or section_description) else "")
        + ("Instruments (name,role): " + json.dumps(vocab.get('context_instruments')) + "\n" if vocab.get('context_instruments') else "")
        + ("Style hints (artist if given): " + ", ".join([x for x in vocab.get('style_keywords') if x]) + "\n" if any(vocab.get('style_keywords')) else "")
        + "Melody notes (order): each = {start,dur,pitch,stress}, stress: 1 strong, 0 weak.\n"
        + json.dumps(preview)
        + "\n\nLYRICS RULES:\n"
        + "- **TARGET DURATION RANGE**: Most notes 0.75–3.0 beats; hard minimum 0.5.\n"
        + "- **PHRASE-BASED COMPOSITION**: Group words into complete phrases of 3–6 words.\n"
        + "- **NO MICRO-NOTES**: Never create notes shorter than 2.0 beats. Group short musical events into longer phrases.\n"
        + "- **CONTINUOUS FLOW**: Connect notes end-to-start. Minimize gaps between notes.\n"
        + "- **MUSICAL COHERENCE**: Each phrase must be a complete musical thought.\n"
        + "- **MAXIMUM MELISMA**: Never exceed 20% of total tokens as melisma.\n"
        + "\nStrict output format (JSON only):\n"
        + "{\n  \"syllables\": [string, string, ...]\n}\n\n"
        + f"Constraints:\n- The array length MUST be exactly {num_slots}.\n"
        + "- No null/empty items.\n- Prefer meaningful words over generic vowels.\n"
        + "- Use '-' only to continue a previous word (melisma); never on a fresh onset.\n"
        + "- Follow universal pop-lyric heuristics: clear images, concrete nouns/verbs, a memorable hook, avoid cliche overload, keep pronoun perspective consistent.\n"
        + "- If inspiration mentions an artist, emulate stylistic fingerprints (rhythm of phrasing, imagery types) without copying lines.\n"
        + "- No markdown or prose, JSON only.\n"
        + ("- Apply plan_hint if present: prioritize entry_beats and phrase_window_beats; target onset_count_min/max and duration_min/max in mapping choices.\n" if isinstance(section_description, str) and 'entry_beats' in (section_description or '') else "")
    )

    # Check API availability - let the retry logic handle failures
    if genai_local is None or not API_KEYS:
        print(Fore.RED + "ERROR: API unavailable - this should be handled by retry logic!" + Style.RESET_ALL)
        return []

    try:
        # Use session override if available, otherwise config (technical parameters come from config.yaml)
        global SESSION_MODEL_OVERRIDE
        model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
        if not model_name:
            model_name = "gemini-2.5-pro"
        try:
            if REQUESTED_SWITCH_MODEL:
                model_name = REQUESTED_SWITCH_MODEL
        except Exception:
            pass
        # Per-part ultra-low temperature unless overridden
        try:
            role_for_temp = _normalize_section_role(section_label)
        except Exception:
            role_for_temp = "verse"
        derived_temp = 0.0
        if role_for_temp in ("backing", "scat", "vowels"):
            derived_temp = 0.1
        elif role_for_temp == "bridge":
            derived_temp = 0.05
        _cfg_temp = config.get("lyrics_temperature") if isinstance(config.get("lyrics_temperature"), (int, float)) else None
        if _cfg_temp is None and isinstance(config.get("temperature"), (int, float)):
            _cfg_temp = config.get("temperature")
        selected_temp = float(_cfg_temp) if isinstance(_cfg_temp, (int, float)) else float(derived_temp)
        generation_config = {"response_mime_type": "application/json", "temperature": selected_temp}
        # Do not set response_schema here to avoid SDK schema normalization errors
        try:
            if isinstance(config.get("max_output_tokens"), int):
                _mx = int(config.get("max_output_tokens"))
                # Clamp to a safe range to avoid API 400 errors
                _mx = max(256, min(_mx, 65536))
        except Exception:
            _mx = 4096
        generation_config["max_output_tokens"] = _mx
        model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        # Add generation_config again here to enforce JSON output
        def _try_once(prompt_text: str) -> List[str] | None:
            # LLM call with key rotation/backoff similar to optimization path
            nonlocal model
            max_attempts = max(3, len(API_KEYS))
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                try:
                    print(Style.DIM + f"[Lyrics] Attempt {attempts}/{max_attempts}" + Style.RESET_ALL)
                    resp = model.generate_content(prompt_text, safety_settings=safety_settings, generation_config=generation_config)
                    raw_txt = getattr(resp, "text", "") or ""
                    try:
                        try:
                            obj = json.loads(raw_txt)
                        except Exception:
                            cleaned = raw_txt.strip().replace("```json", "").replace("```", "")
                            obj = json.loads(cleaned)
                        syll = obj.get("syllables") if isinstance(obj, dict) else None
                        if not (isinstance(syll, list) and len(syll) == num_slots):
                            return None
                        # Basic validation: reject trivial placeholders
                        bad = {"la","na","da","ta","ba","pa","ma"}
                        cleaned_syll = []
                        for s in syll:
                            if not isinstance(s, str) or not s.strip():
                                return None
                            ss = s.strip()
                            cleaned_syll.append(ss)
                        if cleaned_syll and (sum(1 for x in cleaned_syll if x.lower() in bad) / len(cleaned_syll)) > 0.5:
                            return None
                        return cleaned_syll
                    except Exception:
                        # raw dump preview for debugging
                        try:
                            prev = raw_txt[:200].replace("\n"," ")
                            print(Style.DIM + f"[Lyrics] Raw preview: {prev}..." + Style.RESET_ALL)
                        except Exception:
                            pass
                        return None
                except Exception as e:
                    err = str(e).lower()
                    if ('429' in err or 'quota' in err or 'rate limit' in err):
                        qtype = _classify_quota_error(err)
                        KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                        # rotate keys if possible
                        rotated = False
                        n = len(API_KEYS)
                        for off in range(1, n+1):
                            idx = (CURRENT_KEY_INDEX + off) % n
                            until = KEY_COOLDOWN_UNTIL.get(idx, 0)
                            if time.time() < until:
                                continue
                            try:
                                globals()['CURRENT_KEY_INDEX'] = idx
                                genai.configure(api_key=API_KEYS[idx])
                                print(Fore.CYAN + f"[Lyrics] Switching to API key #{idx+1}..." + Style.RESET_ALL)
                                model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                                rotated = True
                                break
                            except Exception:
                                continue
                        if not rotated:
                            # backoff
                            cd = 60 if qtype not in ('per-hour','per-day') else 3600
                            KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                            wait_s = 3 if qtype not in ('per-hour','per-day') else 15
                            _interruptible_backoff(wait_s, config, context_label="Lyrics cooldown")
                            continue
                        else:
                            continue
                    else:
                        # transient non-quota
                        time.sleep(2)
                        continue

        result = _try_once(prompt)
        if result is not None:
            return result
        # Repair pass disabled: return best-effort only
    except Exception:
        try:
            print(Fore.RED + "Lyrics: Generation failed - this should be handled by retry logic!" + Style.RESET_ALL)
        except Exception:
            pass

    # Return empty - let the retry logic handle failures
    return []

def _tokens_to_words(tokens: List[str]) -> List[str]:
    try:
        out = []
        for t in tokens or []:
            tt = str(t).strip()
            if not tt or tt == '-':
                continue
            out.append(tt)
        return out
    except Exception:
        return []

def _normalize_section_role(label: str | None) -> str:
    try:
        if not isinstance(label, str):
            return 'unknown'
        l = label.strip().lower()
        if any(k in l for k in ['chorus', 'refrain', 'hook', 'drop']):
            return 'chorus'
        if any(k in l for k in ['pre-chorus', 'prech', 'pre drop', 'pre-drop', 'build', 'buildup']):
            return 'prechorus'
        if 'bridge' in l:
            return 'bridge'
        if any(k in l for k in ['verse', 'strophe']):
            return 'verse'
        if any(k in l for k in ['intro']):
            return 'intro'
        if any(k in l for k in ['outro', 'ending', 'final']):
            return 'outro'
        return 'unknown'
    except Exception:
        return 'unknown'

def _summarize_vocal_parts(themes: List[Dict], track_index: int, ts: Dict) -> List[Dict]:
    """
    Build compact per-part summaries for the target vocal track:
    { idx, label, num_notes, notes_density, avg_dur, max_dur, sustain_ratio, silent }
    """
    try:
        beats_per_bar = int(ts.get('beats_per_bar', 4))
    except Exception:
        beats_per_bar = 4
    summaries: List[Dict] = []
    for part_idx, th in enumerate(themes or []):
        label = th.get('label', f'Part_{part_idx+1}') if isinstance(th, dict) else f'Part_{part_idx+1}'
        trks = th.get('tracks', []) if isinstance(th, dict) else []
        notes = sorted((trks[track_index].get('notes', []) if (0 <= track_index < len(trks)) else []), key=lambda n: float(n.get('start_beat', 0.0)))
        if not notes:
            summaries.append({"idx": part_idx, "label": label, "num_notes": 0, "notes_density": 0.0, "avg_dur": 0.0, "max_dur": 0.0, "sustain_ratio": 0.0, "silent": True})
            continue
        starts = [float(n.get('start_beat', 0.0)) for n in notes]
        durs = [max(0.0, float(n.get('duration_beats', 0.0))) for n in notes]
        total_beats = (max(starts) - min(starts)) + (durs[-1] if durs else 0.0)
        density = (len(notes) / max(1e-6, total_beats/float(beats_per_bar))) if total_beats > 0 else float(len(notes))
        avg_dur = (sum(durs)/max(1, len(durs)))
        max_dur = max(durs) if durs else 0.0
        sustain_ratio = (sum(durs)/max(1e-6, total_beats)) if total_beats > 0 else 0.0
        summaries.append({
            "idx": part_idx, "label": label, "num_notes": len(notes), "notes_density": round(density,3),
            "avg_dur": round(avg_dur,3), "max_dur": round(max_dur,3), "sustain_ratio": round(sustain_ratio,3), "silent": False
        })
    return summaries

def _summarize_parts_aggregate(themes: List[Dict], ts: Dict, exclude_track_index: int | None = None) -> List[Dict]:
    """
    Aggregate per-part summaries across existing tracks (excluding optional track index).
    Used when a NEW vocal track does not yet exist, providing context for planning.
    { idx, label, num_notes, notes_density, avg_dur, max_dur, sustain_ratio, silent }
    """
    try:
        beats_per_bar = int(ts.get('beats_per_bar', 4)) if isinstance(ts, dict) else 4
    except Exception:
        beats_per_bar = 4
    summaries: List[Dict] = []
    for part_idx, th in enumerate(themes or []):
        label = th.get('label', f'Part_{part_idx+1}') if isinstance(th, dict) else f'Part_{part_idx+1}'
        trks = th.get('tracks', []) if isinstance(th, dict) else []
        # Collect all notes across tracks except excluded index
        all_notes: List[Dict] = []
        try:
            for ti, t in enumerate(trks):
                if exclude_track_index is not None and ti == exclude_track_index:
                    continue
                for n in (t.get('notes', []) or []):
                    all_notes.append(n)
        except Exception:
            all_notes = []
        notes = sorted(all_notes, key=lambda n: float(n.get('start_beat', 0.0)))
        if not notes:
            summaries.append({"idx": part_idx, "label": label, "num_notes": 0, "notes_density": 0.0, "avg_dur": 0.0, "max_dur": 0.0, "sustain_ratio": 0.0, "silent": True})
            continue
        starts = [float(n.get('start_beat', 0.0)) for n in notes]
        durs = [max(0.0, float(n.get('duration_beats', 0.0))) for n in notes]
        total_beats = (max(starts) - min(starts)) + (durs[-1] if durs else 0.0)
        density = (len(notes) / max(1e-6, total_beats/float(beats_per_bar))) if total_beats > 0 else float(len(notes))
        avg_dur = (sum(durs)/max(1, len(durs)))
        max_dur = max(durs) if durs else 0.0
        sustain_ratio = (sum(durs)/max(1e-6, total_beats)) if total_beats > 0 else 0.0
        summaries.append({
            "idx": part_idx, "label": label, "num_notes": len(notes), "notes_density": round(density,3),
            "avg_dur": round(avg_dur,3), "max_dur": round(max_dur,3), "sustain_ratio": round(sustain_ratio,3), "silent": False
        })
    return summaries
def _plan_vocal_roles(config: Dict, genre: str, inspiration: str, bpm: float | int, ts: Dict, summaries: List[Dict], analysis_ctx: Dict, user_prompt: str | None = None, cfg: Dict | None = None) -> List[Dict]:
    """
    Step 0b: Plan vocal roles for each part based on song structure and analysis.
    Focus only on logical role distribution, no hints yet.
    """
    try:
        # Extract analysis context
        hook_canonical = analysis_ctx.get('hook_canonical', '')
        style_tags = analysis_ctx.get('style_tags', [])
        repetition_policy = analysis_ctx.get('repetition_policy', {})
        
        # Extract key_scale from artifact if available, otherwise use config
        # Note: cfg parameter should contain the artifact/progress data with key_scale
        key_scale_to_use = cfg.get('key_scale', '') if cfg else config.get('key_scale', 'Unknown')
        
        # Build role planning prompt
        role_prompt = f"""You are a vocal arrangement specialist. Plan vocal roles for a {genre} track inspired by "{inspiration}".

SONG STRUCTURE:
- {len(summaries)} parts total
- BPM: {bpm}
- Time signature: {ts.get('beats_per_bar', 4)}/4

KEY CONTEXT:
- Key/Scale: {key_scale_to_use} (soft guidance: prefer tonic/dominant focus for monotone-suitable roles)

ANALYSIS CONTEXT:
- Hook: {hook_canonical if hook_canonical else 'None (instrumental focus)'}
- Style: {', '.join(style_tags) if style_tags else 'Standard'}
- Repetition: {repetition_policy}

ROLE CATALOG:
- silence: No vocals, instrumental only
- verse: Main vocal content, storytelling
- chorus: Hook/refrain, repetitive elements
- bridge: Transitional, different energy
- outro: Concluding vocals
- intro: Opening vocals
- whisper: Quiet, intimate delivery
- spoken: Conversational, non-melodic
- shout: Aggressive, high energy
- chant: Rhythmic, repetitive
- adlib: Improvised, expressive
- harmony: Supporting vocals
- doubles: Vocal doubling
- response: Call-and-response
- rap: Rhythmic speech
- choir: Multiple voices
- hum: Non-verbal vocalization
- breaths: Breathing sounds
- tag: Short vocal elements
- phrase_spot: Single phrases
- vocoder: Processed vocals
- talkbox: Synthesized speech
- vocal_fx: Special effects

VOCAL BUDGET:
- Target: 8-12 vocal parts for a {len(summaries)}-part song (more vocal content)
- Minimum gap: 0-1 parts between vocal sections (allow consecutive vocal parts)
- Prefer consistent vocal presence with strategic silence
- Use silence sparingly - only for clear instrumental breaks or dramatic contrast

SILENCE POLICY (STRICT):
- **MINIMIZE SILENCE**: Use role='silence' only for clear instrumental breaks or dramatic contrast
- **PREFER VOCAL CONTENT**: Choose whisper/hum/breaths/vocal_fx over complete silence
- **ATMOSPHERIC ROLES**: Use whisper, hum, breaths, or vocal_fx for atmospheric sections
- **VOCAL DENSITY**: Aim for 70-80% of parts to have some form of vocal content
- **SILENCE EXCEPTIONS**: Only use silence for intro/outro or clear instrumental drops

TASK:
For each part, assign ONLY a role. No hints, descriptions, or explanations.
Return as JSON array: [{{"idx": 1, "role": "silence"}}, {{"idx": 2, "role": "verse"}}, ...]

Focus on:
1. **MAXIMIZE VOCAL CONTENT**: Assign vocal roles to 70-80% of parts
2. Logical role distribution with minimal silence
3. Appropriate vocal density for the genre
4. User intent (if provided)
5. **PREFER ATMOSPHERIC ROLES**: Use whisper/hum/breaths/vocal_fx instead of silence
6. **SILENCE POLICY (STRICT)**: Assign 'silence' only for intro/outro or explicit breaths-only moments. For all other sections, choose a minimal vocal role (hum/whisper/vocal_fx) instead of 'silence'.

Return only the JSON array, no other text."""

        if user_prompt:
            role_prompt += f"\n\nUSER GUIDANCE:\n{user_prompt}"

        # Call LLM for role planning with robust key rotation and retry logic
        try:
            import google.generativeai as genai
        except Exception:
            genai = None
        
        if genai is None or not API_KEYS:
            raise RuntimeError("Role planning unavailable: no LLM SDK or API key")
        
        # Use session override if available, otherwise config (no flash fallback)
        global SESSION_MODEL_OVERRIDE
        model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
        if not model_name:
            model_name = config.get("model_name", "gemini-2.5-pro")
        
        # Plan generation should be highly deterministic
        _plan_temp = 0.0
        try:
            if isinstance(config.get("plan_temperature"), (int, float)):
                _plan_temp = float(config.get("plan_temperature"))
            elif isinstance(config.get("lyrics_temperature"), (int, float)):
                _plan_temp = float(config.get("lyrics_temperature"))
            elif isinstance(config.get("temperature"), (int, float)):
                _plan_temp = float(config.get("temperature"))
        except Exception:
            _plan_temp = 0.0
        
        generation_config = {"response_mime_type": "application/json", "temperature": _plan_temp}
        # Do not set response_schema for roles planning
        try:
            if isinstance(config.get("max_output_tokens"), int):
                _mx = int(config.get("max_output_tokens"))
                _mx = max(256, min(_mx, 8192))
                generation_config["max_output_tokens"] = _mx
        except Exception:
            pass
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Robust retry logic with key rotation
        max_attempts = max(3, len(API_KEYS) * 2)
        quota_rotation_count = 0
        
        for attempt in range(max_attempts):
            try:
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                response = model.generate_content(role_prompt, generation_config=generation_config, safety_settings=safety_settings)
                response_text = response.text if response and response.text else ""
                
                if response_text:
                    # Parse response
                    try:
                        roles_data = json.loads(response_text.strip())
                        if not isinstance(roles_data, list):
                            raise ValueError("Expected JSON array")
                        
                        # Normalize to exact number of parts
                        mapped = []
                        for i in range(len(summaries)):
                            if i < len(roles_data) and isinstance(roles_data[i], dict):
                                role = str(roles_data[i].get('role', 'silence')).strip().lower()
                                # Map to valid roles - handle 'unknown' and other invalid roles
                                valid_roles = ['silence', 'verse', 'chorus', 'bridge', 'outro', 'intro', 'whisper', 'spoken', 'shout', 'chant', 'adlib', 'harmony', 'doubles', 'response', 'rap', 'choir', 'hum', 'breaths', 'tag', 'phrase_spot', 'vocoder', 'talkbox', 'vocal_fx']
                                if role not in valid_roles or role in ['unknown', '']:
                                    role = 'silence'
                                mapped.append({"idx": i+1, "role": role})
                            else:
                                mapped.append({"idx": i+1, "role": "silence"})
                        
                        # Post-validate silence policy: allow silence only at boundaries (intro/outro)
                        try:
                            n_parts = len(summaries)
                            for i in range(len(mapped)):
                                role_i = str(mapped[i].get('role','')).lower()
                                if role_i == 'silence' and i not in (0, max(0, n_parts-1)):
                                    mapped[i]['role'] = 'hum'
                        except Exception:
                            pass
                        return mapped
                    except Exception as e:
                        raise ValueError(f"Role planning parse error: {e}")
                else:
                    raise ValueError("Empty response")
                    
            except Exception as e:
                err = str(e).lower()
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    try:
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        avail = []
                        now = time.time()
                        for ix, _ in enumerate(API_KEYS):
                            tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                            avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                        print(Fore.MAGENTA + Style.BRIGHT + "[Role Planning:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                    except Exception:
                        pass
                    
                    # Try to rotate to next available key
                    n = len(API_KEYS)
                    rotated = False
                    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off*stride) % n
                        if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            rotated = True
                            break
                        except Exception:
                            continue
                    
                    if not rotated:
                        wait_s = 3 if qtype not in ('per-hour','per-day') else 15
                        _interruptible_backoff(wait_s, config, context_label="Role Planning cooldown")
                        quota_rotation_count += 1
                        continue
                    else:
                        continue
                else:
                    # Non-quota error, short backoff
                    time.sleep(0.5)
                    continue
        
        raise ValueError("Role planning failed: max attempts exceeded")
        
    except Exception as e:
        print(f"[Role Planning Error] {e}")
        # No heuristic fallback - let the LLM handle all decisions
        # If LLM fails, return silence for all parts to maintain consistency
        out: List[Dict] = []
        for i, s in enumerate(summaries):
            out.append({"idx": s['idx'], "role": "silence"})
        return out

def _generate_vocal_hints(config: Dict, genre: str, inspiration: str, bpm: float | int, ts: Dict, summaries: List[Dict], roles: List[Dict], analysis_ctx: Dict, user_prompt: str | None = None, cfg: Dict | None = None) -> List[Dict]:
    """
    Step 0c: Generate hints for each part based on assigned roles and context.
    Includes consistency validation.
    """
    try:
        # Extract analysis context
        hook_canonical = analysis_ctx.get('hook_canonical', '')
        style_tags = analysis_ctx.get('style_tags', [])
        repetition_policy = analysis_ctx.get('repetition_policy', {})
        
        # Build hint generation prompt (compact, numeric; no mix/FX/panning prose)
        # Inject key/scale + tonic center for tonal steering
        try:
            # Use artifact key_scale if available, otherwise fallback to config
            key_scale_str = str(cfg.get('key_scale', '') if cfg else config.get('key_scale', '') or '').strip()
            def _infer_root_from_key(ks: str) -> int:
                try:
                    tonic = (ks or '').strip().split()[0].lower()
                    table = {
                        'c': 60, 'c#': 61, 'db': 61,
                        'd': 62, 'd#': 63, 'eb': 63,
                        'e': 64, 'fb': 64, 'e#': 65,
                        'f': 65, 'f#': 66, 'gb': 66,
                        'g': 67, 'g#': 68, 'ab': 68,
                        'a': 57, 'a#': 58, 'bb': 58,  # A3 instead of A4
                        'b': 71, 'cb': 71
                    }
                    return table.get(tonic, 60)
                except Exception:
                    return 60
            # Use artifact root_note if available, otherwise fallback to config
            if cfg and isinstance(cfg.get('root_note'), (int, float)):
                tonic_center = int(cfg.get('root_note'))
            elif isinstance(config.get('root_note'), (int, float)):
                tonic_center = int(config.get('root_note'))
            else:
                tonic_center = _infer_root_from_key(key_scale_str)
            tonic_center = max(55, min(tonic_center, 79))
            scale_type = (key_scale_str.split(' ', 1)[1].lower() if ' ' in key_scale_str else 'major') or 'major'
        except Exception:
            key_scale_str = ''
            tonic_center = 60
            scale_type = 'major'

        hint_prompt = f"""You generate compact, numeric vocal hints directly implementable by notes/tokens. Do NOT include mixing/FX/panning/production instructions. Keep guidance musical and widely applicable. Hints are soft preferences; the Composer (Stage-2) decides exact note placement. In case of any conflict: numeric plan_hint parameters take priority over role defaults. If onset/duration targets feel unfit for melody length, adjust self-consistently (self-repair) rather than forcing exact windows.

TARGET DURATION GUIDELINE: Aim for duration_min≈0.75–1.0, duration_max≈3.0–4.0 (beats). Avoid values <0.5.

SONG STRUCTURE:
- parts_total={len(summaries)}
- bpm={bpm}
- time_signature={ts.get('beats_per_bar', 4)}/4

CONTEXT:
- hook={hook_canonical if hook_canonical else 'None'}
- style_tags={', '.join(style_tags) if style_tags else 'Standard'}
- repetition={repetition_policy}
- key_scale={key_scale_str if key_scale_str else 'Unknown'}; tonic_midi={tonic_center}; scale_type={scale_type}

ASSIGNED ROLES:
{chr(10).join([f"Part {r['idx']}: {r['role']}" for r in roles])}

RULES (HARD):
- Hints MUST be implementable by notes/tokens only. Forbid: sidechain, granular, carrier, vocoder carrier, pan/panning, delay/reverb values, distortion, mix.
- role='silence' → force_silence=1 only (no prose).
- **MINIMIZE force_silence=1**: Only use for clear instrumental breaks or dramatic contrast
- **PREFER VOCAL HINTS**: Generate musical hints for whisper/hum/breaths/vocal_fx roles
- Do NOT prescribe exact onsets or phrase windows. Provide only soft ranges for onset_count and duration; the Composer will place notes freely.
- Use key=value pairs; numbers/arrays only; keep each plan_hint ≤ 240 chars.

PLAN_HINT KEYS (allowed):
- force_silence: 0|1
- onset_count_min, onset_count_max: int
- duration_min, duration_max: float (beats)
- tessitura_center: int (MIDI) [optional]
- pitch_span_semitones: int
- max_unique_pitches: int
- token_policy: lexical|vowel|br|fx_vowel
- repetition_policy: {{allow_loop: bool, motif_len_beats: number}}

VOCAL ROLE CHARACTERISTICS (interpret as musical guidance):
- silence: No vocal content - instrumental only
- chorus: Main vocal focus - full melodic expression, memorable hooks, clear pitch movement
- prechorus: Building energy - rising melodic contours, tension-building phrases
- whisper/spoken: Intimate delivery - subtle pitch variation, more rhythmic than melodic
- phrase_spot: Highlight moments - focused, impactful phrases with clear musical intent
- vocoder: Electronic processing - experimental pitch choices, synthetic vocal textures
- vocal_fx: Sound effects - non-lexical content, atmospheric vocal elements
- breaths: Natural pauses - breathing sounds and vocal punctuation

ROLE DURATION GUIDELINE: All vocal roles should avoid <0.5 beats; prefer 0.75–3.0 beats typical range.

MUSICAL PHILOSOPHY:
- **NATURAL EXPRESSION**: Create melodies that feel organic and emotionally authentic
- **VOCAL REALISM**: Consider how a real singer would approach this material
- **EMOTIONAL COHERENCE**: Let the emotional content of the lyrics guide your musical choices
- **HARMONIC AWARENESS**: Be mindful of the backing music - create melodies that complement rather than compete
- **RHYTHMIC FLOW**: Balance sustained and staccato notes for natural phrasing
- **MELODIC SHAPING**: Create clear musical phrases with beginning, development, and resolution
- **VARIETY WITH PURPOSE**: Use musical variety to serve the emotional narrative, not for its own sake

CREATIVE GUIDANCE:
- **MELODIC FLOW**: Create natural, singable phrases that flow like speech but with musical beauty
- **RHYTHMIC INTELLIGENCE**: Vary note lengths to create engaging, non-repetitive patterns
- **SCALE MASTERY**: Work within the song's key/scale as your foundation, using out-of-scale tones only for expressive moments
- **PHRASE ARCHITECTURE**: Build musical phrases with clear beginning, development, and resolution
- **MOTIVIC DEVELOPMENT**: Create and develop musical ideas throughout the section - don't just repeat randomly
- **VOCAL AUTHENTICITY**: Think about breathing, emphasis, and natural vocal expression
- **EMOTIONAL CONNECTION**: Let the lyrics and musical context guide your creative choices
- **PROFESSIONAL POLISH**: Aim for the quality you'd expect from a commercial release

OUTPUT JSON (strict):
- Return JSON array: [{{"idx": 1, "role": "chorus", "plan_hint": "entry_beats=[2.0]; onset_count_min=3; ..."}}, ...]
- Allowed keys: idx, role, plan_hint. No extra fields.
- **CRITICAL**: Generate musical hints for 70-80% of parts - minimize force_silence=1!
        """

        if user_prompt:
            hint_prompt += f"\n\nUSER GUIDANCE:\n{user_prompt}"

        # Call LLM for hint generation with robust key rotation and retry logic
        try:
            import google.generativeai as genai
        except Exception:
            genai = None
        
        if genai is None or not API_KEYS:
            raise RuntimeError("Hint generation unavailable: no LLM SDK or API key")
        
        # Use session override if available, otherwise config (no flash fallback)
        global SESSION_MODEL_OVERRIDE
        model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
        if not model_name:
            model_name = config.get("model_name", "gemini-2.5-pro")
        
        # Plan generation should be highly deterministic
        _plan_temp = 0.0
        try:
            if isinstance(config.get("plan_temperature"), (int, float)):
                _plan_temp = float(config.get("plan_temperature"))
            elif isinstance(config.get("lyrics_temperature"), (int, float)):
                _plan_temp = float(config.get("lyrics_temperature"))
            elif isinstance(config.get("temperature"), (int, float)):
                _plan_temp = float(config.get("temperature"))
        except Exception:
            _plan_temp = 0.0
        
        generation_config = {"response_mime_type": "application/json", "temperature": _plan_temp}
        try:
            if isinstance(config.get("max_output_tokens"), int):
                _mx = int(config.get("max_output_tokens"))
                _mx = max(256, min(_mx, 8192))
                generation_config["max_output_tokens"] = _mx
        except Exception:
            pass
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Robust retry logic with key rotation
        max_attempts = max(3, len(API_KEYS) * 2)
        quota_rotation_count = 0
        
        for attempt in range(max_attempts):
            try:
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                response = model.generate_content(hint_prompt, generation_config=generation_config, safety_settings=safety_settings)
                response_text = response.text if response and response.text else ""
                
                if response_text:
                    # Parse response
                    try:
                        hints_data = json.loads(response_text.strip())
                        if not isinstance(hints_data, list):
                            raise ValueError("Expected JSON array")
                        
                        # Normalize and validate
                        mapped = []
                        for i in range(len(summaries)):
                            if i < len(hints_data) and isinstance(hints_data[i], dict):
                                role = str(hints_data[i].get('role', 'silence')).strip().lower()
                                hint = str(hints_data[i].get('plan_hint', '')).strip()
                                
                                # Handle invalid roles
                                valid_roles = ['silence', 'verse', 'chorus', 'bridge', 'outro', 'intro', 'whisper', 'spoken', 'shout', 'chant', 'adlib', 'harmony', 'doubles', 'response', 'rap', 'choir', 'hum', 'breaths', 'tag', 'phrase_spot', 'vocoder', 'talkbox', 'vocal_fx']
                                if role not in valid_roles or role in ['unknown', '']:
                                    role = 'silence'
                                
                                # Consistency validation (numeric plan_hint aware)
                                # If force_silence=1 appears with a non-silence role → coerce to silence
                                try:
                                    fs_flag = False
                                    if 'force_silence' in hint:
                                        # naive parse: look for '=1' after key
                                        fs_flag = 'force_silence=1' in hint.replace(' ', '')
                                    if fs_flag and role != 'silence':
                                        role = 'silence'
                                    if role == 'silence':
                                        # normalize hint to minimal form
                                        hint = 'force_silence=1'
                                    else:
                                        # for non-silence roles, ensure we do not carry a silence flag
                                        if 'force_silence' in hint:
                                            # remove any force_silence=x segments
                                            parts = [p.strip() for p in hint.split(';') if p.strip()]
                                            parts = [p for p in parts if not p.strip().startswith('force_silence=')]
                                            hint = '; '.join(parts)
                                        # Strip phrase_window_beats and entry_beats to keep Stage-2 free placement
                                        if 'phrase_window_beats' in hint or 'entry_beats' in hint:
                                            parts = [p.strip() for p in hint.split(';') if p.strip()]
                                            parts = [p for p in parts if not p.lower().startswith('phrase_window_beats') and not p.lower().startswith('entry_beats')]
                                            hint = '; '.join(parts)
                                        # Duration soft-guard: prefer musical ranges
                                        # Ensure duration_min ≥ 0.75 (if present), duration_max ≥ 2.0, onset_count_max ≤ 8
                                        try:
                                            import re
                                            dm = re.search(r'duration_min\s*=\s*([0-9.]+)', hint)
                                            if dm and float(dm.group(1)) < 0.75:
                                                hint = re.sub(r'duration_min\s*=\s*[0-9.]+', 'duration_min=0.75', hint)
                                            # Also enforce duration_max to be reasonable
                                            dm_max = re.search(r'duration_max\s*=\s*([0-9.]+)', hint)
                                            if dm_max and float(dm_max.group(1)) < 2.0:
                                                hint = re.sub(r'duration_max\s*=\s*[0-9.]+', 'duration_max=2.0', hint)
                                            omax = re.search(r'onset_count_max\s*=\s*([0-9]+)', hint)
                                            if omax and int(omax.group(1)) > 8:
                                                hint = re.sub(r'onset_count_max\s*=\s*[0-9]+', 'onset_count_max=8', hint)
                                        except Exception:
                                            pass
                                        # Enforce role-based token_policy defaults and remove contradictory token policies
                                        try:
                                            def _enforce_token_policy(h: str, pol: str) -> str:
                                                parts_local = [p.strip() for p in h.split(';') if p.strip()]
                                                parts_local = [p for p in parts_local if not p.lower().startswith('token_policy=')]
                                                parts_local.append(f'token_policy={pol}')
                                                return '; '.join(parts_local)
                                            if role in ('whisper','spoken','phrase_spot','chorus','prechorus','verse','bridge','intro','outro','backing','harmony','doubles','response','rap','chant','tag'):
                                                # lexical-required roles
                                                hint = _enforce_token_policy(hint, 'lexical')
                                            elif role in ('vocal_fx','vocoder','talkbox','scat','vowels'):
                                                # vowel-based textures
                                                hint = _enforce_token_policy(hint, 'fx_vowel' if role=='vocal_fx' else 'vowel')
                                            elif role == 'breaths':
                                                hint = _enforce_token_policy(hint, 'br')
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                
                                mapped.append({
                                    "idx": i+1, 
                                    "role": role, 
                                    "plan_hint": hint,
                                    "hook_theme": "",
                                    "hook_canonical": "",
                                    "chorus_lines": []
                                })
                            else:
                                mapped.append({
                                    "idx": i+1, 
                                    "role": "silence", 
                                    "plan_hint": "Instrumental only. Maintain vocal silence.",
                                    "hook_theme": "",
                                    "hook_canonical": "",
                                    "chorus_lines": []
                                })
                        
                        return mapped
                    except Exception as e:
                        raise ValueError(f"Hint generation parse error: {e}")
                else:
                    raise ValueError("Empty response")
                    
            except Exception as e:
                err = str(e).lower()
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    try:
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        avail = []
                        now = time.time()
                        for ix, _ in enumerate(API_KEYS):
                            tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                            avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                        print(Fore.MAGENTA + Style.BRIGHT + "[Hint Generation:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                    except Exception:
                        pass
                    
                    # Try to rotate to next available key
                    n = len(API_KEYS)
                    rotated = False
                    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off*stride) % n
                        if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            rotated = True
                            break
                        except Exception:
                            continue
                    
                    if not rotated:
                        wait_s = 3 if qtype not in ('per-hour','per-day') else 15
                        _interruptible_backoff(wait_s, config, context_label="Hint Generation cooldown")
                        quota_rotation_count += 1
                        continue
                    else:
                        continue
                else:
                    # Non-quota error, short backoff
                    time.sleep(0.5)
                    continue
        
        raise ValueError("Hint generation failed: max attempts exceeded")
        
    except Exception as e:
        print(f"[Hint Generation Error] {e}")
        # No heuristic fallback - let the LLM handle all decisions
        # If LLM fails, return silence for all parts to maintain consistency
        out: List[Dict] = []
        for i, s in enumerate(summaries):
            role = roles[i].get('role', 'silence') if i < len(roles) else 'silence'
            out.append({
                "idx": s['idx'], 
                "role": role, 
                "plan_hint": "Instrumental only. Maintain vocal silence." if role == 'silence' else f"Vocal section with {role} characteristics.",
                "hook_theme": "",
                "hook_canonical": "",
                "chorus_lines": []
            })
        return out

def _plan_lyric_sections(config: Dict, genre: str, inspiration: str, bpm: float | int, ts: Dict, summaries: List[Dict], user_prompt: str | None = None) -> List[Dict]:
    """
    Ask LLM to propose a high-level per-part plan: role + plan_hint + optional hook_theme.
    Fallback: lightweight heuristic mapping by density/sustain.
    """
    try:
        import google.generativeai as genai_local
    except Exception:
        genai_local = None
    if not summaries:
        # Strict behavior: do not proceed without summaries
        raise ValueError("No summaries available for planning")
    
    if genai_local is None or not API_KEYS:
        # No planner available → hard fail per user preference
        raise RuntimeError("Planning unavailable: no LLM SDK or API key")
    # Build prompt
    # Use session override if available, otherwise config (no flash fallback)
    global SESSION_MODEL_OVERRIDE
    model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
    try:
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
    except Exception:
        pass
    if not model_name:
        model_name = config.get("model_name", "gemini-2.5-pro")
    # Plan generation should be highly deterministic
    _plan_temp = 0.0
    try:
        if isinstance(config.get("plan_temperature"), (int, float)):
            _plan_temp = float(config.get("plan_temperature"))
        elif isinstance(config.get("lyrics_temperature"), (int, float)):
            _plan_temp = float(config.get("lyrics_temperature"))
        elif isinstance(config.get("temperature"), (int, float)):
            _plan_temp = float(config.get("temperature"))
    except Exception:
        _plan_temp = 0.0
    generation_config = {"response_mime_type": "application/json", "temperature": _plan_temp}
    try:
        if isinstance(config.get("max_output_tokens"), int):
            _mx = int(config.get("max_output_tokens")); _mx = max(256, min(_mx, 8192)); generation_config["max_output_tokens"] = _mx
    except Exception:
        pass
    model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    # Allow model to propose per-part lyric prefs if enabled (default off)
    allow_prefs = False
    try:
        allow_prefs = bool(int(config.get('lyrics_prefs_from_ai', 0)))
    except Exception:
        allow_prefs = False
    prefs_schema = ""
    prefs_rules = ""
    # Global story/title suggestion
    title_hint = _infer_hook_from_text(user_prompt) if isinstance(user_prompt, str) else None
    # Debug toggle for verbose plan logging
    debug_plan = False
    try:
        debug_plan = bool(int(config.get('debug_plan_output', 0)))
    except Exception:
        debug_plan = False
    labels = _get_prompt_labels(config)
    plan_ctx_line = _format_prompt_context_line(
        {
            'genre': genre,
            'bpm': round(float(bpm)),
            'time_signature': f"{ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}"
        },
        labels=labels
    )
    # Build optional role catalog from config or default
    try:
        cfg_roles = config.get('lyrics_roles_catalog') if isinstance(config.get('lyrics_roles_catalog'), dict) else None
    except Exception:
        cfg_roles = None
    default_roles_catalog = {
        "intro": "Set tone economically; minimal gesture.",
        "verse": "Advance ideas; concrete imagery or rhythmic fragments.",
        "prechorus": "Tighten phrasing; lift into chorus; hint only.",
        "chorus": "Peak emphasis; title-drop if hook exists; else compact motif.",
        "bridge": "Contrast section; refresh color; avoid chorus wording.",
        "breakdown": "Reduce density; spotlight a single idea; leave space.",
        "backing": "Echo/answer lead with short repeatable fragments.",
        "scat": "Musical syllables; vary vowels; avoid monotony/full words.",
        "vowels": "Hold open vowels; no semantics; lift.",
        "spoken": "Spoken-word phrasing; natural prosody; few words.",
        "whisper": "Very soft, breathy; sparse tokens; intimacy.",
        "shout": "High energy emphasis; few words; rhythmic hits.",
        "chant": "Mantra-like repetition; short phrases; limited duration.",
        "adlib": "Interjections around lead; sporadic; supportive.",
        "harmony": "Parallel/supporting lines; no new semantics.",
        "doubles": "Unison/close double of lead fragments.",
        "response": "Call-and-response answers to lead.",
        "rap": "Rhythmic speech; clear enunciation; internal rhyme optional.",
        "choir": "Stacked voices; simple syllables/words; pad-like.",
        "hum": "Neutral humming; no words.",
        "breaths": "Notated breaths/inhales/exhales as musical cues.",
        "tag": "Short signature/tag at endings; minimal.",
        "phrase_spot": "One spotlight phrase; very brief.",
        "vocoder": "Processed vocals; simple contours/words.",
        "talkbox": "Talkbox articulation; simple syllables/words.",
        "vocal_fx": "FX-like vocalizations; texture role.",
        "silence": "Intentional no-vocal for this part."
    }
    roles_catalog_map = cfg_roles or default_roles_catalog
    try:
        # keep stable order by listing known keys first, then any extras
        known_order = list(default_roles_catalog.keys())
        extra_keys = [k for k in roles_catalog_map.keys() if k not in known_order]
        ordered_keys = known_order + extra_keys
        role_catalog_lines = [f"- {k}: {str(roles_catalog_map.get(k,''))}" for k in ordered_keys]
        roles_catalog_text = "\n".join(role_catalog_lines)
    except Exception:
        roles_catalog_text = "- intro: Set tone economically\n- verse: Advance ideas\n- chorus: Peak emphasis\n- silence: No vocals"

    # Soft synonym guidance for mapper, not enforced
    synonyms_soft = (
        "ROLE SYNONYMS (soft mapping examples):\n"
        "- 'instrumental only'/'no vocals'/'no vocal presence' ⇒ role='silence' (if consistent).\n"
        "- 'spoken word' ⇒ role='spoken'; 'whispered' ⇒ role='whisper'.\n"
        "- 'call and response' ⇒ role='response'; 'doo-wop/hum' ⇒ role='hum'.\n"
        "- 'vocoder'/'robotic vox' ⇒ role='vocoder'; 'talkbox' ⇒ role='talkbox'.\n\n"
    )

    consistency_soft = (
        "CONSISTENCY SELF-CHECK (MANDATORY):\n"
        "- CRITICAL: role and plan_hint MUST be consistent. NEVER use role='chorus' with 'instrumental only' or 'maintain vocal silence'.\n"
        "- If hint implies instrumental-only, you MUST set role='silence' (exact string).\n"
        "- If role='chorus', plan_hint must describe vocal content, not silence.\n"
        "- Only include hook_canonical/chorus_lines if a hook genuinely exists.\n\n"
    )

    prompt = (
        "You are a vocal arranger. Given a compact sketch of the vocal track across parts, propose a structure plan suited to the given genre/context.\n"
        + (plan_ctx_line + "\n")
        + (f"User intent (high-level; do NOT copy phrases): {user_prompt}\n" if (isinstance(user_prompt, str) and user_prompt.strip()) else "")
        + "Parts (order): {idx,label,num_notes,notes_density,avg_dur,max_dur,sustain_ratio,silent}.\n" + json.dumps(summaries[:64]) + "\n\n"
        + (f"Global title/theme suggestion: {title_hint}\n" if title_hint else "")
        + "Output STRICT JSON only. Return exactly ONE JSON object with keys:\n"
        + "- plan: REQUIRED array of objects; each object MUST include {\"idx\": int, \"role\": string, \"plan_hint\": string}. It MAY also include optional fields: hook_theme, hook_canonical, chorus_lines, repetition_policy, imagery_palette, verb_palette, call_and_response, chant_spots, story, lyrics_prefs.\n"
        + "- global: OPTIONAL object for shared metadata (e.g., hook_canonical, chorus_lines, repetition_policy, imagery_palette, verb_palette).\n"
        + "No code fences. No trailing commas. Use ONLY double quotes.\n\n"
        + "ROLES CATALOG (use EXACT strings; no synonyms):\n" + roles_catalog_text + "\n\n"
        + synonyms_soft
        + consistency_soft
        + "Rules:\n- Keep list length == number of parts provided (<=64).\n- role guides the later lyric style; plan_hint is a very short cue (few words) sized to the part; hook_theme only for chorus.\n- Abstraction: derive from sketch; do NOT repeat user words.\n- HOOK POLICY: Only include hook_canonical or chorus_lines if a hook is explicitly present (quoted or 'hook: ...') or the planner genuinely proposes one; otherwise omit them.\n- Do not include named artists or song titles as lyrics, unless the user explicitly requests a specific hook phrase (e.g., 'hook: ...'); in that case, use only that phrase as the hook and avoid other names.\n- META FILTER (planning): Do NOT turn instruction descriptors (style tags, genre labels, model/file names, parameter keys, or phrases like 'in the style of ...') into lyrics. Do not copy [Plan] tags verbatim. Never place such meta/descriptors into hook_canonical or chorus_lines.\n- SILENCE POLICY (MANDATORY): If user intent or context implies 'instrumental' / 'no vocals' / 'no vocal presence' / 'instrumental only', you MUST set role='silence' (exact string).\n- VOCAL BUDGET (target): Prefer 3-5 vocal parts total for a 16-part song (the remainder should be role='silence' or minimal 'backing'); keep at least 1-2 parts gap between vocal parts when possible.\n"
        + prefs_rules
        + "Songwriting guidance (soft):\n- Favor clear section functions: verses progress story, prechorus builds tension, chorus delivers peak emphasis (hook if present, else a concise motif), bridge adds contrast.\n- Encourage repetition in chorus only if a hook exists; otherwise prefer a compact central motif or rhythmic anchor.\n- Consider line-level phrasing; align phrase ends with rests/gaps.\n- Allow occasional vocalizations (oh/ahh/yeah) for expression; keep them musical.\n- Style flexibility: avoid locking into a single descriptive mode across the song.\n"
        + "Arrangement awareness (soft):\n- Complement other melodies; where texture is dense, consider planning fewer or no words.\n- Entire bars may remain empty if silence serves the arrangement.\n- Avoid prescribing audio effects (reverb/delay/etc.); keep instructions implementable via notes/words only.\n"
        + "FORM GUIDANCE (soft):\n- Macro-form: A (verse) → B (prechorus) → C (chorus) is common, but allow forms without a chorus/hook (e.g., motif or backing fragments).\n- Chorus lines: only if a hook exists; otherwise, give a short non-lyrical or motif-oriented plan_hint sized to part length.\n- Prechorus: a small number of priming lines or motif hints; reusing them before peaks is fine if it serves momentum.\n"
        + "ROLE MAPPING (soft preferences):\n- If a part label contains 'drop' (case-insensitive) or has among the highest notes_density, consider 'chorus' as a peak section; however, peak sections may be motif/backing-driven without a hook.\n- Use 'vowels' mainly where sustain_ratio is high; otherwise prefer 'verse'.\n"
        + "ROLE GUIDANCE (hard, English):\n- verse: Advance ideas; choose a fitting mode (concrete imagery, rhythmic fragments, etc.). Avoid chorus wording; create momentum.\n- prechorus: Tighten phrasing and raise tension; hint a coming peak without requiring a title-drop.\n- chorus: Deliver the title-drop only if a hook exists; otherwise emphasize a compact motif or concentrated idea.\n- bridge: Provide contrast; you may switch mode (e.g., from imagery to deadpan) to refresh color; avoid chorus wording.\n- breakdown: Reduce density and spotlight a single idea or feeling; minimal new wording; leave space.\n- backing: Echo or answer the lead with short, repeatable fragments; avoid introducing new content.\n- scat: Musical syllables supporting groove; vary vowels; avoid monotony and full words.\n- vowels: Hold open vowels on long notes; no semantics, pure sustain for lift.\n- intro: Set tone economically (one image or a minimal gesture).\n- outro: Resolve or fade with a small callback (title/tag) only if appropriate; minimal new wording.\n\n"
        + ""
        + "PLAN_HINT CONTENT (compact, JSON-like allowed; keep concise):\n- Include: narrative_beats=[few short cues], imagery_palette=[few], verb_palette=[few].\n- Optional: chorus_lines=[few very short lines] only if a hook exists; otherwise describe a motif/backing idea.\n- Optional: repetition_policy={chorus: 'fixed'|'varied', verse: 'low'|'medium'} if relevant.\n- Keep hints abstract; avoid full lyrics except brief chorus candidates when a hook exists.\n"
        + "PLAN_HINT LANGUAGE (hard):\n- Write plan_hint in ENGLISH as very brief sentences that instruct what the lyric should do in this part (imperative style).\n- Mention repetition usage qualitatively; if no hook exists, prefer motif/backing guidance.\n- Reference imagery/verb palette lightly.\n\n"
    )
    def _call_with_rotation(prompt_text: str) -> dict | None:
        max_attempts = max(3, len(API_KEYS) * 2)
        attempts = 0
        nonlocal_model = [model]
        while attempts < max_attempts:
            try:
                resp = nonlocal_model[0].generate_content(prompt_text, safety_settings=safety_settings, generation_config=generation_config)
                # Robust: extract from candidates/parts first, fallback to resp.text
                raw = _extract_text_from_response(resp) or ""
                try:
                    if debug_plan:
                        print(Style.DIM + "[Plan Raw Full]" + Style.RESET_ALL)
                        print(raw)
                except Exception:
                    pass
                cleaned = raw.strip().replace("```json", "").replace("```", "")
                try:
                    if debug_plan:
                        print(Style.DIM + "[Plan Cleaned Full]" + Style.RESET_ALL)
                        print(cleaned)
                except Exception:
                    pass
                # Prefer robust JSON extraction first
                try:
                    payload = _extract_json_object(cleaned)
                except Exception:
                    payload = ""
                if payload:
                    try:
                        try:
                            if debug_plan:
                                print(Style.DIM + "[Plan Payload Substring]" + Style.RESET_ALL)
                                print(payload)
                        except Exception:
                            pass
                        # Try strict first; if it fails, run a gentle sanitizer
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            obj = json.loads(_sanitize_json_text_for_load(payload))
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        pass
                # Fallback: naive brace salvage
                try:
                    start = cleaned.find('{'); end = cleaned.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        raw_slice = cleaned[start:end+1]
                        try:
                            obj2 = json.loads(raw_slice)
                        except Exception:
                            obj2 = json.loads(_sanitize_json_text_for_load(raw_slice))
                        return obj2 if isinstance(obj2, dict) else None
                except Exception:
                    pass
                # If we reached here, we did not get valid JSON → retry (optionally rotate key)
                try:
                    print(Fore.YELLOW + f"[Plan Retry {attempts}/{max_attempts}] Non-JSON response; retrying..." + Style.RESET_ALL)
                except Exception:
                    pass
                attempts += 1
                if len(API_KEYS) > 1:
                    rotated = False
                    try:
                        n = len(API_KEYS)
                        stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                        for off in range(1, n+1):
                            idx = (CURRENT_KEY_INDEX + off*stride) % n
                            if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                                continue
                            try:
                                globals()['CURRENT_KEY_INDEX'] = idx
                                genai.configure(api_key=API_KEYS[idx])
                                nonlocal_model[0] = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                                rotated = True
                                break
                            except Exception:
                                continue
                    except Exception:
                        rotated = False
                time.sleep(0.5)
                continue
            except Exception as e:
                err = str(e).lower()
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    try:
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        avail = []
                        now = time.time()
                        for ix, _ in enumerate(API_KEYS):
                            tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                            avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                        print(Fore.MAGENTA + Style.BRIGHT + "[Plan:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                    except Exception:
                        pass
                    n = len(API_KEYS); rotated = False
                    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off*stride) % n
                        if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            nonlocal_model[0] = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                            rotated = True; break
                        except Exception:
                            continue
                    if not rotated:
                        wait_s = 3 if qtype not in ('per-hour','per-day') else 15
                        _interruptible_backoff(wait_s, config, context_label="Lyrics plan cooldown"); continue
                # Non-quota error counts toward attempts
                attempts += 1
                time.sleep(1); continue
        return None
    try:
        est_toks = (len(prompt) + 3) // 4
        print(Style.DIM + f"[Lyrics-Plan] Prompt size: {len(prompt)} chars, ~{est_toks} tokens" + Style.RESET_ALL)
    except Exception:
        pass
    obj = _call_with_rotation(prompt)
    if not isinstance(obj, dict):
        # Invalid model response → hard fail per user preference
        raise ValueError("Invalid planning response (no JSON object)")
    global_meta = obj.get('global') if isinstance(obj.get('global'), dict) else {}
    plan = obj.get('plan')
    if not isinstance(plan, list) or not plan:
        # Empty/invalid plan → hard fail per user preference
        raise ValueError("Planning returned empty/invalid plan list")
    # Normalize plan length to expected number of parts (tolerant to model drift)
    expected_n = len(summaries)
    if len(plan) != expected_n:
        try:
            print(Fore.YELLOW + f"[Plan WARN] Mismatch length: got {len(plan)} items, expected {expected_n} — normalizing." + Style.RESET_ALL)
        except Exception:
            pass
        try:
            # Detect 1-based indexing and remap if appropriate
            idx_vals = [int(it.get('idx')) for it in plan if isinstance(it, dict) and isinstance(it.get('idx'), (int, float, str)) and str(it.get('idx')).strip().lstrip('-').isdigit()]
            if idx_vals:
                mn, mx = min(idx_vals), max(idx_vals)
                if mn == 1 and mx == expected_n:
                    for it in plan:
                        try:
                            it['idx'] = int(it.get('idx', 0)) - 1
                        except Exception:
                            continue
        except Exception:
            pass
        # Build index map and fill gaps
        index_to_item = {}
        for it in plan:
            try:
                i = int(it.get('idx', -1))
            except Exception:
                i = -1
            if 0 <= i < expected_n and i not in index_to_item:
                index_to_item[i] = it
        normalized = []
        for i in range(expected_n):
            if i in index_to_item:
                normalized.append(index_to_item[i])
            else:
                # Heuristic default for missing items
                is_silent = bool((summaries[i] or {}).get('silent'))
                role_def = 'silence' if is_silent else 'verse'
                normalized.append({"idx": i, "role": role_def, "plan_hint": "", "hook_theme": ""})
        plan = normalized
    # Print only a brief OK line; the detailed display follows later at the call site.
    try:
        print(Style.BRIGHT + "[Plan OK]" + Style.RESET_ALL)
        if debug_plan:
            header = f"{'idx':>3}  {'role':<10}  plan_hint"
            print(Style.DIM + header + Style.RESET_ALL)
            for it in plan:
                try:
                    ridx = it.get('idx')
                    role = (it.get('role') or '')[:10]
                    hint = (it.get('plan_hint') or '')
                    hint = hint.replace('\n',' ').strip()
                    if len(hint) > 88:
                        hint = hint[:85] + '…'
                    print(f"{str(ridx).rjust(3)}  {role.ljust(10)}  {hint}")
                except Exception:
                    continue
    except Exception:
        pass
    # Normalize
    mapped = []
    for it in plan[:len(summaries)]:
        try:
            ridx = int(it.get('idx', len(mapped)))
            role = str(it.get('role', 'verse')).lower()
            hint = str(it.get('plan_hint', ''))
            hook = str(it.get('hook_theme', ''))
            hook_can = None
            try:
                hc = it.get('hook_canonical')
                hook_can = (str(hc) if isinstance(hc, str) else None)
            except Exception:
                hook_can = None
            lp = None
            if allow_prefs:
                try:
                    p = it.get('lyrics_prefs')
                    if isinstance(p, dict):
                        lp = {
                            "target_wpb": (float(p.get('target_wpb')) if isinstance(p.get('target_wpb'), (int, float)) else None),
                            "melisma_bias": (max(0.25, min(0.55, float(p.get('melisma_bias')))) if isinstance(p.get('melisma_bias'), (int, float)) else None),
                            "min_word_beats": (float(p.get('min_word_beats')) if isinstance(p.get('min_word_beats'), (int, float)) else None),
                            "allow_nonsense": (int(p.get('allow_nonsense')) if isinstance(p.get('allow_nonsense'), (int, float)) else None)
                        }
                except Exception:
                    lp = None
            mapped_item = {"idx": ridx, "role": role, "plan_hint": hint, "hook_theme": hook}
            if hook_can:
                # sanitize meta-like phrases from hook_canonical (planning stage)
                hc_norm = re.sub(r"\s+", " ", hook_can).strip()
                if not re.search(r"\b(style|genre|model|file|prompt|lyrics?)\b", hc_norm, re.IGNORECASE):
                    mapped_item["hook_canonical"] = hc_norm
            # Optional extended fields from LLM plan
            try:
                if isinstance(it.get('story'), str):
                    mapped_item['story'] = it.get('story')
                if isinstance(it.get('chorus_lines'), list):
                    cl = []
                    for x in it.get('chorus_lines'):
                        if isinstance(x, str):
                            t = re.sub(r"\s+", " ", x).strip()
                            if not re.search(r"\b(style|genre|model|file|prompt|lyrics?)\b", t, re.IGNORECASE):
                                cl.append(t)
                    if cl:
                        mapped_item['chorus_lines'] = cl
                if isinstance(it.get('repetition_policy'), dict):
                    mapped_item['repetition_policy'] = {k: str(v) for k, v in it.get('repetition_policy').items() if isinstance(k, str)}
                if isinstance(it.get('imagery_palette'), list):
                    mapped_item['imagery_palette'] = [str(x) for x in it.get('imagery_palette') if isinstance(x, str)]
                if isinstance(it.get('verb_palette'), list):
                    mapped_item['verb_palette'] = [str(x) for x in it.get('verb_palette') if isinstance(x, str)]
                if isinstance(it.get('call_and_response'), str):
                    mapped_item['call_and_response'] = it.get('call_and_response')
                if isinstance(it.get('chant_spots'), list):
                    mapped_item['chant_spots'] = [str(x) for x in it.get('chant_spots') if isinstance(x, str)]
            except Exception:
                pass
            if lp is not None:
                mapped_item['lyrics_prefs'] = lp
            mapped.append(mapped_item)
        except Exception:
            mapped.append({"idx": len(mapped), "role": "verse", "plan_hint": "", "hook_theme": ""})
    if len(mapped) != len(summaries):
        # pad/truncate to match summaries length
        while len(mapped) < len(summaries):
            mapped.append({"idx": len(mapped)+1, "role": "silence", "plan_hint": "Instrumental only. Maintain vocal silence.", "hook_theme": ""})
        mapped = mapped[:len(summaries)]
    # Debug: before roles
    try:
        roles_before = [x.get('role','') for x in mapped]
        if debug_plan:
            print(Style.DIM + "[Plan Normalize] roles before: " + ", ".join(roles_before[:24]) + (" …" if len(roles_before)>24 else "") + Style.RESET_ALL)
    except Exception:
        roles_before = []
    # Post-normalization: enforce at least two chorus parts and prechorus before each chorus
    try:
        # Gather feature arrays
        density = [float(s.get('notes_density', 0.0)) for s in summaries]
        sustain = [float(s.get('sustain_ratio', 0.0)) for s in summaries]
        labels = [str(s.get('label', '')).lower() for s in summaries]
        roles = [x.get('role', 'verse') for x in mapped]
        n = len(roles)
        # Map obvious labels to chorus/prechorus by keywords
        for i in range(n):
            lab = labels[i]
            if any(k in lab for k in ('drop', 'chorus', 'hook')):
                roles[i] = 'chorus'
            elif any(k in lab for k in ('build', 'buildup', 'rise', 'pre', 'lift')) and roles[i] not in ('chorus','silence'):
                roles[i] = 'prechorus'
        # Ensure at least two peak sections (chorus role) by density peaks if none/one present
        chorus_idx = [i for i, r in enumerate(roles) if r == 'chorus']
        if len(chorus_idx) < 2 and n >= 4:
            order = sorted(range(n), key=lambda i: density[i], reverse=True)
            for i in order:
                if roles[i] != 'chorus':
                    roles[i] = 'chorus'
                    chorus_idx.append(i)
                    if len(chorus_idx) >= 2:
                        break
        # Place prechorus before each chorus when possible
        for ci in chorus_idx:
            j = ci - 1
            if 0 <= j < n and roles[j] not in ('chorus', 'prechorus', 'silence') and summaries[j].get('silent') is not True:
                roles[j] = 'prechorus'
        # Constrain vowels usage
        for i in range(n):
            if roles[i] == 'vowels' and sustain[i] < 0.5:
                roles[i] = 'verse'
        # Write back
        for i in range(n):
            mapped[i]['role'] = roles[i]
        # Enforce role whitelist: any unknown role → silence
        try:
            allowed_roles = {"intro","verse","prechorus","chorus","bridge","breakdown","backing","scat","vowels","silence","spoken","whisper","shout","chant","adlib","harmony","doubles","response","rap","choir","hum","breaths","tag","phrase_spot","vocoder","talkbox","vocal_fx"}
            for i in range(n):
                ri = str(mapped[i].get('role','')).lower()
                if ri not in allowed_roles:
                    mapped[i]['role'] = 'silence'
                    mapped[i]['plan_hint'] = 'Instrumental only. Maintain vocal silence.'
                    mapped[i]['hook_theme'] = ''
                    mapped[i].pop('hook_canonical', None)
        except Exception:
            pass
        # Enforce vocal budget pruning (limit number of vocal-bearing parts)
        try:
            target_max_vocal_parts = int((config.get('max_vocal_parts') if isinstance(config.get('max_vocal_parts'), (int,float)) else 4))
        except Exception:
            target_max_vocal_parts = 4
        try:
            min_gap_parts = int((config.get('min_gap_parts_between_vocals') if isinstance(config.get('min_gap_parts_between_vocals'), (int,float)) else 2))
        except Exception:
            min_gap_parts = 2
        # score vocal candidates
        scored = []
        for i in range(n):
            r = roles[i]
            dens = float(summaries[i].get('notes_density', 0.0))
            silent_flag = summaries[i].get('silent') is True
            score = 100.0
            if r == 'breakdown': score -= 60
            if r == 'verse': score -= 40
            if r == 'backing': score -= 20
            if r == 'chorus': score += 40
            if silent_flag: score += 100
            score += dens * 5
            scored.append((score, i))
        scored.sort()
        selected = []
        for _, idx in scored:
            if len(selected) >= max(0, target_max_vocal_parts):
                break
            if summaries[idx].get('silent') is True:
                continue
            if any(abs(idx - j) < min_gap_parts for j in selected):
                continue
            selected.append(idx)
        # apply pruning
        for i in range(n):
            if i in selected:
                continue
            if roles[i] in ('verse','prechorus','chorus','bridge','breakdown','backing'):
                roles[i] = 'silence'
                mapped[i]['role'] = 'silence'
                mapped[i]['plan_hint'] = 'Instrumental only. Maintain vocal silence.'
                mapped[i]['hook_theme'] = ''
                mapped[i].pop('hook_canonical', None)
        # Hook is optional: do NOT force hook_canonical if absent in plan or user intent
        # If a hook was explicitly provided (quoted/user or planner), it remains; otherwise keep empty
        # Force 'silence' role for parts with no notes; otherwise, sanitize conflicting hints without changing role
        for i in range(n):
            try:
                if summaries[i].get('silent'):
                    mapped[i]['role'] = 'silence'
                    if isinstance(mapped[i], dict):
                        mapped[i]['plan_hint'] = ''
                        mapped[i]['hook_theme'] = ''
                        if 'lyrics_prefs' in mapped[i]:
                            mapped[i].pop('lyrics_prefs', None)
                else:
                    try:
                        ph = str(mapped[i].get('plan_hint','')).lower()
                    except Exception:
                        ph = ''
                    # If a non-silence role carries an instrumental-only hint, clear it so the call site can fill a standard role hint
                    if mapped[i].get('role','') != 'silence' and any(k in ph for k in ('instrumental only','instrumental focus','no vocals','no vocal presence','maintain vocal silence','purely instrumental')):
                        mapped[i]['plan_hint'] = ''
            except Exception:
                pass
        # Debug: after roles + chorus positions
        try:
            if debug_plan:
                print(Style.DIM + "[Plan Normalize] roles after:  " + ", ".join(roles[:24]) + (" …" if len(roles)>24 else "") + Style.RESET_ALL)
                ch = [str(i+1) for i, r in enumerate(roles) if r == 'chorus']
                pr = [str(i+1) for i, r in enumerate(roles) if r == 'prechorus']
                print(Style.DIM + f"[Plan Normalize] chorus at parts: {', '.join(ch) if ch else '-'}; prechorus at parts: {', '.join(pr) if pr else '-'}" + Style.RESET_ALL)
        except Exception:
            pass
    except Exception:
        pass
    # Early, concise Step-0 meta summary at start of the run
    try:
        print(Style.BRIGHT + "\n[Plan Summary]" + Style.RESET_ALL)
        # Hook canonical (unique)
        hooks_can = []
        seen_hc = set()
        for it in mapped:
            hc = it.get('hook_canonical') if isinstance(it, dict) else None
            if isinstance(hc, str):
                t = hc.strip()
                if t and t not in seen_hc:
                    seen_hc.add(t); hooks_can.append(t)
        # Include from global_meta if present
        try:
            if isinstance(global_meta, dict):
                ghc = global_meta.get('hook_canonical')
                if isinstance(ghc, str):
                    t = ghc.strip()
                    if t and t not in seen_hc:
                        seen_hc.add(t); hooks_can.append(t)
        except Exception:
            pass
        if hooks_can:
            print(Style.BRIGHT + "Hook Canonical:" + Style.RESET_ALL, ", ".join([f'\"{s}\"' if '"' not in s else s for s in hooks_can[:2]]) + (" …" if len(hooks_can) > 2 else ""))
        else:
            print(Style.BRIGHT + "Hook Canonical:" + Style.RESET_ALL, "-")
        # Hook themes (unique)
        hooks = []
        seen_h = set()
        for it in mapped:
            ht = it.get('hook_theme') if isinstance(it, dict) else None
            if isinstance(ht, str):
                t = ht.strip()
                if t and t not in seen_h:
                    seen_h.add(t); hooks.append(t)
        try:
            if isinstance(global_meta, dict):
                ght = global_meta.get('hook_theme')
                if isinstance(ght, str):
                    t = ght.strip()
                    if t and t not in seen_h:
                        seen_h.add(t); hooks.append(t)
        except Exception:
            pass
        if hooks:
            print(Style.BRIGHT + "Hook Themes:" + Style.RESET_ALL, ", ".join(hooks[:3]) + (" …" if len(hooks) > 3 else ""))
        else:
            print(Style.BRIGHT + "Hook Themes:" + Style.RESET_ALL, "-")
        # Chorus lines (unique)
        lines_seen = []
        lines_set = set()
        for it in mapped:
            cls = it.get('chorus_lines') if isinstance(it, dict) else None
            if isinstance(cls, list):
                for s in cls:
                    if isinstance(s, str):
                        t = s.strip()
                        if t and t not in lines_set:
                            lines_set.add(t); lines_seen.append(t)
        # From global_meta
        try:
            if isinstance(global_meta, dict):
                gcls = global_meta.get('chorus_lines')
                if isinstance(gcls, list):
                    for s in gcls:
                        if isinstance(s, str):
                            t = s.strip()
                            if t and t not in lines_set:
                                lines_set.add(t); lines_seen.append(t)
        except Exception:
            pass
        # Fallback: parse from plan_hint patterns
        try:
            import re as _re
            for it in mapped:
                hint = it.get('plan_hint') if isinstance(it, dict) else None
                if isinstance(hint, str):
                    m = _re.search(r"chorus_lines\s*=\s*\[(.*?)\]", hint)
                    if m:
                        inner = m.group(1)
                        # split by comma, accept quoted or unquoted
                        raw_items = [x.strip().strip('"\'') for x in inner.split(',') if x.strip()]
                        for ri in raw_items:
                            if ri and ri not in lines_set:
                                lines_set.add(ri); lines_seen.append(ri)
        except Exception:
            pass
        if lines_seen:
            print(Style.BRIGHT + "Chorus Lines:" + Style.RESET_ALL, " | ".join(lines_seen[:3]) + (" …" if len(lines_seen) > 3 else ""))
        else:
            print(Style.BRIGHT + "Chorus Lines:" + Style.RESET_ALL, "-")
        # Repetition policy (top pairs)
        from collections import Counter
        pairs = []
        for it in mapped:
            rp = it.get('repetition_policy') if isinstance(it, dict) else None
            if isinstance(rp, dict):
                for k, v in rp.items():
                    if isinstance(k, str):
                        pairs.append(f"{k}={v}")
        # From global_meta
        try:
            if isinstance(global_meta, dict):
                grp = global_meta.get('repetition_policy')
                if isinstance(grp, dict):
                    for k, v in grp.items():
                        if isinstance(k, str):
                            pairs.append(f"{k}={v}")
        except Exception:
            pass
        # Fallback: parse from plan_hint
        try:
            import re as _re
            for it in mapped:
                hint = it.get('plan_hint') if isinstance(it, dict) else None
                if isinstance(hint, str):
                    m = _re.search(r"repetition_policy\s*=\s*\{(.*?)\}", hint)
                    if m:
                        inner = m.group(1)
                        # split pairs by comma
                        for pair in inner.split(','):
                            if ':' in pair or '=' in pair:
                                kv = pair.replace(':', '=').strip()
                                if kv and kv not in pairs:
                                    pairs.append(kv)
        except Exception:
            pass
        if pairs:
            cnt = Counter(pairs)
            top_rep = ", ".join([f"{p}({c})" for p, c in cnt.most_common(3)])
            print(Style.BRIGHT + "Repetition:" + Style.RESET_ALL, top_rep)
        else:
            print(Style.BRIGHT + "Repetition:" + Style.RESET_ALL, "-")
        # Palettes
        imagery = []
        verbs = []
        seen_i = set(); seen_v = set()
        for it in mapped:
            ip = it.get('imagery_palette') if isinstance(it, dict) else None
            if isinstance(ip, list):
                for s in ip:
                    if isinstance(s, str):
                        t = s.strip()
                        if t and t not in seen_i:
                            seen_i.add(t); imagery.append(t)
            vp = it.get('verb_palette') if isinstance(it, dict) else None
            if isinstance(vp, list):
                for s in vp:
                    if isinstance(s, str):
                        t = s.strip()
                        if t and t not in seen_v:
                            seen_v.add(t); verbs.append(t)
        # From global_meta
        try:
            if isinstance(global_meta, dict):
                gip = global_meta.get('imagery_palette')
                if isinstance(gip, list):
                    for s in gip:
                        if isinstance(s, str):
                            t = s.strip()
                            if t and t not in seen_i:
                                seen_i.add(t); imagery.append(t)
                gvp = global_meta.get('verb_palette')
                if isinstance(gvp, list):
                    for s in gvp:
                        if isinstance(s, str):
                            t = s.strip()
                            if t and t not in seen_v:
                                seen_v.add(t); verbs.append(t)
        except Exception:
            pass
        # Fallback: parse palettes from plan_hint text
        try:
            import re as _re
            for it in mapped:
                hint = it.get('plan_hint') if isinstance(it, dict) else None
                if isinstance(hint, str):
                    mi = _re.search(r"imagery_palette\s*=\s*\[(.*?)\]", hint)
                    if mi:
                        inner = mi.group(1)
                        for x in inner.split(','):
                            t = x.strip().strip('"\'')
                            if t and t not in seen_i:
                                seen_i.add(t); imagery.append(t)
                    mv = _re.search(r"verb_palette\s*=\s*\[(.*?)\]", hint)
                    if mv:
                        inner = mv.group(1)
                        for x in inner.split(','):
                            t = x.strip().strip('"\'')
                            if t and t not in seen_v:
                                seen_v.add(t); verbs.append(t)
        except Exception:
            pass
        if imagery:
            print(Style.BRIGHT + "Imagery:" + Style.RESET_ALL, ", ".join(imagery[:5]) + (" …" if len(imagery) > 5 else ""))
        else:
            print(Style.BRIGHT + "Imagery:" + Style.RESET_ALL, "-")
        if verbs:
            print(Style.BRIGHT + "Verbs:" + Style.RESET_ALL, ", ".join(verbs[:5]) + (" …" if len(verbs) > 5 else ""))
        else:
            print(Style.BRIGHT + "Verbs:" + Style.RESET_ALL, "-")
        # Call-and-response / Chant
        cars = []
        chants = []
        for it in mapped:
            ca = it.get('call_and_response') if isinstance(it, dict) else None
            if isinstance(ca, str) and ca.strip():
                cars.append(ca.strip())
            chs = it.get('chant_spots') if isinstance(it, dict) else None
            if isinstance(chs, list):
                for s in chs:
                    if isinstance(s, str) and s.strip():
                        chants.append(s.strip())
        # From global_meta
        try:
            if isinstance(global_meta, dict):
                gca = global_meta.get('call_and_response')
                if isinstance(gca, str) and gca.strip():
                    cars.append(gca.strip())
                gchs = global_meta.get('chant_spots')
                if isinstance(gchs, list):
                    for s in gchs:
                        if isinstance(s, str) and s.strip():
                            chants.append(s.strip())
        except Exception:
            pass
        # Fallback: parse from plan_hint text
        try:
            import re as _re
            for it in mapped:
                hint = it.get('plan_hint') if isinstance(it, dict) else None
                if isinstance(hint, str):
                    m = _re.search(r"call_and_response\s*=\s*([\"'])(.*?)\1", hint)
                    if m:
                        val = m.group(2).strip()
                        if val:
                            cars.append(val)
                    m2 = _re.search(r"chant_spots\s*=\s*\[(.*?)\]", hint)
                    if m2:
                        inner = m2.group(1)
                        for x in inner.split(','):
                            t = x.strip().strip('"\'')
                            if t:
                                chants.append(t)
        except Exception:
            pass
        if cars:
            uniq = []
            seen_c = set()
            for c in cars:
                if c not in seen_c:
                    seen_c.add(c); uniq.append(c)
            print(Style.BRIGHT + "Call&Response:" + Style.RESET_ALL, ", ".join(uniq[:3]) + (" …" if len(uniq) > 3 else ""))
        else:
            print(Style.BRIGHT + "Call&Response:" + Style.RESET_ALL, "-")
        if chants:
            print(Style.BRIGHT + "Chant Spots:" + Style.RESET_ALL, ", ".join(chants[:3]) + (" …" if len(chants) > 3 else ""))
        else:
            print(Style.BRIGHT + "Chant Spots:" + Style.RESET_ALL, "-")
        # Story (short)
        stories = []
        seen_s = set()
        for it in mapped:
            st = it.get('story') if isinstance(it, dict) else None
            if isinstance(st, str):
                t = st.replace('\n',' ').strip()
                if t and t not in seen_s:
                    seen_s.add(t); stories.append(t)
        if stories:
            def _shorten(x: str) -> str:
                return x if len(x) <= 140 else (x[:137] + '…')
            print(Style.BRIGHT + "Story:" + Style.RESET_ALL, " | ".join([_shorten(s) for s in stories[:2]]) + (" …" if len(stories) > 2 else ""))
        else:
            print(Style.BRIGHT + "Story:" + Style.RESET_ALL, "-")
        # Lyrics prefs (aggregate)
        kvs = []
        for it in mapped:
            lp = it.get('lyrics_prefs') if isinstance(it, dict) else None
            if isinstance(lp, dict) and lp:
                wpb = lp.get('target_wpb'); mb = lp.get('melisma_bias'); mwb = lp.get('min_word_beats'); an = lp.get('allow_nonsense')
                if isinstance(wpb, (int, float)):
                    kvs.append(f"wpb={round(float(wpb),2)}")
                if isinstance(mb, (int, float)):
                    kvs.append(f"melisma_bias={round(float(mb),2)}")
                if isinstance(mwb, (int, float)):
                    kvs.append(f"min_word_beats={round(float(mwb),2)}")
                if isinstance(an, (int, float)):
                    kvs.append(f"allow_nonsense={int(an)}")
        if kvs:
            cnt2 = Counter(kvs)
            top_lp = ", ".join([f"{p}({c})" for p, c in cnt2.most_common(4)])
            print(Style.BRIGHT + "Lyrics Prefs:" + Style.RESET_ALL, top_lp)
        else:
            print(Style.BRIGHT + "Lyrics Prefs:" + Style.RESET_ALL, "-")
        print()
    except Exception:
        pass
    
    
    return mapped

def _generate_lyrics_words_with_spans(config: Dict, genre: str, inspiration: str, track_name: str, bpm: int | float, ts: Dict, notes: List[Dict], section_label: str | None = None, section_description: str | None = None, context_tracks_basic: List[Dict] | None = None, user_prompt: str | None = None, history_context: str | None = None, cfg: Dict | None = None) -> List[str]:
    """
    Word-first lyric generation with melisma spans.
    Returns a per-note token list equal to len(notes): first note of a word gets the word; continuation notes get '-'.
    """
    try:
        import google.generativeai as genai_local
    except Exception:
        genai_local = None

    N = len(notes)
    if N <= 0:
        return []

    # Force reload config to get latest model_name setting
    try:
        fresh_config = load_config(CONFIG_FILE)
        config = fresh_config
    except Exception as e:
        pass
    
    # Reset API key cooldowns for fresh keys
    try:
        _reset_all_cooldowns()
    except Exception:
        pass

    # Build compact note preview with stress as above, and a full ordered note list
    try:
        beats_per_bar = int(ts.get("beats_per_bar", 4))
        def _stress_for(start_beat: float) -> int:
            try:
                pos = start_beat % max(1, beats_per_bar)
                if abs(pos - 0) < 1e-3:
                    return 1
                if beats_per_bar % 2 == 0 and abs(pos - (beats_per_bar/2)) < 1e-3:
                    return 1
                return 0
            except Exception:
                return 0
        ordered_notes = sorted(notes, key=lambda x: float(x.get("start_beat", 0.0)))
        preview = []
        full_notes = []
        for idx, n in enumerate(ordered_notes):
            s = float(n.get("start_beat", 0.0))
            d = max(0.0, float(n.get("duration_beats", 0.0)))
            p = int(n.get("pitch", 60))
            st = _stress_for(s)
            preview.append({"start": round(s,3), "dur": round(d,3), "pitch": p, "stress": st})
            full_notes.append({"i": idx, "start": round(s,3), "dur": round(d,3), "pitch": p, "stress": st})
        # Keep preview short for heuristics, but expose full ordered list in the prompt
        preview = preview[:256]
    except Exception:
        preview = []
        full_notes = []

    # Extract musical parameters from JSON (not config.yaml)
    language = str(cfg.get("lyrics_language", "English")) if cfg else "English"
    # Get key_scale from cfg (no key_scale parameter in this function)
    key_scale = str(cfg.get("key_scale", "")).strip() if cfg else ""
    vocab_ctx = {"context_instruments": (context_tracks_basic or []), "style_keywords": [genre, inspiration][:8]}

    # Extract lyrics preferences from JSON (not config.yaml)
    cfg_target_wpb = cfg.get('lyrics_target_words_per_bar') if cfg else None
    cfg_melisma_bias = cfg.get('lyrics_melisma_bias') if cfg else None
    cfg_min_word_beats = cfg.get('lyrics_min_word_beats') if cfg else None
    cfg_allow_nonsense = cfg.get('lyrics_allow_nonsense') if cfg else None
    
    # Validate and convert to appropriate types
    try:
        if cfg_target_wpb is not None:
            cfg_target_wpb = float(cfg_target_wpb)
        if cfg_melisma_bias is not None:
            cfg_melisma_bias = max(0.0, min(1.0, float(cfg_melisma_bias)))
        if cfg_min_word_beats is not None:
            cfg_min_word_beats = float(cfg_min_word_beats)
        if cfg_allow_nonsense is not None:
            cfg_allow_nonsense = int(cfg_allow_nonsense) == 1
    except (ValueError, TypeError):
        # Reset to None if conversion fails
        cfg_target_wpb = cfg_melisma_bias = cfg_min_word_beats = cfg_allow_nonsense = None

    # Estimate bars in this segment for guidance (heuristic)
    est_bars = None
    try:
        if preview:
            s0 = preview[0]["start"]
            se = max(p["start"] + p["dur"] for p in preview)
            total_beats = max(0.0, se - s0)
            bpb = max(1, int(ts.get("beats_per_bar", 4)))
            est_bars = max(1.0, total_beats / bpb)
    except Exception:
        est_bars = None

    # Auto-derive preferences (always) and gently blend with config hints if provided
    try:
        # Compute quick local metrics
        bpb = int(ts.get("beats_per_bar", 4)) if isinstance(ts, dict) else 4
        note_durs = [float(p.get("dur", 0.0)) for p in (preview or [])]
        note_long_ratio = (sum(1 for d in note_durs if d >= 1.0) / max(1, len(note_durs))) if note_durs else 0.0
        notes_per_bar = (len(preview) / est_bars) if (preview and isinstance(est_bars, float) and est_bars > 0) else 0.0
        short_note_ratio = (sum(1 for d in note_durs if d < 0.33) / max(1, len(note_durs))) if note_durs else 0.0
        # Derive by section role baseline
        role = _normalize_section_role(section_label)
        base_wpb = {
            'chorus': 3.2,
            'prechorus': 2.8,
            'verse': 2.6,
            'bridge': 2.2,
            'backing': 1.8,
            'scat': 1.8,
            'vowels': 1.4,
            'intro': 1.6,
            'outro': 1.6,
        }.get(role, 2.6)
        # Density-driven adjustments
        wpb_adj = 0.0
        if notes_per_bar >= 6.0:
            wpb_adj -= 0.8
        elif notes_per_bar <= 2.0:
            wpb_adj += 0.4
        derived_target_wpb = max(1.0, min(6.0, base_wpb + wpb_adj))

        base_mb = 0.25 if role not in ("vowels", "intro", "outro") else 0.35
        mb_adj = 0.4*note_long_ratio + 0.5*min(1.0, max(0.0, (notes_per_bar-4.0)/4.0)) + 0.3*short_note_ratio
        # Chorus: stronger sustain for carpets of short notes
        if role == 'chorus' and short_note_ratio >= 0.4:
            mb_adj += 0.2
        # More natural phrasing - allow higher melisma for better flow
        derived_melisma_bias = max(0.25, min(0.45, base_mb + mb_adj + 0.05))

        if note_durs:
            nd_sorted = sorted(note_durs)
            med = nd_sorted[len(nd_sorted)//2]
            # Higher push with many short notes (especially chorus) to avoid choppy mapping
            derived_min_word_beats = max(1.0, min(1.3, 0.5*med + 0.45 + 0.25*short_note_ratio + (0.1 if role=='chorus' and short_note_ratio>=0.4 else 0.0)))
        else:
            derived_min_word_beats = 0.6

        derived_allow_nonsense = (role in ("scat", "vowels", "backing"))
        # For a very fragmented chorus allow slightly fewer target words per bar
        if role == 'chorus' and short_note_ratio >= 0.4:
            derived_target_wpb = max(1.0, derived_target_wpb - 0.4)

        # Blend with config hints (25% toward config if provided)
        blend = 0.25
        target_wpb = (
            blend*cfg_target_wpb + (1.0-blend)*derived_target_wpb
            if isinstance(cfg_target_wpb, (int, float)) else derived_target_wpb
        )
        melisma_bias = (
            blend*min(0.75, max(0.45, float(cfg_melisma_bias))) + (1.0-blend)*derived_melisma_bias
            if isinstance(cfg_melisma_bias, (int, float)) else derived_melisma_bias
        )
        min_word_beats = (
            blend*max(1.0, min(1.3, float(cfg_min_word_beats))) + (1.0-blend)*derived_min_word_beats
            if isinstance(cfg_min_word_beats, (int, float)) else derived_min_word_beats
        )
        allow_nonsense = bool(cfg_allow_nonsense) or bool(derived_allow_nonsense)
        
        # Debug: Show which values are being used
        print(f"[DEBUG] Lyrics parameters from JSON: target_wpb={target_wpb:.2f}, melisma_bias={melisma_bias:.2f}, min_word_beats={min_word_beats:.2f}")
    except (ValueError, TypeError, KeyError) as e:
        # Specific fallbacks for JSON extraction errors
        print(Fore.YELLOW + f"[WARNING] JSON extraction failed for lyrics parameters: {e}" + Style.RESET_ALL)
        target_wpb = target_wpb if 'target_wpb' in locals() and isinstance(target_wpb, (int, float)) else 2.6
        melisma_bias = melisma_bias if 'melisma_bias' in locals() and isinstance(melisma_bias, (int, float)) else 0.35
        min_word_beats = min_word_beats if 'min_word_beats' in locals() and isinstance(min_word_beats, (int, float)) else 1.0
        allow_nonsense = allow_nonsense if 'allow_nonsense' in locals() else False
        print(f"[DEBUG] Using FALLBACK lyrics parameters: target_wpb={target_wpb:.2f}, melisma_bias={melisma_bias:.2f}, min_word_beats={min_word_beats:.2f}")
    except Exception as e:
        # Critical fallbacks for unexpected errors
        print(Fore.RED + f"[ERROR] Unexpected error in lyrics parameter extraction: {e}" + Style.RESET_ALL)
        target_wpb = target_wpb if 'target_wpb' in locals() and isinstance(target_wpb, (int, float)) else 2.6
        melisma_bias = melisma_bias if 'melisma_bias' in locals() and isinstance(melisma_bias, (int, float)) else 0.35
        min_word_beats = min_word_beats if 'min_word_beats' in locals() and isinstance(min_word_beats, (int, float)) else 1.0
        allow_nonsense = allow_nonsense if 'allow_nonsense' in locals() else False
        print(f"[DEBUG] Using CRITICAL FALLBACK lyrics parameters: target_wpb={target_wpb:.2f}, melisma_bias={melisma_bias:.2f}, min_word_beats={min_word_beats:.2f}")

    # Check API availability - let the retry logic handle failures
    if genai_local is None or not API_KEYS:
        print(Fore.RED + "ERROR: API unavailable - this should be handled by retry logic!" + Style.RESET_ALL)
        return {
            "words": [],
            "syllables": [],
            "arranger_note": "API unavailable"
        }

    # Use session override if available, otherwise config (no flash fallback)
    global SESSION_MODEL_OVERRIDE
    model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
    try:
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
    except Exception:
        pass
    if not model_name:
        model_name = config.get("model_name", "gemini-2.5-pro")
    # Per-part temperature: default ultra-low; slightly higher for backing/scat/vowels/bridge, overridable via config
    try:
        role_for_temp = _normalize_section_role(section_label)
    except Exception:
        role_for_temp = "verse"
    derived_temp = 0.0
    if role_for_temp in ("backing", "scat", "vowels"):
        derived_temp = 0.1
    elif role_for_temp == "bridge":
        derived_temp = 0.05
    _cfg_temp = config.get("lyrics_temperature") if isinstance(config.get("lyrics_temperature"), (int, float)) else None
    if _cfg_temp is None and isinstance(config.get("temperature"), (int, float)):
        _cfg_temp = config.get("temperature")
    selected_temp = float(_cfg_temp) if isinstance(_cfg_temp, (int, float)) else float(derived_temp)
    generation_config = {"response_mime_type": "application/json", "temperature": selected_temp}
    # Do not set response_schema here to keep compatibility with current SDK
    try:
        if isinstance(config.get("max_output_tokens"), int):
            generation_config["max_output_tokens"] = int(config.get("max_output_tokens"))
    except Exception:
        pass
    model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    section_role = _normalize_section_role(section_label)
    # Prefer planned role from section_description if available (e.g., "[Plan] role=prechorus")
    try:
        if isinstance(section_description, str):
            m = re.search(r"\[\s*Plan\s*\].*?role\s*=\s*([a-zA-Z_]+)", section_description, re.IGNORECASE)
            if m:
                plan_role = _normalize_section_role(m.group(1))
                if plan_role:
                    section_role = plan_role
    except Exception:
        pass
    # Log effective preferences after auto-derivation
    try:
        print(Fore.CYAN + Style.BRIGHT + "[Lyrics:Prefs] " + Style.NORMAL + Fore.WHITE + f"wpb={target_wpb}, melisma_bias={melisma_bias}, min_word_beats={min_word_beats}, nonsense={'on' if allow_nonsense else 'off'}" + Style.RESET_ALL)
    except Exception:
        pass
    # Additional micro-section guidance
    micro_guidance = ""
    try:
        if isinstance(N, int) and N <= 2:
            if N == 1:
                micro_guidance = (
                    "Tiny section: Exactly ONE content word. melisma_spans must be [1].\n"
                    "Prefer a word with an open vowel nucleus (a, ah, o, ai). Avoid closed vowels/clusters.\n"
                    "Do not use [br] or '+' as the only token. If description contains a hook=..., choose a key open-vowel word from it.\n"
                )
            else:
                micro_guidance = (
                    "Tiny section: Keep it extremely concise (1-2 words).\n"
                    "Avoid filler and ensure spans sum to N exactly; favor open vowels on the stronger note.\n"
                    "If absolutely necessary, you may use a single '+' as a continuation indicator (will map to '-'); prefer melisma_spans instead.\n"
                )
    except Exception:
        micro_guidance = ""

    # Compute sustain candidates (long or stressed notes)
    try:
        sustain_candidates = []
        for idx, p in enumerate(full_notes or []):
            d = float(p.get("dur", 0.0))
            st = int(p.get("stress", 0))
            if d >= 1.0 or (st == 1 and d >= 0.75):
                sustain_candidates.append(idx)
    except Exception:
        sustain_candidates = []
    # Prompt gating flags and conditional blocks (role/context-driven)
    try:
        is_chorus = (section_role == 'chorus')
        is_prechorus = (section_role == 'prechorus')
        is_verse = (section_role == 'verse')
        is_backing = (section_role == 'backing')
        is_scat = (section_role == 'scat')
        is_bridge = (section_role == 'bridge')
        is_outro = (section_role == 'outro')
        has_news_vibe = bool(re.search(r"\b(news|report)\b", user_prompt, re.I)) if isinstance(user_prompt, str) else False

        style_block = (
            "STYLE LAYER:\n"
            "- Choose 2–3 mood axes for this part (e.g., luminous / metallic / organic) and keep imagery/metaphor families consistent.\n"
            "- Maintain a focused vocabulary palette per part to create identity.\n\n"
        ) if (is_verse or is_prechorus or is_bridge) else ""

        news_block = (
            "NEWS-REPORT AESTHETIC (optional):\n"
            "- If a 'news/report' vibe is desired for VERSE parts, structure like a bulletin: Headline (1 line), two surreal 'segments', closing sting. Use reporter verbs sparingly (reports/claims/breaking).\n\n"
        ) if (is_verse and has_news_vibe) else ""

        hook_signature_block = (
            "HOOK SIGNATURE:\n"
            "- For chorus/hook, craft a 3–5 word mantra/title-drop and repeat it 2–3× with micro-variation across lines.\n\n"
        ) if is_chorus else ""

        hook_placement_block = (
            "HOOK PLACEMENT (chorus only, if hook exists):\n"
            "- Use the exact wording of the title-drop (no paraphrase) and place it on strong downbeats. Repeat it 2–3 times per chorus. Map the full hook as 1–2 contiguous lines across contiguous onsets; never spell words letter-by-letter.\n\n"
        ) if is_chorus else ""

        pop_repetition_block = (
            "POP REPETITION POLICY (tightened):\n"
            "- CHORUS LINES: Keep 2–4 short lines (5–8 words). Line 1 = exact title-drop; reuse the same wording across all choruses.\n"
            "- VOCABULARY FREEZE: After Chorus 1, do not introduce new vocabulary in later choruses; keep lines verbatim (allow at most a single global tag word for the entire song).\n"
            "- PRE-CHORUS: Reuse 1–2 priming lines verbatim before each chorus; minor tag changes only at the final occurrence.\n"
            "- MICRO-VARIATION BUDGET: Across the whole song, ≤1 token change in chorus lines (not per chorus); keep ≥90% n-gram overlap.\n\n"
        ) if is_chorus else ""

        mapping_repetition_block = (
            "MAPPING FOR REPETITION (hard for chorus):\n"
            "- Map each chorus line onto contiguous onsets without micro-rest interruptions; if a micro-gap ≤ 1/16 occurs, treat as legato (extend previous note).\n"
            "- Do not spell chorus words across >2 syllables; avoid '-' on new onsets in the chorus.\n\n"
        ) if is_chorus else ""

        line_integrity_block = (
            "LINE INTEGRITY (chorus):\n"
            "- Avoid introducing new nouns/verbs in the chorus after the first occurrence; prefer reusing the established chorus lines exactly.\n\n"
        ) if is_chorus else ""

        chant_variance_block = (
            "CHANT VARIANCE & REPETITION CONTROL (outside chorus):\n"
            "- Confine chants to short, marked segments (≤ 2 bars). In each chant loop, vary the last word (rhyme/assonance) to avoid monotony.\n"
            "- Outside chorus/backing/scat chant segments, avoid repeating the same 1–2 token word more than twice in a row; prefer semantic substitution or proceed to the next idea.\n\n"
        ) if (is_backing or is_scat) else ""

        diction_block = (
            "DICTION:\n"
            "- Prefer concrete nouns/verbs over abstractions; avoid tech/FX jargon.\n\n"
        ) if is_verse else ""

        dramaturgy_block = (
            "DRAMATURGY:\n"
            "- Use [br] as a lift-in before chorus; pre-chorus builds energy (imperatives/future forms).\n\n"
        ) if is_prechorus else ""

        prechorus_hook_block = (
            "PRE-CHORUS HOOK DISCIPLINE (prechorus only):\n"
            "- Do NOT use any exact token from hook_canonical here (case-insensitive). Hint only with metaphor/synonyms; strictly avoid title-drop.\n"
            "- If any hook token appears in draft, replace it before returning.\n\n"
        ) if is_prechorus else ""

        # Hook presence signals for additional gating
        has_quoted_hook = False
        try:
            if isinstance(user_prompt, str) and re.search(r'"[^"\n]{2,}"', user_prompt):
                has_quoted_hook = True
        except Exception:
            has_quoted_hook = False
        has_plan_hook = False
        try:
            sd = section_description if isinstance(section_description, str) else ""
            if ("hook_canonical" in sd) or re.search(r"hook\s*=", sd):
                has_plan_hook = True
        except Exception:
            has_plan_hook = False

        # Additional gated blocks
        hook_canonical_detection_block = (
            "HOOK CANONICAL DETECTION (hard):\n"
            "- If the user intent contains a phrase in double quotes, set hook_canonical to that exact phrase (no paraphrase). Use it only in CHORUS sections.\n"
            "- If no quoted phrase exists, do NOT invent a hook; allow CHORUS to operate as a peak section without a title-drop (use motif/backing emphasis instead).\n\n"
        ) if (is_chorus and has_quoted_hook) else ""

        hook_contiguity_repair_block = (
            "HOOK CONTIGUITY REPAIR (hard):\n"
            "- Inside hook_token_ranges: no rests, no '-' placeholders; sustain vowels via melisma instead.\n"
            "- If rests/gaps would break the hook, re-map spans so all hook words lie on contiguous onsets (prefer longer melisma_spans over inserting rests).\n"
            "- If a hook word would be split across non-adjacent onsets, choose a different syllabification or re-map spans to maintain adjacency.\n\n"
        ) if (is_chorus and (has_quoted_hook or has_plan_hook)) else ""

        content_narrative_block = (
            "CONTENT / NARRATIVE:\n"
            "- Aim over each ~2 bars to include at least one concrete image (noun) and one action verb; avoid abstract filler.\n\n"
        ) if (is_verse or is_bridge) else ""

        writing_policy_hook_line = (
            "- If a specific hook phrase was requested, you may use that exact phrase in the chorus only.\n"
        ) if (has_quoted_hook or has_plan_hook) else ""

        sustain_hint_block = (
            "PERFORMANCE HINTS (optional):\n"
            + ("- Vowel sustain candidates (note indices): " + json.dumps(sustain_candidates) + "\n\n" if sustain_candidates else "")
        ) if sustain_candidates else ""

        active_blocks = []
        if style_block: active_blocks.append("STYLE")
        if news_block: active_blocks.append("NEWS")
        if hook_signature_block: active_blocks.append("HOOK_SIGNATURE")
        if hook_placement_block: active_blocks.append("HOOK_PLACEMENT")
        if pop_repetition_block: active_blocks.append("POP_REPETITION")
        if mapping_repetition_block: active_blocks.append("MAPPING_REPETITION")
        if line_integrity_block: active_blocks.append("LINE_INTEGRITY")
        if chant_variance_block: active_blocks.append("CHANT_VARIANCE")
        if diction_block: active_blocks.append("DICTION")
        if dramaturgy_block: active_blocks.append("DRAMATURGY")
        if prechorus_hook_block: active_blocks.append("PRECHORUS_HOOK")
        if hook_canonical_detection_block: active_blocks.append("HOOK_CANONICAL")
        if hook_contiguity_repair_block: active_blocks.append("HOOK_CONTIGUITY")
        if content_narrative_block: active_blocks.append("CONTENT_NARRATIVE")
        if writing_policy_hook_line: active_blocks.append("WRITING_POLICY_HOOK")
        if sustain_hint_block: active_blocks.append("SUSTAIN_HINTS")
        try:
            if active_blocks:
                print(Fore.MAGENTA + Style.BRIGHT + "[Lyrics:Gating] " + Style.NORMAL + Fore.WHITE + ", ".join(active_blocks) + Style.RESET_ALL)
        except Exception:
            pass
    except Exception:
        # On any error in gating logic, fall back silently to no-op blocks
        style_block = news_block = hook_signature_block = hook_placement_block = pop_repetition_block = ""
        mapping_repetition_block = line_integrity_block = chant_variance_block = diction_block = dramaturgy_block = prechorus_hook_block = ""
        hook_canonical_detection_block = hook_contiguity_repair_block = content_narrative_block = writing_policy_hook_line = ""
        sustain_hint_block = ""

    # Compute density/melisma guidance block for the prompt
    try:
        tw_val = float(target_wpb or 2.0)
    except Exception:
        tw_val = 2.0
    try:
        mb_val = float(melisma_bias or 0.5)
    except Exception:
        mb_val = 0.5
    try:
        eb_val = float(est_bars or 0.0)
    except Exception:
        eb_val = 0.0
    try:
        bpb_val = int(ts.get('beats_per_bar', 4)) if isinstance(ts, dict) else 4
    except Exception:
        bpb_val = 4
    bar_slots_str = ""
    try:
        if isinstance(eb_val, float) and eb_val > 0 and int(N) <= 256:
            slots = []
            for b in range(int(eb_val)):
                cnt = 0
                for p in full_notes:
                    try:
                        if int(p['start'] // max(1, bpb_val)) == b:
                            cnt += 1
                    except Exception:
                        continue
                slots.append(cnt)
            bar_slots_str = f"- bar_slots (approx): {json.dumps(slots)}\n"
    except Exception:
        bar_slots_str = ""
    one_span_band = "[0.15..0.40]" if mb_val < 0.6 else "[0.35..0.70]"
    try:
        density_block = (
            "PLACEMENT PREFS (derived):\n"
            f"- target_wpb≈{tw_val:.2f}; melisma_bias≈{mb_val:.2f}; min_word_beats≈{float(min_word_beats if 'min_word_beats' in locals() else 0.6):.2f}\n"
            + (bar_slots_str if bar_slots_str else "")
            + "\n"
        )
    except Exception:
        density_block = (bar_slots_str or "")

    prompt_parts: List[str] = []
    prompt_parts.append("HEADER:\n")
    prompt_parts.append(f"- Role=lyricist · Target=SynthV-ready · SectionRole={section_role}\n")
    prompt_parts.append((f"- Notes={N} · Bars≈{est_bars:.2f}\n" if isinstance(est_bars, float) else f"- Notes={N}\n"))
    prompt_parts.append(f"- Hook={'yes' if (has_quoted_hook or has_plan_hook) else 'no'} · Nonsense={'on' if allow_nonsense else 'off'}\n\n")
    prompt_parts.append("ACTIVE POLICIES: " + ", ".join(active_blocks) + "\n\n")
    prompt_parts.append("ROLE:\nYou are a professional lyricist and vocal arranger. Target: SynthV-ready output.\n\n")
    # Soft guidance for micro-notes / text shaping in word-first mode
    prompt_parts.append("TEXT SHAPING (soft):\n- Prefer whole words; split only when rhythm truly requires it; avoid frequent splits outside the hook.\n- Favor fewer tokens when melodic texture is dense; use '-' sustains instead of adding syllables on very short notes.\n- Where possible, avoid splitting words like 'echo', 'pattern', 'remember', 'light', 'glass', 'shape'.\n- Keep hook words unbroken where possible; only split if musically compelling.\n\n")
    prompt_parts.append("CONTEXT:\n")
    prompt_parts.append(f"- Genre={genre}; Language={language}; Key/Scale={key_scale}; BPM={round(float(bpm))}; TimeSig={ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}\n")
    prompt_parts.append(f"- Track={track_name}; Section={section_label or ''}; Description={section_description or ''}\n")
    if vocab_ctx.get('context_instruments'):
        prompt_parts.append("- Instruments (name,role): " + json.dumps([{k: v for k, v in it.items() if k in ('name','role')} for it in (vocab_ctx.get('context_instruments') or [])]) + "\n")
    if any(vocab_ctx.get('style_keywords')):
        prompt_parts.append("- Style hints: " + ", ".join([x for x in vocab_ctx.get('style_keywords') if x]) + "\n")
    # Remove numeric preferences from prompt; roles guide content only
    if isinstance(est_bars, float):
        prompt_parts.append(f"- Estimated bars: ≈{est_bars:.2f}\n")
    if isinstance(user_prompt, str) and user_prompt.strip():
        prompt_parts.append(f"- User intent (do NOT copy phrases): {user_prompt}\n")
        prompt_parts.append("- If user intent implies a hook/title phrase (quoted or not), set hook_canonical accordingly and reserve exact usage for CHORUS/DROP.\n")
    if isinstance(history_context, str) and history_context.strip():
        prompt_parts.append("- Previous sections (label: text):\n" + history_context + "\n")
    
    if section_role == 'chorus':
        prompt_parts.append("- Section function: CHORUS/HOOK → memorable, repeatable hook; refrain lines may recur.\n")
    elif section_role == 'verse':
        prompt_parts.append("- Section function: VERSE → progress narrative; set up the chorus without reusing lines.\n")
    elif section_role == 'prechorus':
        prompt_parts.append("- Section function: PRE-CHORUS/BUILD → tighten phrasing; lead into chorus.\n")
    elif section_role == 'bridge':
        prompt_parts.append("- Section function: BRIDGE → contrast; avoid chorus wording; prepare lift/dissolve.\n")
    if micro_guidance:
        prompt_parts.append(micro_guidance)
    prompt_parts.append("\nROLE GUIDE (for this part):\n")
    if section_role == 'chorus':
        prompt_parts.append("- chorus: Deliver the title-drop clearly; keep a few short lines sized to the part; reuse almost verbatim; minimize extra tag words.\n")
        prompt_parts.append("- chorus: Keep hook words unbroken where possible; prefer whole words over syllable splits.\n")
    elif section_role == 'prechorus':
        prompt_parts.append("- prechorus: Tighten phrasing, raise tension; hint the title-drop without using it; end with a lift.\n")
    elif section_role == 'verse':
        prompt_parts.append("- verse: Advance the story with concrete imagery and active verbs; avoid chorus wording; set up tension.\n")
    elif section_role == 'bridge':
        prompt_parts.append("- bridge: Contrast the chorus with a fresh angle or twist; avoid chorus wording.\n")
    elif section_role == 'backing':
        prompt_parts.append("- backing: Echo/answer the lead with short repeatable fragments; avoid new content.\n")
    elif section_role == 'scat':
        prompt_parts.append("- scat: Musical syllables supporting groove; vary vowels; avoid monotony.\n")
    else:
        prompt_parts.append("- vowels: Hold open vowels on long notes; no semantics; pure sustain.\n")
    prompt_parts.append("\nPLAN FOR THIS PART:\n")
    # (optional plan hint intentionally disabled)
    prompt_parts.append("\nMELODY NOTES (full, order: {i,start,dur,pitch,stress):\n".replace("(", "(").replace(")", ")") + json.dumps(full_notes) + "\n\n")
    if vocab_ctx.get('context_instruments'):
        prompt_parts.append("OTHER TRACKS (full notes for this part):\n" + json.dumps(vocab_ctx.get('context_instruments')) + "\n\n")
    if any(isinstance(it, dict) and it.get('lyrics_history') for it in (vocab_ctx.get('context_instruments') or [])):
        prompt_parts.append("OTHER VOCAL LYRICS HISTORY (prior parts):\n" + json.dumps([
            {"name": it.get('name'), "role": it.get('role'), "lyrics_history": it.get('lyrics_history')}
            for it in (vocab_ctx.get('context_instruments') or [])
            if (isinstance(it, dict) and it.get('lyrics_history'))
        ]) + "\n\n")
    prompt_parts.append("WRITING POLICY (global):\n- Avoid cliches; prefer concrete imagery; keep hook-worthy phrasing.\n- Avoid production/effect jargon; prefer imagery and narrative.\n- Do not copy user wording; treat artists/titles as style cues only.\n")
    prompt_parts.append(writing_policy_hook_line)
    prompt_parts.append(("- Nonsense syllables may be used sparingly for musical effect (avoid repetitive 'la/na/da').\n" if allow_nonsense else "- Do NOT use placeholder syllables like 'la', 'na', 'da' as words.\n"))
    prompt_parts.append("\n")
    prompt_parts.append(density_block)
    prompt_parts.append(hook_canonical_detection_block)
    prompt_parts.append(content_narrative_block)
    prompt_parts.append(style_block)
    prompt_parts.append(hook_signature_block)
    prompt_parts.append("[soft|global] PHRASE ARCHITECTURE:\n- Meter-aware phrasing: group phrases according to the time signature (e.g., 4+4 in 4/4; 3+3 in 3/4; 2+2+3 in 7/8). Cadence end-lines on downbeats; use micro-pauses only before transitions.\n\n")
    prompt_parts.append(news_block)
    prompt_parts.append(hook_placement_block)
    prompt_parts.append(prechorus_hook_block)
    prompt_parts.append(hook_contiguity_repair_block)
    prompt_parts.append(sustain_hint_block)
    prompt_parts.append(pop_repetition_block)
    prompt_parts.append(mapping_repetition_block)
    prompt_parts.append(chant_variance_block)
    if section_role == 'outro':
        prompt_parts.append("OUTRO REPETITION POLICY (outro only):\n- Avoid overlong 1–2-word loops; for fade-outs prefer a few concise fragments with line_breaks and micro-variation instead of long repetition chains.\n- No hook-fragment spam; sparse callbacks are sufficient.\n\n")
    prompt_parts.append(line_integrity_block)
    prompt_parts.append("[soft|global] PROSODY CHECKLIST:\n- Place strong verbs on ascending motion; use falling imagery on descending lines.\n- Use internal rhyme/assonance sparingly (≈1 per part) to enhance memorability.\n\n")
    prompt_parts.append("SMOOTHNESS / LEGATO (global):\n- Avoid choppy delivery. Map key content syllables to longer or stressed notes; keep very short notes for function words or continuations.\n- Prefer open vowels (a/ah/o/ai) on sustained notes; avoid long held consonants; move codas to the next onset if needed.\n- Group lines into phrase windows and keep within-phrase legato (minimize micro-gaps).\n\n")
    prompt_parts.append("NONSENSE BUDGET (only if allowed):\n- If nonsense syllables are used, apply them sparingly (small proportion) and vary them; avoid back-to-back nonsense lines.\n\n")
    prompt_parts.append(diction_block)
    prompt_parts.append("CAPITALIZATION POLICY:\n- Use Title Case only for the exact hook_canonical string if one exists; otherwise use normal sentence case. Avoid arbitrary ALL CAPS.\n\n")
    prompt_parts.append("VOCAL COLOR:\n- On high notes, keep closed vowels (i/e) short; sustain open vowels (a/ah/o/ai) on long/stressed notes.\n\n")
    prompt_parts.append(dramaturgy_block)
    prompt_parts.append("ADAPTIVE MODE (soft):\n- Always return aligned words/spans for this part. If mapping feels forced, include a short 'note_adaptation_vision' and a 'placement_difficulty' in [0..1] describing the challenge.\n- The vision should suggest phrasing windows, downbeat targets for the title-drop, and whether melisma/merges would help; avoid prescriptive micro-edits.\n\n")
    # DECISION rubric for model-controlled silence
    prompt_parts.append("DECISION: intentional_silence (hard):\n")
    prompt_parts.append("- Set intentional_silence=true (and return words=[], melisma_spans=[]) if any applies:\n")
    prompt_parts.append("  • Plan/Hints indicate 'instrumental only' / 'maintain vocal silence' / 'no vocals'.\n")
    prompt_parts.append("  • The note grid provides no clear musical window (majority of onsets are micro-slots < ≈1/4 beat for potential content).\n")
    prompt_parts.append("  • Role is breaths/vocal_fx but there is no plausible small window for 1–2 short onsets without crowding.\n")
    prompt_parts.append("- Otherwise set intentional_silence=false and provide words + melisma_spans.\n\n")
    # Role-specific hard guidelines for token types
    if section_role == 'breaths':
        prompt_parts.append("ROLE=breaths (hard):\n- Output tokens MUST be '[br]' only (1–2 events typical); no lexical words or vowel placeholders.\n- If no plausible short window exists, set intentional_silence=true.\n\n")
    if section_role in ('whisper','spoken'):
        prompt_parts.append("ROLE=whisper/spoken (hard):\n- Use lexical words/phrases. Do NOT use pure vowel placeholders like 'Ah', 'Ooh', 'Mmm' as tokens.\n- Keep phrasing minimal and intimate; avoid dense onsets.\n\n")
    prompt_parts.append("SYNTHV RULES (hard):\n")
    prompt_parts.append("- '+' usage: Allowed only as a continuation indicator when absolutely necessary (it will be mapped to '-' in export). Prefer melisma via spans instead. No in-word hyphenation. One token = one word.\n")
    prompt_parts.append("- Phrase starts: After a rest, the first note MUST be a content syllable (never '-').\n")
    prompt_parts.append("- Forbidden tokens at word starts: single consonants, punctuation-only tokens. Use a content word or, if necessary, map that onset as a continuation '-' with melisma (do not invent vowels).\n")
    prompt_parts.append("- **VOCAL EXCELLENCE FOR SYNTHV**:\n")
    prompt_parts.append("  • Create singable, meaningful words that flow naturally when performed\n")
    prompt_parts.append("  • Choose words that SynthV can articulate beautifully and clearly\n")
    prompt_parts.append("  • Select words that carry emotional weight and serve the musical moment\n")
    prompt_parts.append("  • Prefer words with open vowels (ah, oh, oo, ee, ay, ai) for sustained notes\n")
    prompt_parts.append("  • Use complete words rather than fragments or single letters\n")
    prompt_parts.append("  • Extend words with melisma ('-') when it serves the musical expression\n")
    prompt_parts.append("  • Aim for professional quality that would work in a commercial release\n")
    prompt_parts.append("  • Focus on basic UST parameters: NoteNum, Length, Lyric, and Velocity\n")
    prompt_parts.append("  • Create lyrics that work well with UTAU's basic phoneme system\n")
    # Breath-specific guidance only when role demands it
    if section_role in ('breaths', 'prechorus'):
        prompt_parts.append("- Breath policy: If the role or plan hints request a breath, output the token '[br]' on a dedicated short rest-like note; do NOT replace it with words.\n")
        prompt_parts.append("- Micro-onset policy (with breaths): Onsets shorter than ~1/4 beat should prefer '-' continuation or '[br]' (if specifically intended) unless a naturally short syllable fits.\n")
    else:
        prompt_parts.append("- Micro-onset policy: Onsets shorter than ~1/4 beat should prefer '-' continuation unless a naturally short syllable fits.\n")
    prompt_parts.append("- Multi-syllable words: With ≥2 onsets, split into syllables; do not put '-' on a fresh onset.\n")
    prompt_parts.append("- Monosyllables on many onsets: Prefer a 2-syllable synonym; else at most one '-' continuation; avoid '- -'.\n")
    prompt_parts.append("- Placement: New syllables on note onsets; sustain open vowels (a/ah/o/ai) on long stressed notes; do not hold consonant codas.\n")
    if section_role in ('breaths',):
        prompt_parts.append("- '[br]' usage: Only on dedicated short rest notes; never inside words; never as the only token in 1-note segments (use a short content impulse instead, if needed).\n")
    prompt_parts.append("- Optional phoneme_hints per note (e.g., [k a t]) for ambiguous words (separate list).\n\n")
    prompt_parts.append("LYRICS RULES:\n- **TARGET DURATION**: Aim 0.75–3.0 beats; allow few 0.5–0.75 pickups (<15%).\n- **PHRASE-BASED COMPOSITION**: Group words into complete phrases of 3–6 words.\n- **NO MICRO-NOTES**: Avoid notes shorter than 0.5 beats entirely; keep 0.5–0.75 rare.\n- **CONTINUOUS FLOW**: Connect notes end-to-start. Minimize gaps between notes.\n- **MUSICAL COHERENCE**: Each phrase must be a complete musical thought.\n- **MAXIMUM MELISMA**: Never exceed 20% of total tokens as melisma.\n\n")
    prompt_parts.append("EXAMPLES:\n")
    prompt_parts.append("- monosyllable over 3 onsets → words=['shine'], spans=[3] ⇒ tokens=['shine','-','-'].\n")
    prompt_parts.append("- 2 words over 3 onsets → words=['o-ver','load'], spans=[1,2] ⇒ tokens=['o-ver','load','-'].\n\n")
    prompt_parts.append("OUTPUT FORMAT (STRICT JSON):\n")
    prompt_parts.append("{\n  \"words\": [string, ...],\n  \"melisma_spans\": [int, ...]\n}\n\n")
    prompt_parts.append(f"CONSTRAINTS:\n- Sum(melisma_spans) == {N}.\n- len(words) == len(melisma_spans).\n- Use '-' only to continue a previous word (melisma).\n\n")
    # Checklist with role gating
    prompt_parts.append("CHECKLIST:\n- No '+'; no in-word '-' hyphens\n- No leading '-' after a rest; no consecutive '-' chain\n")
    if section_role not in ('breaths',):
        prompt_parts.append("- No breath in 1-note parts; [br] only on rest notes\n")
    prompt_parts.append("- Spans sum/lengths match; JSON only\n\n")
    prompt_parts.append("SELF-CHECK & AUTO-REPAIR (hard):\n")
    prompt_parts.append("- Compute the following metrics before returning: words_per_bar, one_span_ratio (#spans==1 / #words).\n")
    prompt_parts.append("- If melisma_bias≥0.4 AND one_span_ratio > 1.0, increase melisma by holding vowels ('-') on weak beats for suitable words until one_span_ratio ∈ [0.35..1.0].\n")
    prompt_parts.append("- If melisma_bias<0.4 AND one_span_ratio < 0.15, reduce holds slightly by introducing new syllables/short content words on eligible onsets until ∈ [0.15..0.40].\n")
    prompt_parts.append("- If words_per_bar < target_wpb-0.20, reduce melisma moderately and choose shorter content words where possible while respecting min_word_beats.\n")
    prompt_parts.append("- If words_per_bar > target_wpb+0.20, increase melisma moderately or choose fewer/longer words while respecting min_word_beats.\n")
    prompt_parts.append("- Also enforce existing hook/section policies as already specified. Return ONLY the final JSON.\n\n")
    # JSON-only enforced via response_mime_type/response_schema
    prompt = "".join(prompt_parts)

    def _call_with_rotation(prompt_text: str) -> dict | None:
        max_attempts = max(3, len(API_KEYS))
        attempts = 0
        nonlocal_model = [model]
        while attempts < max_attempts:
            try:
                resp = nonlocal_model[0].generate_content(prompt_text, safety_settings=safety_settings, generation_config=generation_config)
                raw = getattr(resp, "text", "") or ""
                cleaned = raw.strip().replace("```json", "").replace("```", "")
                obj = json.loads(cleaned)
                # Minimal hard validation: only retry on fatal/empty
                if isinstance(obj, dict):
                    w = obj.get('words'); s = obj.get('melisma_spans')
                    if (obj.get('intentional_silence') is True) or (isinstance(w, list) and isinstance(s, list) and len(w) >= 1 and len(s) >= 1):
                        return obj
                # else -> fatal/empty
                raise ValueError("fatal or empty stage-1 output")
            except Exception as e:
                err = str(e).lower()
                # Treat various quota/rate signals as rotation triggers (do NOT consume attempts)
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    # Cool down the failing key so we don't bounce back immediately
                    try:
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        try:
                            avail = []
                            now = time.time()
                            for ix, _ in enumerate(API_KEYS):
                                tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                                avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                            print(Fore.MAGENTA + Style.BRIGHT + "[Lyrics:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # rotate keys if possible
                    n = len(API_KEYS)
                    rotated = False
                    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off*stride) % n
                        if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            try:
                                print(Fore.CYAN + f"[Lyrics] Switching to API key #{idx+1}..." + Style.RESET_ALL)
                            except Exception:
                                pass
                            nonlocal_model[0] = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                            rotated = True
                            break
                        except Exception:
                            continue
                    if not rotated:
                        # backoff before retrying the same key set
                        wait_s = 3
                        if qtype in ('per-hour', 'per-day'):
                            # longer sleeps to avoid hammering; if per-day, back off minutes instead of seconds
                            wait_s = 60 if qtype == 'per-hour' else 300
                        _interruptible_backoff(wait_s, config, context_label="Lyrics words cooldown")
                        continue
                    # after rotation, immediately retry without consuming attempts
                    continue
                # Non-quota/transient error: log once and retry quickly (consumes attempt)
                try:
                    print(Fore.YELLOW + f"[Lyrics] Model error (transient): {str(e)[:160]}" + Style.RESET_ALL)
                except Exception:
                    pass
                attempts += 1
                time.sleep(1)
                continue
        return None

    def _expand_tokens(words: List[str], spans: List[int]) -> List[str] | None:
        if not (isinstance(words, list) and isinstance(spans, list)): return None
        if len(words) != len(spans): return None
        if not all(isinstance(x, (int, float)) and x >= 1 for x in spans): return None
        if sum(int(x) for x in spans) != N: return None
        tokens: List[str] = []
        for w, span in zip(words, spans):
            wstr = str(w).strip()
            span_i = int(span)
            if not wstr:
                return None
            tokens.append(wstr)
            for _ in range(max(0, span_i-1)):
                tokens.append('-')
        return tokens if len(tokens) == N else None

    def _validate_and_feedback(words: List[str], spans: List[int]) -> tuple[bool, dict]:
        issues = {}
        bad = {"la","na","da","ta","ba","pa","ma"}
        if words:
            ph = sum(1 for w in words if str(w).strip().lower() in bad) / max(1,len(words))
            if ph > 0.5:
                issues['too_many_placeholders'] = round(ph,2)
        # Inline '+' misuse check (should not appear inside words)
        try:
            if words and any('+' in str(w) for w in words):
                issues['inline_plus_forbidden'] = True
        except Exception:
            pass
        # Inline '-' hyphenation misuse check
        try:
            if words and any(('-' in str(w)) and (str(w).strip() != '-') for w in words):
                issues['inline_hyphen_forbidden'] = True
        except Exception:
            pass
        # For single-note parts, forbid [br] or breath tokens as the only word
        try:
            if isinstance(N, int) and N == 1 and words and len(words) == 1:
                w0 = str(words[0]).strip().lower()
                if w0 in ('[br]', 'br', 'breath'):
                    issues['no_breath_in_single_note'] = True
        except Exception:
            pass
        # Anti-echo: penalize excessive overlap with user guidance
        try:
            if isinstance(user_prompt, str) and user_prompt.strip() and words:
                up = re.sub(r"[^a-zA-Z0-9\s]", " ", user_prompt.lower())
                up_tokens = {t for t in up.split() if len(t) >= 4}
                gen_tokens = {str(w).strip().lower() for w in words if isinstance(w, str)}
                overlap = len(up_tokens & gen_tokens) / max(1, len(up_tokens))
                if overlap > 0.3:
                    issues['too_much_prompt_overlap'] = round(overlap,2)
        except Exception:
            pass
        # Loosen validator for tiny parts (<=2 notes): don't flag density/melisma issues
        tiny_part = (isinstance(N, int) and N <= 2)
        if isinstance(est_bars, float) and est_bars > 0 and words and not tiny_part:
            wpb = len(words)/est_bars
            if isinstance(target_wpb, float):
                if wpb > target_wpb*1.6:
                    issues['too_many_words_per_bar'] = round(wpb,2)
                if wpb < target_wpb*0.4:
                    issues['too_few_words_per_bar'] = round(wpb,2)
        if isinstance(melisma_bias, float) and spans and not tiny_part:
            one_ratio = sum(1 for s in spans if int(s)==1)/max(1,len(spans))
            if melisma_bias >= 0.4 and one_ratio > 1.0:
                issues['increase_melisma'] = round(one_ratio,2)
        return (len(issues)==0), issues

    # Retry loop with validation feedback (no fixed attempt count for content; capped to avoid stall)
    best_tokens: List[str] | None = None
    # Repairs disabled: single-shot generation
    obj = _call_with_rotation(prompt)
    if obj is None:
        return ["la" for _ in range(N)]
    words = obj.get("words") if isinstance(obj, dict) else None
    spans = obj.get("melisma_spans") if isinstance(obj, dict) else None
    phoneme_hints = obj.get("phoneme_hints") if isinstance(obj, dict) else None
    # Optional adaptive/meta fields
    try:
        part_key = f"{track_name}|{section_label or ''}"
        meta_entry = {
            "placement_difficulty": (float(obj.get("placement_difficulty")) if isinstance(obj.get("placement_difficulty"), (int, float)) else None),
            "note_adaptation_vision": (str(obj.get("note_adaptation_vision")).strip() if isinstance(obj.get("note_adaptation_vision"), str) else None),
            "hook_canonical": (str(obj.get("hook_canonical")).strip() if isinstance(obj.get("hook_canonical"), str) else None),
            "hook_token_ranges": (obj.get("hook_token_ranges") if isinstance(obj.get("hook_token_ranges"), list) else None),
            "line_breaks": (obj.get("line_breaks") if isinstance(obj.get("line_breaks"), list) else None),
            "chorus_lines": (obj.get("chorus_lines") if isinstance(obj.get("chorus_lines"), list) else None),
            "chant_segments": (obj.get("chant_segments") if isinstance(obj.get("chant_segments"), list) else None),
            "phrase_windows": (obj.get("phrase_windows") if isinstance(obj.get("phrase_windows"), list) else None),
            "vowel_sustain_targets": (obj.get("vowel_sustain_targets") if isinstance(obj.get("vowel_sustain_targets"), list) else None),
            "note_rewrite_request": (bool(obj.get("note_rewrite_request")) if obj.get("note_rewrite_request") is not None else None),
            "note_rewrite_intent": (str(obj.get("note_rewrite_intent")).strip() if isinstance(obj.get("note_rewrite_intent"), str) else None),
            "mapping_feasibility_score": (float(obj.get("mapping_feasibility_score")) if isinstance(obj.get("mapping_feasibility_score"), (int,float)) else None),
            "repetition_report": (obj.get("repetition_report") if isinstance(obj.get("repetition_report"), dict) else None),
            "self_check": (obj.get("self_check") if isinstance(obj.get("self_check"), dict) else None)
        }
        # Top-level intentional silence flag (preferred short-circuit)
        try:
            if obj.get("intentional_silence") is True:
                meta_entry["intentional_silence"] = True
        except Exception:
            pass
        LYRICS_PART_META[part_key] = meta_entry
        pd = LYRICS_PART_META[part_key]["placement_difficulty"]
        if pd is not None:
            try:
                print(Style.DIM + f"[Placement] difficulty={pd:.2f} for '{section_label or ''}'" + Style.RESET_ALL)
            except Exception:
                pass
    except Exception:
        pass
    # Enforce strict micro-section behavior for N==1
    if words is not None and spans is not None and isinstance(N, int) and N == 1:
        try:
            if (len(words) == 1) and (len(spans) == 1) and int(spans[0]) == 1 and isinstance(words[0], str) and words[0].strip():
                tokens = [str(words[0]).strip()]
            else:
                tokens = None
        except Exception:
            tokens = None
    else:
        tokens = _expand_tokens(words, spans) if words is not None and spans is not None else None
        if tokens is None:
            return ["la" for _ in range(N)]
    ok, issues = _validate_and_feedback([str(w) for w in words], [int(s) for s in spans])
    try:
        if section_role == 'chorus' and isinstance(words, list):
            if 'too_much_prompt_overlap' in issues and issues['too_much_prompt_overlap'] <= 0.35:
                issues.pop('too_much_prompt_overlap', None)
    except Exception:
        pass
    return tokens if ok else tokens
# --- Lyrics-first STAGE 1: Free text with syllables (no grid constraint) ---
def _plan_lyrical_concept(config: Dict, genre: str, inspiration: str, section_role: str, section_label: str | None = None, user_prompt: str | None = None, history_context: str | None = None, cfg: Dict | None = None) -> Dict:
    """Stage 1a: Plan lyrical concept, theme, and emotional direction for a section."""
    # Input validation
    if not isinstance(config, dict):
        print("[Concept Planning Error] Invalid config: must be dict")
        return {}
    if not isinstance(genre, str) or not genre.strip():
        print("[Concept Planning Error] Invalid genre: must be non-empty string")
        return {}
    if not isinstance(inspiration, str) or not inspiration.strip():
        print("[Concept Planning Error] Invalid inspiration: must be non-empty string")
        return {}
    if not isinstance(section_role, str) or not section_role.strip():
        print("[Concept Planning Error] Invalid section_role: must be non-empty string")
        return {}
    
    try:
        import google.generativeai as genai_local
        # Force reload config to get latest model_name setting
        try:
            fresh_config = load_config(CONFIG_FILE)
            config = fresh_config
        except Exception as e:
            pass
        
        # Resolve model with priority: hotkey (this step) > config model > session override > lyrics_model
        global SESSION_MODEL_OVERRIDE, REQUESTED_SWITCH_MODEL
        model_name = REQUESTED_SWITCH_MODEL or config.get('model_name') or config.get('model') or SESSION_MODEL_OVERRIDE or config.get('lyrics_model', 'gemini-2.5-pro')
        try:
            if not model_name:
                model_name = config.get("model_name", "gemini-2.5-pro")
        except Exception:
            model_name = config.get("model_name", "gemini-2.5-pro")
        try:
            if not model_name:
                model_name = config.get("model_name", "gemini-2.5-pro")
        except Exception:
            model_name = config.get("model_name", "gemini-2.5-pro")
        # Use artifact lyrics_language if available, otherwise fallback to config
        language = str(cfg.get("lyrics_language", "") if cfg else config.get("lyrics_language", "English"))
        key_scale = str(cfg.get("key_scale", "") if cfg else config.get("key_scale", "")).strip()
        
        # Extract key information for tonal context
        try:
            def _infer_root_from_key(ks: str) -> int:
                """Infer MIDI root note from key string like 'C major' or 'A minor'."""
                if not ks:
                    return 60
                ks_lower = ks.lower().strip()
                note_map = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4, 'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9, 'a#': 10, 'bb': 10, 'b': 11}
                for note, semitone in note_map.items():
                    if ks_lower.startswith(note):
                        return 60 + semitone  # C4 = 60
                return 60
            tonic_midi = _infer_root_from_key(key_scale) if key_scale else 60
        except Exception:
            tonic_midi = 60
            
        concept_prompt = f"""You are a lyrical concept specialist. Plan the conceptual foundation for a {genre} track section.

GLOBAL CONTEXT:
- Genre: {genre} | Key/Scale: {key_scale} | Tonic: {tonic_midi}
- Section: {section_label} | Role: {section_role}
- User Inspiration: {inspiration}

CONCEPTUAL PLANNING:
- Define the core emotional theme for this section
- Establish the narrative perspective (1st/2nd/3rd person)
- Identify key imagery and metaphors to use
- Plan the overall mood and atmosphere
- Consider how this section fits into the larger song narrative

ROLE-SPECIFIC CONCEPT:
- {section_role}: Plan appropriate conceptual approach
- Consider the section's function in the song structure
- Ensure concept aligns with the role's purpose

OUTPUT (JSON):
{{
  "core_theme": "string - main emotional/narrative theme",
  "perspective": "string - narrative voice (I/you/we/they)",
  "key_imagery": ["string", ...] - 3-5 key visual/metaphorical elements",
  "mood": "string - overall emotional atmosphere",
  "narrative_function": "string - how this section advances the story",
  "tone": "string - vocal delivery approach (intimate/ethereal/aggressive/etc)"
}}"""
        
        # Add retry and API key rotation logic
        max_attempts = max(3, len(API_KEYS))
        attempts = 0
        quota_rotation_count = 0
        
        while attempts < max_attempts:
            try:
                # Configure model with current API key
                model = genai_local.GenerativeModel(model_name=model_name)
                
                resp = model.generate_content(concept_prompt)
                raw = _extract_text_from_response(resp) or ""
                cleaned = raw.strip().replace("```json", "").replace("```", "")
                payload = _extract_json_object(cleaned)
                concept = json.loads(payload or cleaned)
                
                if isinstance(concept, dict) and concept:
                    return concept
                else:
                    raise ValueError("Empty or invalid concept response")
                    
            except Exception as e:
                err = str(e).lower()
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    try:
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        avail = []
                        now = time.time()
                        for ix, _ in enumerate(API_KEYS):
                            tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                            avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                        print(Fore.MAGENTA + Style.BRIGHT + "[Concept Planning:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                    except Exception:
                        pass
                    
                    # Try to rotate to next available key
                    n = len(API_KEYS)
                    rotated = False
                    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off*stride) % n
                        if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            rotated = True
                            break
                        except Exception:
                            continue
                    
                    if not rotated:
                        wait_s = 3
                        if _all_keys_daily_exhausted():
                            _schedule_hourly_probe_if_needed()
                            wait_s = _seconds_until_hourly_probe()
                        _interruptible_backoff(wait_s, config, context_label="Concept Planning cooldown")
                        quota_rotation_count += 1
                        continue
                    else:
                        continue
                else:
                    # Non-quota error: brief pause then retry and count attempt
                    time.sleep(0.5)
                    attempts += 1
                    continue
        
        print(f"[Concept Planning Error] Failed after {max_attempts} attempts")
        return {}
        
    except Exception as e:
        print(f"[Concept Planning Error] Unexpected error: {e}")
        return {}

def _plan_lyrical_phrases(config: Dict, genre: str, section_role: str, concept: Dict, notes: List[Dict], theme_len_bars: int | float, bpm: int | float, ts: Dict, cfg: Dict | None = None) -> Dict:
    """Stage 1b: Plan phrase structure and lyrical content based on concept and musical constraints."""
    # Input validation
    if not isinstance(config, dict):
        print("[Phrase Planning Error] Invalid config: must be dict")
        return {}
    if not isinstance(concept, dict):
        print("[Phrase Planning Error] Invalid concept: must be dict")
        return {}
    if not isinstance(notes, list):
        print("[Phrase Planning Error] Invalid notes: must be list")
        return {}
    if not isinstance(theme_len_bars, (int, float)) or theme_len_bars <= 0:
        print("[Phrase Planning Error] Invalid theme_len_bars: must be positive number")
        return {}
    if not isinstance(bpm, (int, float)) or bpm <= 0:
        print("[Phrase Planning Error] Invalid bpm: must be positive number")
        return {}
    
    try:
        import google.generativeai as genai_local
        # Force reload config to get latest model_name setting
        try:
            fresh_config = load_config(CONFIG_FILE)
            config = fresh_config
        except Exception as e:
            pass
        
        # Resolve model with priority: hotkey (this step) > config model > session override > lyrics_model
        global SESSION_MODEL_OVERRIDE, REQUESTED_SWITCH_MODEL
        model_name = REQUESTED_SWITCH_MODEL or config.get('model_name') or config.get('model') or SESSION_MODEL_OVERRIDE or config.get('lyrics_model', 'gemini-2.5-pro')
        
        # Calculate musical constraints
        bpb = max(1, int(ts.get('beats_per_bar', 4)))
        total_beats = float(theme_len_bars) * bpb
        num_notes = len(notes)
        
        phrase_prompt = f"""You are a lyrical phrase specialist. Structure the conceptual foundation into singable phrases.

CONCEPTUAL FOUNDATION:
- Core Theme: {concept.get('core_theme', 'Unknown')}
- Perspective: {concept.get('perspective', 'Unknown')}
- Key Imagery: {concept.get('key_imagery', [])}
- Mood: {concept.get('mood', 'Unknown')}
- Tone: {concept.get('tone', 'Unknown')}

MUSICAL CONSTRAINTS:
- Section Role: {section_role}
- Total Beats: {total_beats} | Notes: {num_notes}
- BPM: {bpm} | Time Signature: {bpb}/4

PHRASE STRUCTURE PLANNING:
- Plan 2-4 main phrases that fit the musical structure
- Each phrase should be 1-2 lines, 4-8 words
- Consider natural breathing points and musical phrasing
- Ensure phrases work with the note count and timing
- Maintain the established mood and imagery

ROLE-SPECIFIC PHRASING:
- {section_role}: Structure phrases appropriately for this role
- Consider repetition, development, or contrast as needed
- Ensure phrases serve the section's musical function

OUTPUT (JSON):
{{
  "phrases": ["string", ...] - 2-4 main lyrical phrases",
  "phrase_breaks": [int, ...] - note indices where phrases end",
  "repetition_strategy": "string - how to handle repetition",
  "breathing_points": [int, ...] - note indices for natural pauses",
  "emotional_arc": "string - how emotion develops through phrases"
}}"""
        
        # Add retry and API key rotation logic
        max_attempts = max(3, len(API_KEYS))
        attempts = 0
        quota_rotation_count = 0
        
        while attempts < max_attempts:
            try:
                # Configure model with current API key
                model = genai_local.GenerativeModel(model_name=model_name)
                
                resp = model.generate_content(phrase_prompt)
                raw = _extract_text_from_response(resp) or ""
                cleaned = raw.strip().replace("```json", "").replace("```", "")
                payload = _extract_json_object(cleaned)
                phrases = json.loads(payload or cleaned)
                
                if isinstance(phrases, dict) and phrases:
                    return phrases
                else:
                    raise ValueError("Empty or invalid phrases response")
                    
            except Exception as e:
                err = str(e).lower()
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    qtype = _classify_quota_error(err)
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    try:
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        avail = []
                        now = time.time()
                        for ix, _ in enumerate(API_KEYS):
                            tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                            avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                        print(Fore.MAGENTA + Style.BRIGHT + "[Phrase Planning:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                    except Exception:
                        pass
                    
                    # Try to rotate to next available key
                    n = len(API_KEYS)
                    rotated = False
                    stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                    for off in range(1, n+1):
                        idx = (CURRENT_KEY_INDEX + off*stride) % n
                        if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                            continue
                        try:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            rotated = True
                            break
                        except Exception:
                            continue
                    
                    if not rotated:
                        wait_s = 3
                        if _all_keys_daily_exhausted():
                            _schedule_hourly_probe_if_needed()
                            wait_s = _seconds_until_hourly_probe()
                        _interruptible_backoff(wait_s, config, context_label="Phrase Planning cooldown")
                        quota_rotation_count += 1
                        continue
                    else:
                        continue
                else:
                    # Non-quota error: brief pause then retry and count attempt
                    time.sleep(0.5)
                    attempts += 1
                    continue
        
        print(f"[Phrase Planning Error] Failed after {max_attempts} attempts")
        return {}
        
    except Exception as e:
        print(f"[Phrase Planning Error] Unexpected error: {e}")
        return {}

def _generate_lyrics_free_with_syllables(config: Dict, genre: str, inspiration: str, track_name: str, bpm: int | float, ts: Dict, section_label: str | None = None, section_description: str | None = None, context_tracks_basic: List[Dict] | None = None, user_prompt: str | None = None, history_context: str | None = None, theme_len_bars: int | float | None = None, cfg: Dict | None = None, key_scale: str | None = None, part_idx: int = 0) -> Dict:
    try:
        import google.generativeai as genai_local
    except Exception:
        raise RuntimeError("Lyrics Stage-1 unavailable: no LLM SDK")
    try:
        # Force reload config to get latest model_name setting
        try:
            fresh_config = load_config(CONFIG_FILE)
            config = fresh_config
        except Exception as e:
            pass
        
        # Force reset session override for lyrics generation to use config model
        global SESSION_MODEL_OVERRIDE
        SESSION_MODEL_OVERRIDE = None
        model_name = config.get("model_name") or config.get("model")
        try:
            if REQUESTED_SWITCH_MODEL:
                model_name = REQUESTED_SWITCH_MODEL
        except Exception:
            pass
        if not model_name:
            model_name = config.get("model_name", "gemini-2.5-pro")
        # Use config temperature settings (not from artifact)
        temperature = float(config.get("lyrics_temperature", config.get("temperature", 0.6)))
        generation_config = {"response_mime_type": "application/json", "temperature": temperature}
        model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
        # Extract musical parameters from JSON (not config.yaml)
        language = str(cfg.get("lyrics_language", "English")) if cfg else "English"
        # Use the passed key_scale parameter first, then fallback to cfg
        if key_scale and str(key_scale).strip():
            key_scale = str(key_scale).strip()
            pass
        else:
            key_scale = str(cfg.get("key_scale", "")).strip() if cfg else ""
        section_role = _normalize_section_role(section_label)
        # Prefer planned role from section_description if available (e.g., "[Plan] role=prechorus")
        try:
            if isinstance(section_description, str):
                m = re.search(r"\[\s*Plan\s*\].*?role\s*=\s*([a-zA-Z_]+)", section_description, re.IGNORECASE)
                if m:
                    plan_role = _normalize_section_role(m.group(1))
                    if plan_role:
                        section_role = plan_role
        except Exception:
            pass
        # Genre-specific lyric style (4x4/Dance/Psytrance etc.)
        try:
            genre_lc = str(genre or "").lower(); insp_lc = str(inspiration or "").lower()
        except Exception:
            genre_lc, insp_lc = "", ""
        is_dance_4x4 = any(k in genre_lc or k in insp_lc for k in ("psytrance","psy","trance","house","techno","edm","dance","infected mushroom"))
        # Role flags for selective gating
        rl = str(section_role or '').lower()
        is_chorus = (rl == 'chorus') or (isinstance(section_label, str) and 'drop' in section_label.lower())
        is_prechorus = (rl == 'prechorus')
        is_verse = (rl == 'verse')
        is_bridge = (rl == 'bridge')
        is_breakdown = (rl == 'breakdown')
        is_backing = (rl == 'backing')
        is_scat = (rl == 'scat')
        is_vowels = (rl == 'vowels')
        is_intro = (rl == 'intro')
        is_outro = (rl == 'outro')
        dance_lyric_block = (
            "GENRE LYRIC STYLE (4x4/Dance):\n"
            "- Prefer modular, non-linear phrasing over long narrative arcs across the whole song.\n"
            "- Chorus/title-drop should recur with consistent wording; verses use vivid images, slogans, or short fragments.\n"
            "- Allow chant/mantra moments; keep lines short and hook-centric; avoid story continuity across sections.\n"
            "- Pre-chorus: tighten phrasing, lift energy, hint the hook without fully stating it.\n\n"
        )
        # Compose brief summary of other tracks
        ctx_summary = []
        try:
            bpb = max(1, int(ts.get('beats_per_bar', 4)))
            for t in (context_tracks_basic or []):
                nm = get_instrument_name(t)
                rl = t.get('role', 'complementary')
                nlist = sorted(t.get('notes', []) or [], key=lambda x: float(x.get('start_beat', 0.0)))
                starts = [float(n.get('start_beat', 0.0)) for n in nlist]
                density = (len(nlist) / max(1e-6, ((max(starts)-min(starts)) if nlist else 0)/bpb)) if nlist else 0.0
                ctx_summary.append({"name": nm, "role": rl, "density": round(density, 2)})
        except Exception:
            ctx_summary = []

        # Extract an explicit hook text if present in description or user prompt
        hook_text_hint = None
        try:
            if isinstance(section_description, str):
                m1 = re.search(r"\[\s*Plan\s*\].*?hook_canonical\s*=\s*\"([^\"]+)\"", section_description, re.IGNORECASE)
                if m1:
                    hook_text_hint = m1.group(1).strip()
            try:
                pr_flag = int(config.get('pass_raw_prompt_to_stages', 0))
            except Exception:
                pr_flag = 0
            if (not hook_text_hint) and pr_flag and isinstance(user_prompt, str):
                m2 = re.search(r'"([^"\n]{3,64})"', user_prompt)
                if m2:
                    hook_text_hint = m2.group(1).strip()
        except Exception:
            hook_text_hint = None

        # Build role-specific gating block + minimal toolbox
        role_gate_block = []
        if is_chorus:
            role_gate_block.append("ROLE-SPECIFIC: Chorus/Drop → begin with the exact HOOK as the first line; keep a few short lines sized to the part; reuse hook across repeats; no long story arcs.")
            role_gate_block.append("Keep hook words unbroken; avoid syllable splitting inside hook.")
        elif is_prechorus:
            role_gate_block.append("ROLE-SPECIFIC: Pre-chorus → tighten phrasing; raise tension; hint hook without stating it; end with lift.")
        elif is_verse:
            role_gate_block.append("ROLE-SPECIFIC: Verse → advance narrative with concrete images; avoid chorus wording; one surreal twist max.")
        elif is_bridge:
            role_gate_block.append("ROLE-SPECIFIC: Bridge → contrast; fresh angle; avoid chorus wording; fewer new facts, more color.")
        elif is_backing:
            role_gate_block.append("ROLE-SPECIFIC: Backing → echo/answer lead with short repeatable fragments; no new content.")
        elif is_scat:
            role_gate_block.append("ROLE-SPECIFIC: Scat → musical syllables only; vary vowels; avoid monotony; no semantics.")
        elif is_vowels:
            role_gate_block.append("ROLE-SPECIFIC: Vowels → hold open vowels on long notes; no semantics; pure sustain.")
        elif section_role in ['whisper', 'hum']:
            role_gate_block.append("ROLE-SPECIFIC: Whisper/Hum → atmospheric, breathy, minimal meaningful words; create mood through texture; avoid pure vowels unless absolutely necessary.")
        elif section_role in ['adlib', 'vocal_fx', 'vocoder']:
            role_gate_block.append("ROLE-SPECIFIC: Adlib/VocalFX/Vocoder → experimental vocalizations, varied consonants, creative sound exploration; use meaningful words when possible, reserve pure vowels for special effects.")
        elif is_intro or is_outro or is_breakdown:
            role_gate_block.append("ROLE-SPECIFIC: Intro/Outro/Breakdown → minimal wording; allow silence; spotlight single striking image if any.")

        role_gate_text = ("ROLE GATING:\n- " + "\n- ".join(role_gate_block) + "\n") if role_gate_block else ""
        toolbox = _vocal_toolbox_block(
            1,
            is_chorus=is_chorus,
            is_drop=False,
            is_verse=is_verse,
            is_prechorus=is_prechorus,
            is_bridge=is_bridge,
            is_backing=is_backing,
            is_scat=is_scat,
            is_vowels=is_vowels,
            is_intro=is_intro,
            is_outro=is_outro,
            bpb=int(ts.get('beats_per_bar', 4)) if isinstance(ts, dict) else 4
        )

        # Handle silence parts with minimal content
        try:
            desc_lc = str(section_description or '').lower()
            if (section_role == 'silence') or ('role=silence' in desc_lc) or ('no vocal content' in desc_lc):
                # For silence parts, provide minimal atmospheric content instead of complete silence
                # Generate 2-3 atmospheric notes instead of just one
                atmospheric_words = ["ah", "oh", "mm"]
                atmospheric_syllables = [["ah"], ["oh"], ["mm"]]
                return {
                    "words": atmospheric_words, 
                    "syllables": atmospheric_syllables, 
                    "arranger_note": "minimal atmospheric content for silence role"
                }
        except Exception:
            pass

        # Multi-stage content generation
        print(f"{Fore.CYAN}🎵 Generating lyrics for '{section_label}'{Style.RESET_ALL}")
        
        # Stage 1a: Conceptual Planning
        concept = _plan_lyrical_concept(config, genre, inspiration, section_role, section_label, user_prompt, history_context, cfg)
        print(f"{Fore.GREEN}  📝 Theme: {concept.get('core_theme', 'Unknown')}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}  🎭 Mood: {concept.get('mood', 'Unknown')}{Style.RESET_ALL}")
        
        # Stage 1b: Phrase Planning (if we have notes)
        phrases = {}
        # For new vocal tracks, we need to work with the existing notes from context_tracks_basic
        if context_tracks_basic and len(context_tracks_basic) > 0:
            # Extract notes from context tracks for phrase planning
            all_notes = []
            for track in context_tracks_basic:
                if isinstance(track, dict) and 'notes' in track:
                    all_notes.extend(track['notes'])
            
            if all_notes:
                print(f"{Fore.CYAN}  🎵 Using {len(all_notes)} notes from context for phrase planning{Style.RESET_ALL}")
                # Create preview from context notes
                preview = []
                for note in all_notes[:20]:  # Limit to first 20 notes for preview
                    if isinstance(note, dict) and 'pitch' in note:
                        preview.append({
                            "start": note.get('start_beat', 0),
                            "dur": note.get('duration_beats', 1),
                            "pitch": note.get('pitch', 60),
                            "stress": 1 if note.get('velocity', 64) > 80 else 0
                        })
            else:
                print(f"{Fore.YELLOW}  ⏭️  No notes found in context tracks{Style.RESET_ALL}")
                preview = []
        else:
            print(f"{Fore.YELLOW}  ⏭️  No context tracks available{Style.RESET_ALL}")
            preview = []
        
        # Stage 1c: Final lyrics generation with concept and phrase guidance
        
        # Build enhanced prompt with concept and phrase guidance
        concept_guidance = ""
        if concept:
            concept_guidance = f"""
CONCEPTUAL FOUNDATION:
- Core Theme: {concept.get('core_theme', 'Unknown')}
- Perspective: {concept.get('perspective', 'Unknown')}
- Key Imagery: {', '.join(concept.get('key_imagery', []))}
- Mood: {concept.get('mood', 'Unknown')}
- Tone: {concept.get('tone', 'Unknown')}
- Narrative Function: {concept.get('narrative_function', 'Unknown')}

CONCEPTUAL GUIDANCE:
- Use the established theme and imagery consistently
- Maintain the chosen perspective and mood throughout
- Ensure the tone matches the vocal delivery approach
- Advance the narrative as planned for this section
"""
        
        phrase_guidance = ""
        if phrases and phrases.get('phrases'):
            phrase_guidance = f"""
PHRASE STRUCTURE:
- Planned Phrases: {phrases.get('phrases', [])}
- Repetition Strategy: {phrases.get('repetition_strategy', 'Unknown')}
- Emotional Arc: {phrases.get('emotional_arc', 'Unknown')}

PHRASING GUIDANCE:
- Structure content around the planned phrases
- Use the repetition strategy appropriately
- Follow the emotional development arc
- Maintain natural breathing points
"""

        # Prompt for free lyrics + syllables + optional arranger note (+ optional hook/chorus_lines)
        labels = _get_prompt_labels(config)
        line_ctx = _format_prompt_context_line(
            {
                'genre': genre,
                'language': language,
                'key_scale': key_scale,
                'bpm': round(float(bpm)),
                'time_signature': f"{ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}",
                'track': track_name,
                'section': (section_label or ''),
                'description': (section_description or '')
            },
            labels=labels
        )
        # Balanced prompt: simplified but complete
        prompt = (
            f"Generate lyrics for: {section_label} ({section_role})\n\n"
            + f"CONTEXT: {line_ctx}\n"
            + (f"THEME: {concept.get('core_theme', 'Unknown')}\n" if concept else "")
            + (f"MOOD: {concept.get('mood', 'Unknown')}\n" if concept else "")
            + (f"OTHER TRACKS: {json.dumps(ctx_summary)}\n" if ctx_summary else "")
            + (f"PREVIOUS SECTIONS (LYRICAL CONTEXT):\n{history_context}\n" if (isinstance(history_context, str) and history_context.strip()) else "")
            + (f"MELODY: {json.dumps(preview)}\n" if preview else "")
            + "\nSONG POSITION GATING:\n"
            + (f"- FIRST PART: You are creating the opening of the song. Establish the main theme, mood, and narrative foundation. Set up key imagery and emotional tone that will carry through the entire song.\n" if part_idx == 0 else "")
            + (f"- MIDDLE PART: You are in the middle of the song. Build on established themes, develop the narrative, and create emotional progression. Reference earlier parts while advancing the story.\n" if 0 < part_idx < 15 else "")
            + (f"- FINAL PART: You are concluding the song. Bring closure to the narrative, resolve emotional arcs, and create a satisfying ending. Reference earlier themes for unity.\n" if part_idx >= 15 else "")
            + (f"- CONSECUTIVE PARTS: You are following directly after the previous part. Create smooth transition and maintain narrative flow. Build directly on the previous section's content.\n" if part_idx > 0 else "")
            + "\nROLE-SPECIFIC RULES:\n"
            + (f"- CHORUS: Use hook '{hook_text_hint}' clearly with memorable lines\n" if (section_role == 'chorus' and hook_text_hint) else "")
            + (f"- VERSE: Progress the song with imagery, avoid chorus wording\n" if section_role == 'verse' else "")
            + (f"- PRE-CHORUS: Build tension, hint at hook\n" if section_role == 'prechorus' else "")
            + (f"- BRIDGE: Provide contrast, switch style if needed\n" if section_role == 'bridge' else "")
            + (f"- INTRO: Set tone economically, minimal gesture\n" if section_role == 'intro' else "")
            + (f"- OUTRO: Concluding vocals, fade out\n" if section_role == 'outro' else "")
            + (f"- BREAKDOWN: Reduce density, spotlight single idea\n" if section_role == 'breakdown' else "")
            + (f"- WHISPER: Very soft, breathy, sparse tokens\n" if section_role == 'whisper' else "")
            + (f"- SPOKEN: Conversational, natural prosody\n" if section_role == 'spoken' else "")
            + (f"- SHOUT: High energy emphasis, few words\n" if section_role == 'shout' else "")
            + (f"- CHANT: Mantra-like repetition, short phrases\n" if section_role == 'chant' else "")
            + (f"- ADLIB: Interjections around lead, sporadic\n" if section_role == 'adlib' else "")
            + (f"- HARMONY: Parallel/supporting lines\n" if section_role == 'harmony' else "")
            + (f"- HUM: Non-verbal vocalization, no words\n" if section_role == 'hum' else "")
            + (f"- BREATHS: Use '[br]' tokens only, no words\n" if section_role == 'breaths' else "")
            + (f"- VOCAL_FX: Special effects, texture role\n" if section_role == 'vocal_fx' else "")
            + (f"- VOCAL_FX: Special effects, texture role\n" if section_role == 'vocoder' else "")
            + (f"- VOCAL_FX: Special effects, texture role\n" if section_role == 'talkbox' else "")
            + (f"- RAP: Rhythmic speech, clear enunciation\n" if section_role == 'rap' else "")
            + (f"- CHOIR: Stacked voices, simple syllables\n" if section_role == 'choir' else "")
            + (f"- SILENCE: Use words=[] for instrumental only\n" if section_role == 'silence' else "")
            + "\nTEXT RULES:\n"
            + "- Use meaningful words, avoid generic vowels (ah/oh/eh)\n"
            + "- Keep phrases natural and singable\n"
            + "- Match musical mood and rhythm\n"
            + "- Avoid meta-words from instructions\n"
            + "- Prefer whole words, split only if rhythm requires\n"
            + "\nDENSITY & BREATHING (soft):\n"
            + "- Favor singable, memorable lines over sheer word count; let the music breathe.\n"
            + "- Leave small gaps between phrases when musically natural (avoid wall-to-wall text).\n"
            + "- If the note grid is dense, choose fewer/longer words; if sparse, choose concise imagery.\n"
            + "- Prioritize pronounceability and articulation; place consonant clusters on stable onsets.\n"
            + "\nLYRICAL COHERENCE (CRITICAL):\n"
            + "- Build on previous sections: reference themes, imagery, or emotions from earlier parts\n"
            + "- Maintain narrative flow: each section should advance the story or emotional journey\n"
            + "- Use consistent vocabulary and imagery throughout the song\n"
            + "- Create logical progression: intro → verse → pre-chorus → chorus → bridge → outro\n"
            + "- Reference earlier lyrics when appropriate to create unity\n"
            + "\nOUTPUT JSON:\n"
            + "{\n"
            + "  \"words\": [\"word1\", \"word2\", \"word3\"],\n"
            + "  \"syllables\": [[\"syl\", \"la\"], [\"ble\"]] or null,\n"
            + "  \"arranger_note\": \"optional note\" or null,\n"
            + "  \"hook_canonical\": \"hook phrase\" or null,\n"
            + "  \"chorus_lines\": [\"line1\", \"line2\"] or null,\n"
            + "  \"story\": \"brief story\" or null\n"
            + "}\n"
        )

        # Retry with key rotation and interruptible backoff on quota/429
        def _call_with_rotation_free(prompt_text: str) -> dict | None:
            nonlocal model
            max_attempts = max(3, len(API_KEYS))
            attempts = 0
            while attempts < max_attempts:
                try:
                    resp = model.generate_content(prompt_text)
                    raw = _extract_text_from_response(resp) or ""
                    cleaned = raw.strip().replace("```json", "").replace("```", "")
                    payload = _extract_json_object(cleaned)
                    obj = json.loads(payload or cleaned)
                    
                    # Pretty preview of AI result
                    if isinstance(obj, dict):
                        _w = obj.get('words') if isinstance(obj.get('words'), list) else []
                        _prev = " ".join(_w[:12]) + ("..." if len(_w) > 12 else "")
                        try:
                            print(Fore.MAGENTA + Style.BRIGHT + f"  🧠 AI LYRICS PREVIEW: \"{_prev}\"" + Style.RESET_ALL)
                        except Exception:
                            pass
                        return obj
                    else:
                        try:
                            _raw_prev = (raw[:160] + "...") if len(raw) > 160 else raw
                            print(Fore.MAGENTA + "  🧠 AI RAW (truncated): " + _raw_prev + Style.RESET_ALL)
                        except Exception:
                            pass
                        return None
                except Exception as e:
                    err = str(e).lower()
                    if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                        qtype = _classify_quota_error(err)
                        KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                        try:
                            cd = 60 if qtype not in ('per-hour','per-day') else 3600
                            KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                            avail = []
                            now = time.time()
                            for ix, _ in enumerate(API_KEYS):
                                tleft = max(0.0, KEY_COOLDOWN_UNTIL.get(ix, 0) - now)
                                avail.append(f"#{ix+1}:{'OK' if tleft<=0 else f'cooldown {int(tleft)}s'}")
                            print(Fore.MAGENTA + Style.BRIGHT + "[Lyrics:Quota] " + Style.NORMAL + Fore.WHITE + f"signal={qtype}; keys=" + ", ".join(avail) + Style.RESET_ALL)
                        except Exception:
                            pass
                        # rotate keys if possible
                        n = len(API_KEYS)
                        rotated = False
                        stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                        for off in range(1, n+1):
                            idx = (CURRENT_KEY_INDEX + off*stride) % n
                            if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                                continue
                            try:
                                globals()['CURRENT_KEY_INDEX'] = idx
                                genai.configure(api_key=API_KEYS[idx])
                                model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                                rotated = True
                                break
                            except Exception:
                                continue
                        if not rotated:
                            wait_s = 3
                            if qtype in ('per-hour','per-day'):
                                wait_s = 60 if qtype == 'per-hour' else 300
                            _interruptible_backoff(wait_s, config, context_label="Lyrics Stage-1 cooldown")
                            continue
                        else:
                            continue
                    # Non-quota error: brief pause then retry and count attempt
                    attempts += 1
                    time.sleep(1)
                    continue
            return None

        obj = _call_with_rotation_free(prompt)
        if isinstance(obj, dict):
            # Role-aware auto-repair for outro/vowels on empty words (prevents resume aborts)
            try:
                rl_fix = str(section_role or '').lower()
            except Exception:
                rl_fix = ''
            words_list = obj.get('words') if isinstance(obj.get('words'), list) else []
            if (rl_fix in ('outro','vowels','silence','breaths')) and (not words_list or len(words_list) == 0):
                # Allow empty words for these roles and mark intentional non-lexical/silence
                try:
                    obj['arranger_note'] = (obj.get('arranger_note','') + ' ; intentional silence: per role').strip()
                except Exception:
                    obj['arranger_note'] = 'intentional silence: per role'
            # Normalize optional fields
            if obj.get('syllables') is not None and not isinstance(obj.get('syllables'), list):
                obj['syllables'] = None
            if isinstance(obj.get('words'), list) and len(obj.get('words')) >= 1:
                return obj
            # Permit empty words for certain roles without failing Stage-1
            if rl_fix in ('outro','vowels','silence','breaths'):
                return obj
        raise ValueError("Lyrics Stage-1 invalid: expected non-empty 'words' list (syllables optional)")
    except Exception as e:
        raise e

# --- Heuristic hook inference from user intent (quoted or implied) ---
def _infer_hook_from_text(text: str | None) -> str | None:
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        t = text.strip()
        # Quoted phrase preferred
        m = re.search(r'"([^"\n]{3,64})"', t)
        if m:
            return m.group(1).strip()
        # Look for 'hook:' or 'hook' clause
        m2 = re.search(r'hook\s*[:\-]?\s*([^\n\.;]{3,120})', t, flags=re.IGNORECASE)
        if m2:
            phrase = m2.group(1).strip()
            words = re.findall(r"[A-Za-z'][A-Za-z']*", phrase)
            if not words:
                return None
            if len(words) > 9:
                words = words[:9]
            if len(words) < 3:
                return None
            return ' '.join(words)
        return None
    except Exception:
        return None

# --- Lyrics-first STAGE 2: Compose notes to syllables (uses full context) ---
def _compose_notes_for_syllables(config: Dict, genre: str, inspiration: str, track_name: str, bpm: float | int, ts: Dict, theme_len_bars: int, syllables: List[List[str]], arranger_note: str | None, context_tracks_basic: List[Dict] | None = None, key_scale: str | None = None, section_label: str | None = None, hook_canonical: str | None = None, chorus_lines: List[str] | None = None, section_description: str | None = None, lyrics_words: List[str] | None = None, cfg: Dict | None = None) -> Dict:
    try:
        import google.generativeai as genai_local
    except Exception:
        return {}
    try:
        # Force reload config to get latest model_name setting
        try:
            fresh_config = load_config(CONFIG_FILE)
            config = fresh_config
        except Exception as e:
            pass
        
        # Use session override if available, otherwise config (no flash fallback)
        global SESSION_MODEL_OVERRIDE
        model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
        try:
            if REQUESTED_SWITCH_MODEL:
                model_name = REQUESTED_SWITCH_MODEL
        except Exception:
            pass
        if not model_name:
            model_name = config.get("model_name", "gemini-2.5-pro")
        # Use a slightly higher temperature for Stage-2 to avoid deterministic loops
        try:
            _base_t = float(config.get("lyrics_temperature", config.get("temperature", 0.5)))
        except Exception:
            _base_t = 0.5
        _temp = max(0.3, min(1.0, _base_t))
        generation_config = {"response_mime_type": "application/json", "temperature": _temp}
        model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
        # Extract tessitura_center from plan_hint if available
        hint_tess = None
        try:
            # Try to extract tessitura_center from plan_hint
            if isinstance(arranger_note, str) and 'tessitura_center' in arranger_note:
                import re
                match = re.search(r'tessitura_center[:\s]*(\d+)', arranger_note)
                if match:
                    hint_tess = int(match.group(1))
        except Exception:
            pass
        
        # Context summary and available scale notes
        try:
            # Infer dynamic root_note from key_scale if not explicitly set
            def _infer_root_from_key(ks: str) -> int:
                try:
                    tonic = (ks or "").strip().split()[0].lower()
                    table = {
                        'c': 60, 'c#': 61, 'db': 61,
                        'd': 62, 'd#': 63, 'eb': 63,
                        'e': 64, 'fb': 64, 'e#': 65,
                        'f': 65, 'f#': 66, 'gb': 66,
                        'g': 67, 'g#': 68, 'ab': 68,
                        'a': 57, 'a#': 58, 'bb': 58,  # A3 instead of A4
                        'b': 59, 'cb': 59
                    }
                    return table.get(tonic, 60)
                except Exception:
                    return 60
            # Use artifact root_note and scale_type if available, otherwise fallback to config
            # Use the passed key_scale parameter first, then fallback to cfg/config
            if key_scale and str(key_scale).strip():
                key_scale_to_use = str(key_scale).strip()
            else:
                key_scale_to_use = str(cfg.get("key_scale", "") if cfg else config.get("key_scale", "")).strip()
            
            if cfg and isinstance(cfg.get("root_note"), (int, float)):
                root_note = int(cfg.get("root_note"))
            elif isinstance(config.get("root_note"), (int, float)):
                root_note = int(config.get("root_note"))
            else:
                root_note = _infer_root_from_key(key_scale_to_use)
            
            scale_type = str(cfg.get("scale_type", "") if cfg else config.get("scale_type", (key_scale_to_use or "major").split(' ',1)[-1] if key_scale_to_use else "major"))
            
            # SCALE VALIDATION: Check if the scale makes sense for the actual music
            print(f"[DEBUG] Scale Validation:")
            print(f"  - Key/Scale from JSON: {key_scale_to_use}")
            print(f"  - Root note (MIDI): {root_note}")
            print(f"  - Scale type: {scale_type}")
            print(f"  - Generated scale notes: {get_scale_notes(root_note, scale_type)[:12]}...")  # Show first 12 notes
            
            # Check for potential scale mismatches
            if key_scale_to_use and "major" in key_scale_to_use.lower() and "minor" in scale_type.lower():
                print(f"[WARNING] Scale mismatch detected: key_scale says major but scale_type is minor!")
            elif key_scale_to_use and "minor" in key_scale_to_use.lower() and "major" in scale_type.lower():
                print(f"[WARNING] Scale mismatch detected: key_scale says minor but scale_type is major!")
            
            # Use the scale from JSON - no automatic "correction"
            print(f"[INFO] Using scale from JSON: {key_scale_to_use} (root={root_note}, type={scale_type})")
        except Exception:
            root_note, scale_type = 60, "major"
        scale_notes = get_scale_notes(root_note, scale_type)
        print(f"[DEBUG] Composer Scale Info: root_note={root_note}, scale_type={scale_type}, scale_notes={scale_notes}")
        bpb = int(ts.get('beats_per_bar', 4)) if isinstance(ts, dict) else 4
        # Full context of other tracks (send full notes of this part)
        context_full = []
        try:
            for t in (context_tracks_basic or []):
                nm = get_instrument_name(t)
                rl = t.get('role','complementary')
                # Skip current vocal track if present in summary
                if isinstance(nm, str) and 'vocal' in nm.lower():
                    continue
                nts = sorted(t.get('notes', []) or [], key=lambda x: float(x.get('start_beat', 0.0)))
                context_full.append({"name": nm, "role": rl, "notes": nts})
        except Exception:
            context_full = []
        # Build strict composition prompt
        role_norm = _normalize_section_role(section_label)
        # Prefer planned role from section_description if available (e.g., "[Plan] role=prechorus")
        try:
            if isinstance(section_description, str):
                m = re.search(r"\[\s*Plan\s*\].*?role\s*=\s*([a-zA-Z_]+)", section_description, re.IGNORECASE)
                if m:
                    plan_role = _normalize_section_role(m.group(1))
                    if plan_role:
                        role_norm = plan_role
        except Exception:
            pass
        is_drop = isinstance(section_label, str) and ('drop' in section_label.lower())
        is_chorus = (role_norm == 'chorus') or is_drop
        is_verse = (role_norm == 'verse')
        is_prechorus = (role_norm == 'prechorus')
        is_bridge = (role_norm == 'bridge')
        is_backing = (role_norm == 'backing')
        is_spoken = (role_norm == 'spoken')
        is_whisper = (role_norm == 'whisper')
        is_shout = (role_norm == 'shout')
        is_chant = (role_norm == 'chant')
        is_adlib = (role_norm == 'adlib')
        is_harmony = (role_norm == 'harmony')
        is_doubles = (role_norm == 'doubles')
        is_response = (role_norm == 'response')
        is_rap = (role_norm == 'rap')
        is_choir = (role_norm == 'choir')
        is_hum = (role_norm == 'hum')
        is_breaths = (role_norm == 'breaths')
        is_tag = (role_norm == 'tag')
        is_phrase_spot = (role_norm == 'phrase_spot')
        is_vocoder = (role_norm == 'vocoder')
        is_talkbox = (role_norm == 'talkbox')
        is_vocal_fx = (role_norm == 'vocal_fx')
        is_scat = (role_norm == 'scat')
        is_vowels = (role_norm == 'vowels')
        is_intro = (role_norm == 'intro')
        is_outro = (role_norm == 'outro')
        is_breakdown = (role_norm == 'breakdown')
        # Quick skip if arranger_note signals intentional silence
        try:
            if isinstance(arranger_note, str) and 'intentional silence' in arranger_note.lower():
                return {"notes": [], "tokens": []}
        except Exception:
            pass

        # Extract numeric hint parameters from section_description (if present)
        hint_tess = None
        hint_span = None
        hint_maxup = None
        try:
            if isinstance(section_description, str) and section_description:
                m1 = re.search(r"tessitura_center\s*=\s*(\d+)", section_description, re.IGNORECASE)
                m2 = re.search(r"pitch_span_semitones\s*=\s*(\d+)", section_description, re.IGNORECASE)
                m3 = re.search(r"max_unique_pitches\s*=\s*(\d+)", section_description, re.IGNORECASE)
                if m1:
                    hint_tess = int(m1.group(1))
                if m2:
                    hint_span = int(m2.group(1))
                if m3:
                    hint_maxup = int(m3.group(1))
        except Exception:
            hint_tess = hint_tess if isinstance(hint_tess, int) else None
            hint_span = hint_span if isinstance(hint_span, int) else None
            hint_maxup = hint_maxup if isinstance(hint_maxup, int) else None

        labels = _get_prompt_labels(config)
        line_ctx = _format_prompt_context_line(
            {
                'genre': genre,
                'key_scale': key_scale or (cfg.get('key_scale','') if cfg else config.get('key_scale','?')),
                'bpm': round(float(bpm)),
                'time_signature': f"{bpb}/{ts.get('beat_value','?')}",
                'length_bars': theme_len_bars,
            },
            labels=labels
        )
        # Per-call nonce to decorrelate near-identical prompts across runs/parts
        try:
            from uuid import uuid4
            _nonce = str(uuid4())[:8]
        except Exception:
            import random
            _nonce = hex(random.getrandbits(24))[2:]

        # Target onsets per bar for shaping density
        try:
            target_wpb = float(config.get('lyrics_target_words_per_bar', 2.0))
        except Exception:
            target_wpb = 2.0

        # Structured basic instructions like instrument prompts
        basic_instructions = (
            f"**Role:** Vocal Composer\n"
            + f"**Section:** {line_ctx} bars\n"
            + f"**Song Key:** {scale_type.upper()} - Root: {root_note} (MIDI {root_note})\n"
            + f"**Available Notes:** {scale_notes}\n"
            + f"**Planned Role:** {role_norm}\n" if isinstance(role_norm, str) and role_norm else ""
            + f"**Section Description:** {section_description.strip()}\n" if isinstance(section_description, str) and section_description.strip() else ""
            + f"**Hook:** \"{hook_canonical.strip()}\"\n" if isinstance(hook_canonical, str) and hook_canonical.strip() else ""
        )

        # Context information
        context_info = ""
        if context_full:
            context_info += f"**Context Tracks (other instruments in this section):**\n{json.dumps(context_full)}\n\n"
        if any(isinstance(x, dict) and x.get('name')=='__PRIOR_VOCAL__' for x in (context_tracks_basic or [])):
            context_info += f"**Previous Vocal Parts:**\n{json.dumps([x for x in (context_tracks_basic or []) if isinstance(x, dict) and x.get('name')=='__PRIOR_VOCAL__'])}\n\n"
        if isinstance(arranger_note, str) and arranger_note.strip():
            context_info += f"**Arranger Note:** {arranger_note.strip()}\n\n"

        # Lyrics information
        lyrics_info = ""
        if isinstance(lyrics_words, list) and lyrics_words:
            lyrics_info += f"**Lyrics to Set:**\n{json.dumps([w for w in lyrics_words])}\n\n"
        lyrics_info += f"**Syllable Input (optional):**\n{json.dumps(syllables or [])}\n\n"

        prompt = (
            basic_instructions + "\n"
            + context_info
            + lyrics_info
            + "SYLLABLE-TO-NOTE MAPPING INTELLIGENCE:\n"
            + "- Analyze syllable count per word and map accordingly:\n"
            + "  • 1 syllable → 1 note (1.0-2.0 beats) - NO SHORTER!\n"
            + "  • 2 syllables → 1-2 notes (use melisma '-' SPARINGLY)\n"
            + "  • 3+ syllables → 2+ notes with melisma for natural flow\n"
            + "- Consider word stress patterns: stressed syllables get longer notes\n"
            + "- Match note durations to natural speech rhythm\n"
            + "- ARTICULATION PRIORITY: Keep words clear and singable\n"
            + "  • **AVOID MICRO-NOTES**: Never create notes < 0.6 beats\n"
            + "  • Prefer longer, more singable note durations\n"
            + "  • Only use melisma when musically justified, not as default\n\n"
            + f"🎵 SONG KEY: {scale_type.upper()} - Root: {root_note} (MIDI {root_note})\n"
            + f"🎵 AVAILABLE PITCHES: {scale_notes}\n\n"
            + ("NUMERIC HINTS (from plan):\n"
               + (f"- tessitura_center={hint_tess}\n" if isinstance(hint_tess, int) else "")
               + (f"- pitch_span_semitones={hint_span}\n" if isinstance(hint_span, int) else "")
               + (f"- max_unique_pitches={hint_maxup}\n" if isinstance(hint_maxup, int) else "")
               + ("\n" if any(isinstance(x, int) for x in (hint_tess, hint_span, hint_maxup)) else "")
            )
            + f"NONCE: { _nonce }\n\n"
            + "PROSODY & MAPPING POLICY:\n"
            + "- Map exactly one lyric token to one note. Use '-' tokens to extend the previous note (melisma).\n"
            + "- Avoid inserting micro rests (< 0.5 beats); extend the preceding note instead.\n"
            + "- Place stressed syllables on stronger beats and stable tones; function words on weak parts.\n"
            + "- Target onsets per bar ≈ " + str(round(target_wpb,2)) + "; do not exceed it unless musically justified by role/plan.\n"
            + "- Favor legato and coherent phrase arcs over fragmented mapping.\n"
            + "- DURATION & RHYTHM: Create natural, singable rhythms:\n"
            + "  • Match note lengths to the natural flow of the words\n"
            + "  • Consider how a singer would naturally phrase these lyrics\n"
            + "  • Use melisma ('-') when it serves the musical expression\n"
            + "  • Balance between clear articulation and smooth flow\n"
            + "  • Think about breathing and natural pauses\n"
            + "  • Avoid over-fragmenting words unless it serves a musical purpose\n"
            + "  • **Micro-timing for Groove**: Slightly anticipate beats (pushing) for urgency, or delay them (pulling) for relaxed feel\n"
            + "  • **Human Feel**: Add subtle rhythmic variation to avoid robotic precision\n\n"
            + "MUSICAL EXCELLENCE:\n"
            + "- Every note needs a pitch value - this is required for the system to function.\n"
            + "- Create melodies that would make a professional vocalist excited to perform them.\n\n"
            + "**CRITICAL: USE ALL CONTEXT INFORMATION**:\n"
            + "- **ANALYZE OTHER TRACKS**: Study the rhythm, melody, and harmony of other instruments.\n"
            + "- **MATCH MUSICAL STYLE**: Your vocal must fit the genre, mood, and energy of the backing tracks.\n"
            + "- **RESPECT LYRICS MEANING**: Set the provided lyrics with appropriate emotion and phrasing.\n"
            + "- **FOLLOW SECTION ROLE**: If this is a chorus, be bold and memorable. If verse, be narrative.\n"
            + "- **COMPLEMENT, DON'T COMPETE**: Fill gaps in the music, don't crowd dense sections.\n"
            + "- **THINK LIKE A PRODUCER**: Your vocal is one element in a complete arrangement.\n\n"
            + "CREATIVE INTERPRETATION:\n"
            + "- Each vocal role has its own character - interpret this musically.\n"
            + "- Listen to the backing music - create melodies that complement the overall sound.\n"
            + "- Think like a singer - where would you naturally breathe, emphasize, or add expression?\n"
            + "- Use pitch movement to support the emotional narrative of the lyrics.\n\n"
            + "**MUSICAL STRUCTURE & EVOLUTION:**\n"
            + "- **Motif Development**: Introduce a core melodic idea and develop it through variation, repetition, and contrast\n"
            + "- **Avoid Robotic Repetition**: Don't just repeat the same pattern - evolve and grow your musical ideas\n"
            + "- **Tension & Release**: Build musical tension through pitch movement and resolve it at phrase endings\n"
            + "- **Voice-leading**: Create smooth melodic movement between notes - think about how each note leads to the next\n"
            + "- **Passing Tones**: Use scale notes predominantly, but occasional passing tones can add musical interest\n"
            + "- **Clarity through Space**: Don't create a constant wall of sound - use rests effectively for musical impact\n"
            + "- **Dynamic Phrasing**: Use pitch variation to create accents and shape the energy of your phrases\n\n"
            + "CADENCE & FORM:\n"
            + "- Favor cadences on scale degrees 1/3/5; use 4→3 and 7→1 resolutions near phrase ends.\n"
            + "- One ornamental gesture per 4 bars; otherwise prefer stepwise motion and clear arcs.\n\n"
            + "**CRITICAL: ADAPT TO OTHER TRACKS**:\n"
            + "- **IF OTHER TRACKS ARE DENSE (many notes)**: Use FEWER notes, LONGER durations (≥3.0 beats).\n"
            + "- **IF OTHER TRACKS HAVE GAPS**: Fill space with more notes, but still maintain ≥2.0 beats minimum.\n"
            + "- **NEVER COMPETE WITH DENSE TRACKS**: Less is more. Create breathing space.\n"
            + "- **THINK LIKE AN ARRANGER**: Your vocal must complement, not crowd the mix.\n\n"
            + "**HOW TO USE CONTEXT TRACKS**:\n"
            + "- **RHYTHM ANALYSIS**: Match the rhythmic feel of drums/bass - don't fight the groove.\n"
            + "- **HARMONIC SUPPORT**: Use notes that work with the chord progressions in other tracks.\n"
            + "- **DYNAMIC BALANCE**: If other tracks are busy, be sparse. If sparse, add more vocal content.\n"
            + "- **MELODIC INTERACTION**: Create call-and-response with lead instruments, or harmonize with them.\n"
            + "- **EMOTIONAL MATCHING**: Match the energy level and mood of the backing tracks.\n\n"
            + "PHRASING HINTS BY ROLE:\n"
            + "- intro/whisper/spoken: fewer onsets, longer durations (≥ 1.0–1.25 beats typical).\n"
            + "- phrase_spot/prechorus: legato motifs, default min durations ≥ 1.0 beats when in doubt.\n"
            + "- chorus: clear arcs; repetition is fine; avoid syllable-chopping (< 0.5 beats).\n"
            + "- vocal_fx/breaths: very sparse; musical placement; lots of space.\n"
            + "- monotone roles: add rhythmic variation, micro-melodies (±2-3 semitones), dynamic accents.\n\n"
            + _vocal_toolbox_block(
                2,
                is_chorus=is_chorus,
                is_drop=is_drop,
                is_verse=is_verse,
                is_prechorus=is_prechorus,
                is_bridge=is_bridge,
                is_backing=is_backing,
                is_scat=is_scat,
                is_vowels=is_vowels,
                is_intro=is_intro,
                is_outro=is_outro,
                bpb=bpb
            )
            + "VOCAL EXCELLENCE FOR SYNTHV:\n"
            + "- Use complete, meaningful words that flow naturally when sung\n"
            + "- Choose words that SynthV can pronounce beautifully and clearly\n"
            + "- Prefer words with open vowels (ah, oh, oo, ee) for sustained notes\n"
            + "- Create phrases that feel organic and authentic to the musical context\n\n"
            + "🚨 COMPOSITION GUIDANCE (musical, flexible, anti-micro-note):\n"
            + "- **DURATION SHAPING (soft)**: Prefer 0.75–3.0 beats; hard floor 0.5. Mix short pickups, medium flow notes, longer anchors.\n"
            + "- **PHRASES (soft)**: 3–6 words per phrase, natural breathing and small gaps.\n"
            + "- **FLOW & AIR**: Connect notes organically, but leave occasional space (avoid wall-to-wall filling).\n"
            + "- **RHYTHMIC VARIETY (soft)**: Use occasional offbeats/syncopation; avoid long runs of identical durations.\n"
            + "- **OCCUPANCY (soft)**: Let vocals share space with the arrangement; don't fill every bar end-to-end unless the style demands it.\n"
            + "- **ARRANGEMENT AWARENESS (soft)**:\n"
            + "  • Scan other tracks' notes: favor entries where space is available; avoid masking lead hooks.\n"
            + "  • If the texture is dense: support harmonically (unison/thirds/sixths, sustained pads, echoes), reduce onset density.\n"
            + "  • If the texture is sparse: take brief spotlight with short phrases or one-word fills on meaningful accents.\n"
            + "  • Use call-and-response with prominent instruments; avoid clashing rhythms; leave room for kick/snare accents.\n"
            + "- **MUSICAL COHERENCE**: Each phrase should form a complete musical thought.\n"
            + "- **STYLE OVERRIDE (soft)**: If the genre/inspiration strongly suggests full bars of vocals, it's okay to fill; otherwise prefer some space and contrast.\n\n"
            + "**IMPORTANT RULES:**\n"
            + '1.  **JSON OBJECT ONLY:** Your entire response MUST be only the raw JSON object, starting with "{" and ending with "}".\n'
            + f'2.  **CRITICAL - PITCH REQUIRED:** Every single note MUST have a "pitch" field with a MIDI note number from {scale_notes}.\n'
            + f'3.  **Stay in Key:** Only use pitches from the provided list of scale notes: {scale_notes}.\n'
            + f'4.  **Minimum Duration (soft):** Prefer 0.75–3.0 beats; avoid <0.5. Use 0.5–0.75 sparingly as pickups.\n'
            + f'5.  **Required Fields:** Every note MUST have: start_beat, duration_beats, pitch\n'
            + f'6.  **Timing is Absolute:** start_beat is the absolute position from the beginning of the section.\n\n'
            + "**OUTPUT FORMAT:**\n"
            + "- Return JSON with 'notes' array containing objects with: start_beat, duration_beats, pitch\n"
            + f"- Example: {{\"notes\": [{{\"start_beat\": 0.0, \"duration_beats\": 2.5, \"pitch\": {root_note}}}, {{\"start_beat\": 3.0, \"duration_beats\": 2.0, \"pitch\": {root_note + 2}}}, {{\"start_beat\": 5.5, \"duration_beats\": 2.5, \"pitch\": {root_note + 4}}}]}}\n"
            + f"- **VARIETY EXAMPLE**: Notice how the example uses {root_note}, {root_note + 2}, and {root_note + 4} - different scale notes!\n"
            + f"- **CRITICAL WARNING**: If you don't include 'pitch' in every note, the system will default to C4 (MIDI 60)!\n\n"
            + "VARIATION POLICY:\n"
            + "- Prefer subtle variation over strict loops, unless the context suggests repetition.\n"
            + "- For spoken/whisper roles, repetition of words is acceptable; prioritize timing/placement over pitch variety.\n"
            + "- **WORD INTEGRITY**: The lyrics were already tokenized for you. Keep words intact on single notes when possible.\n"
            + "- **ANTI-MICRO-NOTE ENFORCEMENT**: Avoid <0.5 beats; keep 0.5–0.75 sparse and purposeful (pickups, passing).\n"
            + "- Aim for musical intent: when in doubt, choose clarity and groove over forced variation.\n"
            + f"- **SCALE EXPLORATION**: Challenge yourself to use at least 3-4 different notes from {scale_notes}!\n"
            + f"- **ROOT NOTE RETURN**: Always return to {root_note} (MIDI {root_note}) to ground your melody.\n\n"
            + "**ROLE-SPECIFIC GUIDANCE:**\n"
            # Role-specific guidance blocks (only emitted for the active role)
            + ("VERSE FOCUS (role-specific):\n- Advance the narrative with concrete images.\n- Keep phrasing punchy and rhythmic; prefer short words/lines.\n- Two-phrase A–A' per 4 bars; A' is a minimal variation.\n\n" if is_verse else "")
            + ("PRE-CHORUS FOCUS (role-specific):\n- Build tension and lift into the chorus; rising contour preferred.\n- Reuse a short priming phrase; avoid revealing the hook.\n\n" if is_prechorus else "")
            + ("BRIDGE FOCUS (role-specific):\n- Provide contrast in color/angle; introduce a complementary motif.\n- Keep range moderate; avoid direct chorus wording.\n\n" if is_bridge else "")
            + ("BACKING FOCUS (role-specific):\n- Echo/answer the lead with musical phrases; avoid micro-fragments.\n- Stay out of the way; simple contours and sparse rhythm.\n- **MINIMUM DURATION**: Each note ≥ 2.0 beats for musical coherence.\n\n" if is_backing else "")
            + ("SPOKEN FOCUS (role-specific):\n- Natural spoken phrases with musical rhythm; use 1-2 pitches for musicality.\n- Use lexical words (no pure 'Ah/Ooh/Mmm' placeholders).\n- **MINIMUM DURATION**: Each note ≥ 2.0 beats for natural speech flow.\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 2-3 different scale notes for natural speech patterns\n\n" if is_spoken else "")
            + ("WHISPER FOCUS (role-specific):\n- Very quiet, breathy words/phrases; leave space.\n- Do NOT use pure vowel placeholders ('Ah/Ooh/Mmm'); use lexical tokens.\n- **MINIMUM DURATION**: Each note ≥ 2.0 beats for natural breathy flow.\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 2-3 different scale notes for intimate expression\n\n" if is_whisper else "")
            + ("CHANT FOCUS (role-specific):\n- Simple repetitive cells sized to bar; minimal vocabulary; percussive alignment.\n- Avoid lyrical progression; keep motif tight.\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 2-3 different scale notes for rhythmic patterns\n\n" if is_chant else "")
            + ("ADLIB FOCUS (role-specific):\n- Occasional interjections around lead phrases; extremely sparse.\n- Prefer short ascents/descents; avoid stepping on main content.\n\n" if is_adlib else "")
            + ("HARMONY/DOUBLES FOCUS (role-specific):\n- Support lead on sustained notes or exact doubles; do not introduce new text.\n- Keep lower velocity/volume implied; align tightly.\n\n" if (is_harmony or is_doubles) else "")
            + ("RESPONSE FOCUS (role-specific):\n- Call-and-response one-liners; answer lead with 1–2 words.\n- Ensure rests before/after; avoid overlap.\n\n" if is_response else "")
            + ("RAP FOCUS (role-specific):\n- Rhythm-first syllables; keep internal rhyme optional but minimal.\n- Avoid overfilling; respect micro-onset rules.\n\n" if is_rap else "")
            + ("CHOIR/HUM FOCUS (role-specific):\n- Sustained vowels (hum/aa/oo) in simple chords/unison; no semantics.\n- Very slow movement; long sustains.\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 2-4 different scale notes for harmonic richness\n\n" if (is_choir or is_hum) else "")
            + ("BREATHS/TAG FOCUS (role-specific):\n- Breaths/noise as musical cues; or tiny end-tags (1–2 words).\n- Extremely sparse; do not crowd.\n- If role=breaths: create 1–2 short onsets mapped to the explicit token '[br]'; do NOT use words.\n\n" if (is_breaths or is_tag) else "")
            + ("SILENCE FOCUS (role-specific):\n- Minimal atmospheric content - use single sustained vowel 'ah' or 'oh'\n- Very sparse placement - 1-2 notes maximum\n- Low volume, sustained tones for atmospheric effect\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 2-3 different scale notes for subtle atmospheric movement\n\n" if (role_norm == 'silence') else "")
            + ("PHRASE_SPOT FOCUS (role-specific):\n- Single short phrase placed in a free window; everything else rests.\n- Duration ≤ 1 bar; clear entry/exit.\n\n" if is_phrase_spot else "")
            + ("VOCODER/TALKBOX/FX FOCUS (role-specific):\n- Treat tokens as syllabic carriers; simple pitch shapes; sparse usage.\n- Avoid semantic density; use rests generously.\n- If role=vocal_fx: output at least 1–2 short onsets within the part; if no musical placement is feasible, set intentional_silence=true in Stage-1 (preferred).\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 3-6 different scale notes for experimental textures\n\n" if (is_vocoder or is_talkbox or is_vocal_fx) else "")
            + ("SCAT FOCUS (role-specific):\n- Use percussive, non-lexical syllables (da/ka/tek...).\n- Lock to groove accents; avoid long sustains and semantics.\n\n" if is_scat else "")
            + ("VOWELS FOCUS (role-specific):\n- Sustained open vowels on long notes; no semantics.\n- Favor legato and simple stepwise contour.\n- **MANDATORY**: Include pitch information for every note\n- **SCALE AWARENESS**: Use notes from the song's scale: {scale_notes}\n- **ROOT CENTERING**: Center around {root_note} (MIDI {root_note}) for musical coherence\n- **PITCH VARIATION**: Use 2-4 different scale notes for melodic flow\n\n" if is_vowels else "")
            + ("INTRO FOCUS (role-specific):\n- Set the tone with a sparse gesture; minimal new content.\n- Prefer longer sustains and clear downbeat anchoring.\n\n" if is_intro else "")
            + ("IMPORTANT DROP STYLE (recommendation):\n- Prefer monotonic or 2-note pitch sets with small stepwise motion.\n- Use repetitive, hypnotic rhythm; micro-variations only.\n- Favor hook tokens/title words; keep lines succinct and percussive.\n\n" if is_drop else "")
            + ("MOTIF & SIMPLICITY (soft):\n- Build a compact motif and reuse with minimal change per repeat (size it to the part).\n- Limit pitch set; use longer sustains where possible.\n- Align main syllables to clear accents; avoid filler between accent peaks.\n\n")
            + ("DROP ONSET PREFERENCE (soft):\n- For DROP-like sections, keep onset density moderate relative to texture and tempo; brief peaks are fine if musically justified.\n- Prefer turning minor notes into '-' sustains on open vowels rather than adding new onsets when texture feels crowded.\n\n" if is_drop else "")
            + "ANTI-LOOP CONTROL (soft):\n- Avoid three identical bars in a row; prefer subtle micro-variation across repeats.\n\n"
            + "HOOK USAGE (hard for CHORUS/DROP):\n"
            + "- Use hook_canonical verbatim at least once per chorus/drop occurrence (place where it best serves form and length).\n"
            + "- Map ONE note per hook word in order; do NOT split a hook word across multiple tokens/notes.\n"
            + "- Keep hook words unbroken; prefer whole words over syllable splits.\n"
            + "- Keep hook tokens contiguous (no rests inside); prefer sustaining with '-' rather than inserting short notes.\n"
            + "- Outside CHORUS/DROP: avoid exact hook; hint with synonyms/metaphor if needed.\n\n"
            + "DENSITY-AWARE MAPPING (recommendations):\n- If onset rate feels low: increase melisma; hold open vowels with '-' on weak beats; avoid creating micro rests.\n- If onset rate feels moderate/high: prefer one syllable per note; avoid over-fragmenting short words.\n- Inside a word: do not create gaps; continuation notes must start exactly at previous end.\n- Minimum content duration: avoid launching new content on notes shorter than ≈1/8 beat; prefer '-' sustains.\n\n"
            + "SMOOTHING & LEGATO (recommendations):\n- Avoid very short notelets unless they land on a clear accent; otherwise merge into adjacent durations.\n- Prefer legato within words: extend previous duration instead of inserting micro-gaps/rests.\n- Use inter-word gaps only when musically justified; otherwise smooth by sustain.\n- Long vowels (a/ah/o) should carry sustained notes; consonant clusters avoid fragmentation.\n\n"
            + "MICRO-NOTE HANDLING (soft):\n- Do NOT create new onsets on very short notes; prefer '-' sustain or a rest.\n- For short consecutive notes on the same pitch: map them under a single token; fill intervening notes with '-'.\n- Place consonant attacks primarily on longer or on-beat notes; avoid launching full syllables on micro-notes.\n- When density feels high, lower onset count by sustaining open vowels rather than adding tokens.\n\n"
            + "AESTHETIC GUIDELINES (genre/context-led):\n"
            + "- Prioritize natural, singable contours; prefer stepwise motion; resolve leaps.\n"
            + "- Match density to context (other tracks, hint onset_count/duration_min); avoid over-filling.\n"
            + "- Land open vowels (a/ah/o/ai) on longer/stressed notes; keep i/e short (especially near the top of range).\n"
            + "- Keep tessitura centered; avoid lingering at register extremes.\n"
            + "- Phrase endings on downbeats or clear rests; avoid trailing micro-gaps (<1/8 beat).\n"
            + "- If in doubt, simpler is better: fewer notes, longer sustains, clearer motifs.\n"
            + ("- Chorus/Drop range discipline: keep span compact (≈≤ 5 semitones); anchor the hook onset on a downbeat; reuse the exact hook contour in repeats.\n" if is_chorus else "")
            + ("- Verse call-and-response (light): two-phrase A–A' structure per 4 bars; A' is a minimal variation of A.\n" if is_verse else "")
            + "- Contour budget: within each 4-bar unit, allow only one small ornamental gesture; keep the rest flat.\n\n"
            + ("HOOK DURATION & PLACEMENT (soft for chorus/drop):\n- Aim to start the first hook usage on a strong accent; keep hook tokens as contiguous as the melody reasonably allows.\n- Prefer one note per hook word in order; avoid splitting a single hook word.\n- Sustain core hook vowels noticeably; avoid micro-slicing.\n- If the melody is highly chopped with large rests, map the hook in compact segments that respect musical phrasing rather than forcing a fully contiguous chain.\n\n" if (is_chorus or is_drop) else "")
            + ("OUTRO MINIMUM (hard for outro):\n- Output MUST include at least one note and one token.\n- If no lyrics words provided, you MAY use a single sustained open vowel (e.g., 'Ah').\n- Prefer smooth, sustained contour; avoid dense onsets; no empty arrays.\n\n" if is_outro else "")
            + "OUTPUT (STRICT JSON):\n{\n  \"notes\": [{\"start_beat\": number, \"duration_beats\": number, \"pitch\": int}, ...],\n  \"tokens\": [string, ...]\n}\n\n- Only these two top-level keys are allowed: notes, tokens. No other keys.\n\n"
            + "CONSTRAINTS:\n- Map the provided lyrics in order; you MAY split words or use '-' for sustained continuations.\n- Try to keep len(tokens) ≈ number of notes; exact equality is NOT required if musically justified (but avoid large mismatches).\n- Preserve reading order of words; do not spell letter-by-letter.\n- Clamp total beats to theme length (bars * beats_per_bar); no overlaps; no negative durations.\n"
            + "- HARD: If tokens are provided (len(tokens) ≥ 1), notes MUST NOT be empty (len(notes) ≥ 1). Never return empty notes when tokens exist.\n"
            + "- DURATION QUALITY: Ensure all note durations are musically appropriate:\n"
            + "  • No notes shorter than 0.5 beats unless absolutely necessary\n"
            + "  • Match duration to syllable/word complexity and natural speech rhythm\n"
            + "  • Create natural, singable phrasing with adequate note lengths\n"
            + "  • Use melisma ('-') for multi-syllable words rather than creating too many short notes\n"
            + "- ARTICULATION CLARITY: Prioritize clear, singable word delivery:\n"
            + "  • Keep simple words (1-2 syllables) as single notes when possible\n"
            + "  • Avoid over-fragmenting words into micro-notes that are hard to sing\n"
            + "  • Use longer note durations for better word clarity and musicality\n"
            + "  • Only split words when rhythmically essential, not as default approach\n"
            + "- NATURAL PHRASING: Think like a singer delivering a coherent message:\n"
            + "  • Group words into meaningful phrases, not isolated individual words\n"
            + "  • Use melisma to connect words within phrases for natural flow\n"
            + "  • Avoid choppy, word-by-word delivery that breaks sentence meaning\n"
            + "  • Create flowing, understandable phrases with natural pauses\n"
            + "  • MINIMUM PHRASE LENGTH: Create phrases of 3-5 words minimum\n"
            + "  • AVOID MICRO-NOTES: Don't split words into tiny fragments\n"
            + "  • SINGLE WORDS: Use melisma only if it enhances phrase flow\n"
            + "  • MULTI-SYLLABLE WORDS: Use melisma to maintain phrase coherence\n"
            + "  • MAXIMUM MELISMA: Never exceed 20% of total tokens as melisma\n"
            + "  • VOCAL COHERENCE: Prioritize meaning and flow over strict note-to-syllable mapping\n"
            + "- Only these two top-level keys are allowed: notes, tokens. No other keys.\n\n"
            + ("- HARD (breaths): If lyrics contain '[br]', produce at least one short note per '[br]'.\n" if is_breaths else "")
            + ("- HARD (vocal_fx): If lyrics are provided, produce at least one short onset; if impossible, prefer intentional silence (model-driven).\n" if is_vocal_fx else "")
            + "\n**FINAL REMINDER:**\n"
            + f"- Use ONLY these scale notes: {scale_notes}\n"
            + f"- Center melodies around {root_note} (MIDI {root_note})\n"
            + "- Each note ≥ 0.6 beats for singable phrasing\n"
            + "- Return valid JSON with 'notes' array\n"
            + f"- **MANDATORY**: Every note MUST have a 'pitch' field - NO EXCEPTIONS!\n"
            + f"- **CONSEQUENCE**: Missing 'pitch' fields will default to C4 (MIDI 60)!\n"
        )
        # Retry wrapper for Stage-2 similar to Stage-1
        def _call_with_rotation_comp(prompt_text: str) -> dict | None:
            nonlocal model
            max_attempts = max(5, len(API_KEYS))
            attempts = 0
            while attempts < max_attempts:
                try:
                    resp_local = model.generate_content(prompt_text)
                    raw_local = _extract_text_from_response(resp_local) or ""
                    cleaned_local = (raw_local.strip().replace("```json", "").replace("```", ""))
                    payload_slice = _extract_json_object(cleaned_local)
                    obj_local = json.loads(payload_slice or cleaned_local)
                    # Minimal hard validation: only retry on fatal/empty
                    if isinstance(obj_local, dict):
                        notes2 = obj_local.get('notes'); tokens2 = obj_local.get('tokens')
                        # Hard prune: if excessive length, trim to max allowed (avoid retries)
                        MAX_NOTES = 256
                        if isinstance(notes2, list) and len(notes2) > MAX_NOTES:
                            obj_local['notes'] = notes2[:MAX_NOTES]
                        if isinstance(tokens2, list) and len(tokens2) > MAX_NOTES:
                            obj_local['tokens'] = tokens2[:MAX_NOTES]
                        # Accept only when consistent: if tokens exist, notes must exist
                        if isinstance(tokens2, list) and len(tokens2) >= 1:
                            if isinstance(notes2, list) and len(notes2) >= 1:
                                return obj_local
                            raise ValueError('tokens present but notes empty')
                        # If no tokens, accept when notes exist
                        if isinstance(notes2, list) and len(notes2) >= 1:
                            return obj_local
                        if (isinstance(obj_local.get('notes'), list) and len(obj_local.get('notes')) == 0) and (isinstance(obj_local.get('tokens'), list) and len(obj_local.get('tokens')) == 0):
                            # Accept empty only when step-1 signaled intentional silence; we don't have that here, so retry once
                            raise ValueError('empty stage-2 output')
                    raise ValueError('fatal stage-2 parse')
                except Exception as e:
                    try:
                        print(Fore.YELLOW + f"[Stage-2 Retry {attempts}] Parse/compose error: {str(e)[:140]}" + Style.RESET_ALL)
                    except Exception:
                        pass
                    err = str(e).lower()
                    if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                        qtype = _classify_quota_error(err)
                        KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                        try:
                            cd = 60 if qtype not in ('per-hour','per-day') else 3600
                            KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        except Exception:
                            pass
                        n = len(API_KEYS)
                        rotated = False
                        stride = max(1, int(globals().get('KEY_ROTATION_STRIDE', 1)))
                        for off in range(1, n+1):
                            idx = (CURRENT_KEY_INDEX + off*stride) % n
                            if time.time() < KEY_COOLDOWN_UNTIL.get(idx, 0):
                                continue
                            try:
                                globals()['CURRENT_KEY_INDEX'] = idx
                                genai.configure(api_key=API_KEYS[idx])
                                model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                                rotated = True
                                break
                            except Exception:
                                continue
                        if not rotated:
                            wait_s = 3
                            if qtype in ('per-hour','per-day'):
                                wait_s = 60 if qtype == 'per-hour' else 300
                            _interruptible_backoff(wait_s, config, context_label="Lyrics Stage-2 cooldown")
                            continue
                        else:
                            continue
                    attempts += 1
                    time.sleep(1)
                    continue
            return None

        # Schema validator for Stage-2
        def _validate_stage2_obj(o: Dict) -> tuple[bool, str]:
            try:
                if not isinstance(o, dict):
                    return False, "not a JSON object"
                allowed = {"notes", "tokens"}
                extra = [k for k in o.keys() if k not in allowed]
                if extra:
                    return False, f"unexpected keys: {extra[:4]}"
                notes = o.get('notes'); tokens = o.get('tokens')
                if not isinstance(notes, list) or not isinstance(tokens, list):
                    return False, "notes/tokens must be arrays"
                # Relaxed validation: allow empty notes/tokens for silence sections
                if is_outro and len(notes) == 0 and len(tokens) == 0:
                    # Only require content if it's not intentionally silent
                    pass
                # Relaxed per-note check - only validate if notes exist
                if len(notes) > 0:
                    for n in notes[:32]:  # Check fewer notes
                        if not isinstance(n, dict):
                            return False, "note item not object"
                        if not {"start_beat", "duration_beats", "pitch"}.issubset(set(n.keys())):
                            return False, "missing note keys"
                        # Duration guard (soft): reject only if absurdly short
                        dur = float(n.get('duration_beats', 0))
                        if dur < 0.5:
                            return False, f"note duration {dur} < 0.5 beats (REJECTED)"
                # Relaxed token check - only validate if tokens exist
                if len(tokens) > 0:
                    for t in tokens[:64]:  # Check fewer tokens
                        if not isinstance(t, str):
                            return False, "token not string"
                return True, "ok"
            except Exception as e:
                return False, f"exception: {e}"

        # Robust call sequence with internal repair prompts on schema violations
        obj = _call_with_rotation_comp(prompt)
        ok, reason = _validate_stage2_obj(obj or {})
        if ok:
            # Post-validation: enforce pitch variety & token-note mapping where appropriate
            try:
                role_for_var = role_norm
                notes_chk = obj.get('notes') if isinstance(obj, dict) else []
                tokens_chk = obj.get('tokens') if isinstance(obj, dict) else []
                pitches = [int(n.get('pitch', 60)) for n in (notes_chk or []) if isinstance(n, dict) and 'pitch' in n]
                distinct = len(set(pitches)) if pitches else 0
                # Longest consecutive same-pitch run
                longest_run = 0
                cur_run = 0
                prev_p = None
                for p in pitches:
                    if p == prev_p:
                        cur_run += 1
                    else:
                        cur_run = 1
                        prev_p = p
                    if cur_run > longest_run:
                        longest_run = cur_run
                # Role-based thresholds
                monotone_roles = {"spoken","whisper","hum","vowels","vocal_fx","chant","tag"}
                need_variety = role_for_var not in monotone_roles
                max_run_allowed = 2 if (role_for_var == 'chorus') else 3
                # Root/tess center proximity check
                center_ref = hint_tess if isinstance(hint_tess, int) else root_note
                center_off_ok = True
                try:
                    if pitches:
                        med = sorted(pitches)[len(pitches)//2]
                        center_off_ok = (abs(int(med) - int(center_ref)) <= max(4, int(hint_span or 5)))
                except Exception:
                    center_off_ok = True
                variety_ok = True
                if need_variety and distinct < 2:
                    variety_ok = False
                if need_variety and longest_run > max_run_allowed:
                    variety_ok = False
                # discourage pathological all-60 when center/root suggests otherwise
                if pitches and all(p == 60 for p in pitches) and int(center_ref) != 60:
                    variety_ok = False
                # Token-note mapping sanity: penalize huge mismatch (prevents '-' floods at export)
                if isinstance(tokens_chk, list) and isinstance(notes_chk, list) and len(tokens_chk) >= 1 and len(notes_chk) >= 1:
                    ratio = len(notes_chk) / max(1, len(tokens_chk))
                    if ratio > 1.8 and role_for_var not in {"vowels","vocal_fx","hum","chant"}:
                        variety_ok = False
                if variety_ok:
                    return obj
                # Attempt up to 2 guided repairs to increase pitch variety
                guide = "\nVARIETY/MAPPING FIX (hard): Increase pitch variety for role='" + str(role_for_var) + "'. " \
                        + (f"Center around tessitura_center={center_ref}; " if isinstance(center_ref, (int,float)) else "") \
                        + "prefer chord tones of CONTEXT TRACKS; keep tokens/durations similar; avoid large leaps; maintain mapping order. " \
                        + (f"Allow at most {max_run_allowed} identical consecutive content pitches." if need_variety else "Use 1–2 pitch cells near center, not constant 60.")
                # If mapping mismatch is high, also request light merges to reduce note count toward token count.
                if isinstance(tokens_chk, list) and isinstance(notes_chk, list) and len(tokens_chk) >= 1 and len(notes_chk) >= 1:
                    ratio = len(notes_chk) / max(1, len(tokens_chk))
                    if ratio > 1.8:
                        guide += " Reduce number of notes by merging adjacent short notes on same pitch into longer sustains so that note count approaches token count."
                for _ in range(2):
                    obj_fix = _call_with_rotation_comp(prompt + guide)
                    ok_fix, _ = _validate_stage2_obj(obj_fix or {})
                    if not ok_fix:
                        continue
                    notes_fix = obj_fix.get('notes') if isinstance(obj_fix, dict) else []
                    pitches_fix = [int(n.get('pitch', 60)) for n in (notes_fix or []) if isinstance(n, dict) and 'pitch' in n]
                    distinct_fix = len(set(pitches_fix)) if pitches_fix else 0
                    # recompute longest run
                    longest_run_fix = 0
                    cur_run = 0
                    prev_p = None
                    for p in pitches_fix:
                        if p == prev_p:
                            cur_run += 1
                        else:
                            cur_run = 1
                            prev_p = p
                        if cur_run > longest_run_fix:
                            longest_run_fix = cur_run
                    variety_ok2 = True
                    if need_variety and distinct_fix < 2:
                        variety_ok2 = False
                    if need_variety and longest_run_fix > max_run_allowed:
                        variety_ok2 = False
                    if pitches_fix and all(p == 60 for p in pitches_fix) and int(center_ref) != 60:
                        variety_ok2 = False
                    if variety_ok2:
                        return obj_fix
            except Exception:
                pass
            return obj
        try:
            print(Fore.YELLOW + f"[Stage-2] Invalid schema: {reason}" + Style.RESET_ALL)
        except Exception:
            pass
        # Up to 2 repair attempts with explicit correction instruction
        for fix_try in range(2):
            repair_suffix = (
                "\nSCHEMA FIX (hard): Your previous output was invalid (" + str(reason) +
                "). Return ONLY a JSON object with exactly two keys: 'notes' and 'tokens'. Do NOT include any other keys."
            )
            obj_fix = _call_with_rotation_comp(prompt + repair_suffix)
            ok2, reason2 = _validate_stage2_obj(obj_fix or {})
            if ok2:
                return obj_fix
            reason = reason2
        return {}
    except Exception:
        return {}
# --- AI-assisted note adjustment suggestions + application (conservative) ---
def _propose_lyric_note_adjustments(config: Dict, genre: str, inspiration: str, track_name: str, bpm: float | int, ts: Dict, notes: List[Dict], tokens: List[str], section_label: str, section_description: str, context_tracks: List[Dict] | None = None, cfg: Dict | None = None) -> Dict:
    """
    Asks the LLM to propose merges and extensions to better fit words to notes.
    Schema: { "merge_spans": [[i,j], ...], "extend": { "k": delta_beats } }
    Returns empty dict on failure.
    """
    try:
        import google.generativeai as genai_local
    except Exception:
        return {}
    try:
        # Force reload config to get latest model_name setting
        try:
            fresh_config = load_config(CONFIG_FILE)
            config = fresh_config
        except Exception as e:
            pass
        
        # Use session override if available, otherwise config (no flash fallback)
        global SESSION_MODEL_OVERRIDE
        model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
        if REQUESTED_SWITCH_MODEL:
            model_name = REQUESTED_SWITCH_MODEL
        if not model_name:
            model_name = config.get("model_name", "gemini-2.5-pro")
        generation_config = {"response_mime_type": "application/json", "temperature": float(config.get("lyrics_temperature", config.get("temperature", 0.6)))}
        model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
        # Compute simple stress flags (strong on bar starts and mid-beat in even meters)
        try:
            bpb = max(1, int(ts.get('beats_per_bar', 4)))
        except Exception:
            bpb = 4
        preview = []
        ordered = sorted(notes, key=lambda x: float(x.get("start_beat", 0.0)))
        for idx, n in enumerate(ordered):
            s = float(n.get("start_beat", 0.0))
            d = max(0.0, float(n.get("duration_beats", 0.0)))
            p = int(n.get("pitch", 60))
            pos = s % bpb
            stress = 1 if (abs(pos-0) < 1e-3 or (bpb % 2 == 0 and abs(pos-(bpb/2)) < 1e-3)) else 0
            preview.append({
                "i": idx,
                "start": round(s, 3),
                "dur": round(d, 3),
                "pitch": p,
                "stress": stress,
                "token": tokens[idx] if idx < len(tokens) else ""
            })
        # Build compact context summary of other tracks (accents/roles/register)
        ctx_summary = []
        try:
            bpb = max(1, int(ts.get('beats_per_bar', 4)))
            for t in (context_tracks or []):
                try:
                    nm = get_instrument_name(t)
                    rl = t.get('role', 'complementary')
                    nlist = sorted(t.get('notes', []), key=lambda x: float(x.get('start_beat', 0.0)))
                    if not nlist:
                        ctx_summary.append({"name": nm, "role": rl, "density": 0.0, "accents": []})
                        continue
                    starts = [float(n.get('start_beat', 0.0)) for n in nlist]
                    dur_sum = sum(max(0.0, float(n.get('duration_beats', 0.0))) for n in nlist)
                    total_beats = (max(starts) - min(starts)) + (max(0.0, float(nlist[-1].get('duration_beats', 0.0))) if nlist else 0.0)
                    density = (len(nlist) / max(1e-6, total_beats/bpb)) if total_beats > 0 else len(nlist)
                    # Pick top accent onsets (first 16) modulo bar
                    accents = []
                    for s in starts[:64]:
                        accents.append(round(s % bpb, 3))
                    ctx_summary.append({"name": nm, "role": rl, "density": round(density, 2), "accents": accents})
                except Exception:
                    continue
        except Exception:
            ctx_summary = []
        # Optional rule set from config
        extra_rules: List[str] = []
        try:
            if isinstance(config.get('note_adjustment_rules'), list):
                extra_rules = [str(r) for r in config.get('note_adjustment_rules') if isinstance(r, (str, int, float))]
        except Exception:
            extra_rules = []
        # Mode & bounds (expressive allows more freedom)
        try:
            mode = str(config.get('note_adjustment_mode', 'freeform')).lower()
        except Exception:
            mode = 'freeform'
        if mode == 'conservative':
            shift_bound, repitch_bound, extend_bound = 0.25, 2, 0.5
        elif mode == 'expressive':
            shift_bound, repitch_bound, extend_bound = 0.5, 4, 1.0
        else:  # freeform
            shift_bound, repitch_bound, extend_bound = 1.0, 12, 2.0
        section_role = _normalize_section_role(section_label)
        base_rules = [
            "Only contiguous indices in merge_spans (i<=j).",
            "Prefer merging micro-notes within the SAME word (token=='' means continuation).",
            (f"shift[k] in beats within ±{shift_bound} (timing adjustments allowed; preserve order; clamp to avoid overlap)." if mode in ('expressive','freeform') else "Do NOT change start times; preserve bar structure."),
            (f"repitch[k] in semitones within ±{min(4, int(repitch_bound))} (prefer nearest scale tones; avoid leaps). Prioritize repitch only on '-' continuation notes; avoid repitch on first content syllables." if mode in ('expressive','freeform') else "Avoid repitch; keep pitch unless merging equal-pitch spans."),
            f"extend[k] is small duration delta in beats within ±{extend_bound} (clamped by next start, never overlapping).",
            "No gaps inside a single word: continuation notes must start exactly at previous note's end.",
            "If a micro-gap (≤1/16 beat) occurs between syllables, extend the previous note to eliminate the gap (legato).",
            "Never merge across word boundaries: merge only if the first note holds a content token and all following notes in the span are continuation ('').",
            "Preserve the count of content tokens (non-continuations); do not collapse two different syllables/words into one note.",
            "Do not insert rests; preserve continuous vocal flow unless at phrase ends.",
            "Merges only if all pitches in span are equal AND each note is very short (≈≤1/16 beat); never merge on stressed onsets.",
            "Align content words to strong beats or drum accents when helpful; sustain vowels on longer notes.",
            "Phrase ends on rests or bar ends when feasible; avoid consonant clusters on very short notes.",
            "Respect key/scale and natural vocal range; avoid extreme micro-fragments.",
            "Do not introduce '+' tokens or hyphenation; continuation is indicated by '-' on continuation notes only.",
            "Breaths only on rest notes; never convert a content note into a rest.",
            ("Allow modest repetition for HOOK/CHORUS; keep verses more varied." if section_role == 'chorus' else "")
        ]
        # Hook contiguity and preservation (if provided by earlier step)
        try:
            part_key = f"{track_name}|{section_label or ''}"
            hook_info = LYRICS_PART_META.get(part_key) or {}
            hook_ranges = hook_info.get('hook_token_ranges') if isinstance(hook_info.get('hook_token_ranges'), list) else None
            hook_can = hook_info.get('hook_canonical') if isinstance(hook_info.get('hook_canonical'), str) else None
            self_check = hook_info.get('self_check') if isinstance(hook_info.get('self_check'), dict) else None
            phrase_windows = hook_info.get('phrase_windows') if isinstance(hook_info.get('phrase_windows'), list) else None
            vowel_targets = hook_info.get('vowel_sustain_targets') if isinstance(hook_info.get('vowel_sustain_targets'), list) else None
        except Exception:
            hook_ranges = None; hook_can = None; self_check = None; phrase_windows = None; vowel_targets = None
        if section_role == 'chorus':
            base_rules.append("If a hook_canonical exists, keep its mapped token span contiguous; do not insert rests inside; prefer extends over gaps.")
            base_rules.append("Do not repitch or shift the first content note of the hook; apply only minimal timing extensions to maintain legato.")
        # If words-stage flagged infeasible mapping and requested note rewrite, allow broader timing edits (still bounded)
        try:
            if isinstance(self_check, dict):
                pass
            rewrite_req = bool((LYRICS_PART_META.get(f"{track_name}|{section_label or ''}") or {}).get('note_rewrite_request'))
            rewrite_intent = (LYRICS_PART_META.get(f"{track_name}|{section_label or ''}") or {}).get('note_rewrite_intent')
        except Exception:
            rewrite_req = False; rewrite_intent = None
        if rewrite_req:
            base_rules.append("Note rewrite requested by words-stage: you may propose more liberal extend/merge within phrase_windows to realize smoother phrasing; avoid disrupting non-target tracks.")
            if isinstance(rewrite_intent, str) and rewrite_intent.strip():
                base_rules.append("Rewrite intent: " + rewrite_intent.strip())
        # Legato guidance from phrase windows and vowel sustain targets
        if isinstance(phrase_windows, list) and phrase_windows:
            base_rules.append("Within phrase_windows, prefer extends over shifts; minimize micro-gaps; avoid creating rests.")
        if isinstance(vowel_targets, list) and vowel_targets:
            base_rules.append("For vowel_sustain_targets, prefer extending the corresponding notes slightly (within bounds) rather than breaking into multiple short notes.")
        rules_text = "\n- " + "\n- ".join([r for r in base_rules + extra_rules if r])
        prompt = (
            "You are a vocal arranger. Propose minimal note edits to fit words naturally and musically.\n"
            f"Global: Genre={genre}; Key/Scale={cfg.get('key_scale','') if cfg else config.get('key_scale','?')}; BPM={round(float(bpm))}; TimeSig={ts.get('beats_per_bar','?')}/{ts.get('beat_value','?')}. Track={track_name}.\n"
            + (f"Section: {section_label}. {section_description}\n" if section_label else "")
            + ("Context tracks (name, role, density, accentsWithinBar): " + json.dumps(ctx_summary) + "\n" if ctx_summary else "")
            + "Notes (order, full for this part): {i,start,dur,pitch,stress,token}.\n" + json.dumps(preview) + "\n\n"
            + "Output STRICT JSON only:\n{\n  \"merge_spans\": [[int,int],...],\n  \"extend\": { \"int\": number },\n  \"shift\": { \"int\": number },\n  \"repitch\": { \"int\": number }\n}\n\n"
            + ("Rules:" + rules_text + "\n")
            + ("Hook (context): hook_canonical=\"" + hook_can + "\", token_ranges=" + json.dumps(hook_ranges) + "\n" if section_role == 'chorus' and hook_can else "")
            + ("Phrase windows (context): " + json.dumps(phrase_windows) + "\n" if isinstance(phrase_windows, list) else "")
            + ("Vowel sustain targets (context): " + json.dumps(vowel_targets) + "\n" if isinstance(vowel_targets, list) else "")
            + ("Diagnostics (from words-stage self_check): " + json.dumps(self_check) + "\n" if isinstance(self_check, dict) else "")
            + ("Adjustment guidance (soft):\n"
               "- If stress_alignment_score is low, prefer micro timing shifts (shift within bounds) or small extends over merges; avoid changing primary content syllables.\n"
               "- If repetition_ngram_overlap is high, avoid creating extra echoes via timing; only consolidate where it reduces redundant repeats outside chorus.\n"
               "- If nonsense_ratio is high, avoid turning content notes into rests; keep semantic flow outside chant_segments.\n"
               "- Maintain hook exact wording and contiguity; avoid new rests within hook; apply extends to bridge micro-gaps; repitch only on '-' continuation notes if musically needed.\n"
              )
            + "SynthV alignment guidance (apply if consistent with the music):\n- Avoid gaps within a word; continuation notes should be adjacent (legato).\n- Ensure sufficient space between phrases; short rests are fine between phrases.\n- Do not stack overlapping notes; one voice per track.\n- If a long word spans many notes, prefer longer spans or consider fewer notes (merges) rather than many micro-slices.\n"
        )
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", "") or ""
        cleaned = raw.strip().replace("```json", "").replace("```", "")
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            merges = obj.get("merge_spans", [])
            extends = obj.get("extend", {})
            shifts = obj.get("shift", {})
            repitches = obj.get("repitch", {})
            if not isinstance(merges, list) or not isinstance(extends, dict):
                return {}
            # basic validation
            valid_merges = []
            for span in merges:
                if isinstance(span, list) and len(span) == 2 and all(isinstance(x, (int, float)) for x in span):
                    i, j = int(span[0]), int(span[1])
                    if 0 <= i <= j < len(notes):
                        valid_merges.append([i, j])
            valid_extends = {}
            for k, v in extends.items():
                try:
                    ki = int(k)
                    dv = float(v)
                    if 0 <= ki < len(notes):
                        valid_extends[str(ki)] = dv
                except Exception:
                    continue
            valid_shifts = {}
            for k, v in (shifts or {}).items():
                try:
                    ki = int(k)
                    dv = float(v)
                    if 0 <= ki < len(notes):
                        dv = max(-shift_bound*1.5, min(dv, shift_bound*1.5))
                        valid_shifts[str(ki)] = dv
                except Exception:
                    continue
            valid_repitches = {}
            for k, v in (repitches or {}).items():
                try:
                    ki = int(k)
                    dv = float(v)
                    if 0 <= ki < len(notes):
                        # Allow small repitch only on continuation notes '-'
                        tok = ""
                        try:
                            tok = preview[ki].get("token", "") if isinstance(preview, list) and ki < len(preview) and isinstance(preview[ki], dict) else ""
                        except Exception:
                            tok = ""
                        if isinstance(tok, str) and tok.strip() == '-':
                            max_rp = float(min(4, int(repitch_bound)))
                            dv = max(-max_rp, min(dv, max_rp))
                            valid_repitches[str(ki)] = dv
                        else:
                            # Skip repitch on primary content syllables
                            continue
                except Exception:
                    continue
            return {"merge_spans": valid_merges, "extend": valid_extends, "shift": valid_shifts, "repitch": valid_repitches}
        return {}
    except Exception:
        return {}

# --- Stage-0: Analyze user prompt into structured directives ---
def _analyze_user_prompt(config: Dict, genre: str, inspiration: str, user_prompt: str | None) -> Dict:
    try:
        import google.generativeai as genai_local
    except Exception:
        return {}
    try:
        # Use session override if available, otherwise config (no flash fallback)
        global SESSION_MODEL_OVERRIDE
        model_name = SESSION_MODEL_OVERRIDE or config.get("model_name") or config.get("model")
        try:
            if REQUESTED_SWITCH_MODEL:
                model_name = REQUESTED_SWITCH_MODEL
        except Exception:
            pass
        if not model_name:
            model_name = config.get("model_name", "gemini-2.5-pro")
        generation_config = {"response_mime_type": "application/json", "temperature": float(config.get("plan_temperature", config.get("lyrics_temperature", config.get("temperature", 0.4))))}
        try:
            if isinstance(config.get("max_output_tokens"), int):
                _mx = int(config.get("max_output_tokens")); _mx = max(256, min(_mx, 8192)); generation_config["max_output_tokens"] = _mx
        except Exception:
            pass
        model = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        prompt = (
            "ROLE: You analyze a free-form lyric/style prompt and extract compact directives for a vocal generation pipeline.\n\n"
            f"CONTEXT: Genre={genre}; Inspiration={inspiration}.\n\n"
            + ("USER PROMPT:\n" + user_prompt.strip() + "\n\n" if isinstance(user_prompt, str) and user_prompt.strip() else "USER PROMPT:\n(EMPTY)\n\n")
            + "OUTPUT (STRICT JSON):\n{\n"
            + "  \"hook_canonical\": string|null,\n"
            + "  \"chorus_lines\": [string,...]|null,\n"
            + "  \"title_suggestions\": [string,...]|null,\n"
            + "  \"style_tags\": [string,...]|null,\n"
            + "  \"language\": string|null,\n"
            + "  \"code_switching_ratio\": number|null,\n"
            + "  \"pov\": string|null,\n"
            + "  \"tense\": string|null,\n"
            + "  \"narrator_persona\": string|null,\n"
            + "  \"rhyme_scheme\": string|null,\n"
            + "  \"internal_rhyme\": boolean|null,\n"
            + "  \"assonance_level\": string|null,\n"
            + "  \"line_length_target\": number|null,\n"
            + "  \"stress_policy\": string|null,\n"
            + "  \"repetition_policy\": {\"chorus\": string|null, \"verse\": string|null}|null,\n"
            + "  \"density_targets\": {\"wpb_hint\": number|null, \"min_word_beats_hint\": number|null, \"melisma_bias_hint\": number|null}|null,\n"
            + "  \"motif_len_bars\": number|null,\n"
            + "  \"phrase_windows\": [[number,number],...]|null,\n"
            + "  \"cadence_targets\": [number,...]|null,\n"
            + "  \"tessitura_center\": number|null,\n"
            + "  \"range_limit\": number|null,\n"
            + "  \"articulation\": string|null,\n"
            + "  \"vowel_pref\": [string,...]|null,\n"
            + "  \"lexicon_prefer\": [string,...]|null,\n"
            + "  \"lexicon_avoid\": [string,...]|null,\n"
            + "  \"nonsense_budget\": number|null,\n"
            + "  \"chant_syllables\": [string,...]|null,\n"
            + "  \"chant_segments\": [[number,number],...]|null,\n"
            + "  \"drop_style\": string|null,\n"
            + "  \"rhythm_grid\": string|null,\n"
            + "  \"onset_rate_preference\": string|null,\n"
            + "  \"section_goals\": object|null,\n"
            + "  \"notes\": string|null\n"
            + "}\n\n"
            + "RULES:\n- Consider the genre context: hook-heavy genres (pop, rock, hip-hop, dance, R&B, country) typically benefit from catchy hooks, while instrumental genres (ambient, minimal techno, IDM, classical, jazz) often work better without them.\n- If a clear title/hook phrase is explicitly provided (quoted or not), use that exact phrase.\n- For genres that traditionally use hooks, consider generating a hook_canonical if it would enhance the song's appeal and memorability.\n- For instrumental-focused genres, only include hook_canonical if explicitly requested or if the context strongly suggests vocal content.\n- Provide 2–4 short chorus_lines if a chorus idea exists; else null.\n- Keep lists short and practical; avoid repeating the entire user prompt.\n"
        )
        # Robust generation with key rotation and retries
        max_attempts = max(3, len(API_KEYS))
        attempts = 0
        nonlocal_model = [model]
        while attempts < max_attempts:
            attempts += 1
            try:
                resp = nonlocal_model[0].generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
                raw = getattr(resp, "text", "") or ""
                try:
                    obj = json.loads(raw)
                except Exception:
                    cleaned = raw.strip().replace("```json", "").replace("```", "")
                    obj = json.loads(cleaned)
                return obj if isinstance(obj, dict) else {}
            except Exception as e:
                err = str(e).lower()
                # Quota/key rotation handling
                if ('429' in err) or ('quota' in err) or ('rate limit' in err) or ('resource exhausted' in err) or ('exceeded' in err):
                    try:
                        qtype = _classify_quota_error(err)
                        KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                        # cooldown
                        cd = 60 if qtype not in ('per-hour','per-day') else 3600
                        KEY_COOLDOWN_UNTIL[CURRENT_KEY_INDEX] = max(KEY_COOLDOWN_UNTIL.get(CURRENT_KEY_INDEX,0), time.time()+cd)
                        # rotate
                        idx = _next_available_key(CURRENT_KEY_INDEX)
                        if idx is not None and idx != CURRENT_KEY_INDEX:
                            globals()['CURRENT_KEY_INDEX'] = idx
                            genai.configure(api_key=API_KEYS[idx])
                            nonlocal_model[0] = genai_local.GenerativeModel(model_name=model_name, generation_config=generation_config)
                            continue
                    except Exception:
                        pass
                # Non-quota/transient: small backoff then retry
                time.sleep(0.5)
                continue
        return {}
    except Exception:
        return {}

def _apply_note_adjustments_conservative(notes: List[Dict], tokens: List[str], plan: Dict) -> (List[Dict], List[str]):
    """
    Applies merges and small extensions deterministically with guardrails.
    - Merge only spans where all pitches equal. New note: start of first, dur=sum, pitch same.
    - Tokens after merge: keep first token; drop merged continuation tokens.
    - Extend: clamp so end <= next start; ignore if negative. Does not shift starts.
    Returns adjusted (notes, tokens). If invalid, returns originals.
    """
    try:
        if not isinstance(plan, dict):
            return notes, tokens
        # Minimum duration to avoid micro-notes
        MIN_NOTE_BEATS = 1.0/16.0
        mspans = plan.get("merge_spans") or []
        extends = plan.get("extend") or {}
        shifts = plan.get("shift") or {}
        repitches = plan.get("repitch") or {}
        # Work on indexed copies
        idx_order = list(range(len(notes)))
        # Normalize notes sorted by start
        notes_sorted = sorted([(i, n) for i, n in enumerate(notes)], key=lambda x: float(x[1].get("start_beat", 0.0)))
        reorder = {old_i: new_i for new_i, (old_i, _) in enumerate(notes_sorted)}
        notes_arr = [n for _, n in notes_sorted]
        tokens_arr = [tokens[i] if i < len(tokens) else "" for i, _ in notes_sorted]

        # Apply merges from longest spans first to avoid index churn
        mspans_sorted = sorted([[reorder.get(i,i), reorder.get(j,j)] for i,j in mspans], key=lambda x: (x[1]-x[0]), reverse=True)
        merged_mask = [False]*len(notes_arr)
        for i,j in mspans_sorted:
            if not (0 <= i <= j < len(notes_arr)): continue
            if any(merged_mask[k] for k in range(i, j+1)): continue
            # Same pitch guardrail
            pitches = {int(notes_arr[k].get('pitch', 60)) for k in range(i, j+1)}
            if len(pitches) != 1: continue
            # Do not merge across content-token boundaries: first must be content, the rest must be continuation '-'
            first_token = tokens_arr[i] if i < len(tokens_arr) else ""
            if not isinstance(first_token, str) or first_token.strip() in ('', '-'):
                continue
            boundary_violation = False
            for k in range(i+1, j+1):
                tk = tokens_arr[k] if k < len(tokens_arr) else ""
                if not (isinstance(tk, str) and tk.strip() == '-'):
                    boundary_violation = True; break
            if boundary_violation:
                continue
            # Merge
            first = notes_arr[i]
            start = float(first.get('start_beat', 0.0))
            total = 0.0
            for k in range(i, j+1):
                total += max(0.0, float(notes_arr[k].get('duration_beats', 0.0)))
            new_note = dict(first)
            new_note['duration_beats'] = total
            # Replace range with single note
            notes_arr = notes_arr[:i] + [new_note] + notes_arr[j+1:]
            keep_token = tokens_arr[i]
            tokens_arr = tokens_arr[:i] + [keep_token] + tokens_arr[j+1:]
            merged_mask = [False]*len(notes_arr)

        # Apply extensions (guard: no overlap). Also keep intra-word legato: if next token is '-',
        # ensure current note ends exactly at next start after extension/clamp.
        for k_str, dv in extends.items():
            try:
                k = int(k_str)
                if not (0 <= k < len(notes_arr)): continue
                if abs(dv) <= 1e-6: continue
                # Clamp by next start
                start_k = float(notes_arr[k].get('start_beat', 0.0))
                dur_k = float(notes_arr[k].get('duration_beats', 0.0))
                next_start = None
                for t in notes_arr[k+1:]:
                    next_start = float(t.get('start_beat', 0.0)); break
                # Allow shrink or extend but never overlap next start
                max_dur = (next_start - start_k - 1e-6) if next_start is not None else (dur_k + max(0.0, dv))
                new_dur = max(1e-4, min(dur_k + dv, max_dur))
                if new_dur > dur_k:
                    notes_arr[k]['duration_beats'] = new_dur
                elif new_dur < dur_k:
                    notes_arr[k]['duration_beats'] = new_dur
                # Legato enforcement for continuation token
                try:
                    if (k+1) < len(notes_arr) and isinstance(tokens_arr[k+1], str) and tokens_arr[k+1] == '-' and next_start is not None:
                        # Snap end exactly to next_start
                        notes_arr[k]['duration_beats'] = max(1e-4, next_start - start_k)
                except Exception:
                    pass
            except Exception:
                continue

        # Apply micro timing shifts (preserve order and avoid overlaps). For continuation tokens '-',
        # enforce legato: next note must start at previous end to avoid intra-word gaps.
        try:
            # Build per-note target shift, then sort by index to apply safely
            for k_str, dv in (shifts or {}).items():
                try:
                    k = int(k_str)
                    delta = float(dv)
                except Exception:
                    continue
                if not (0 <= k < len(notes_arr)):
                    continue
                if abs(delta) <= 1e-6:
                    continue
                start_k = float(notes_arr[k].get('start_beat', 0.0))
                dur_k = float(notes_arr[k].get('duration_beats', 0.0))
                prev_end = None
                if k > 0:
                    prev_start = float(notes_arr[k-1].get('start_beat', 0.0))
                    prev_dur = float(notes_arr[k-1].get('duration_beats', 0.0))
                    prev_end = prev_start + prev_dur
                next_start = None
                if k+1 < len(notes_arr):
                    next_start = float(notes_arr[k+1].get('start_beat', 0.0))
                # New start clamped between previous end and next start - duration
                min_start = (prev_end + 1e-6) if prev_end is not None else (start_k - 9999)
                max_start = (next_start - dur_k - 1e-6) if next_start is not None else (start_k + 9999)
                new_start = max(min_start, min(start_k + delta, max_start))
                # If current token is continuation '-', force legato start
                try:
                    if isinstance(tokens_arr[k], str) and tokens_arr[k] == '-' and prev_end is not None:
                        new_start = prev_end
                except Exception:
                    pass
                notes_arr[k]['start_beat'] = new_start
        except Exception:
            pass

        # Apply repitches (clamped range; keep within reasonable MIDI range 36..84)
        for k_str, dv in (repitches or {}).items():
            try:
                k = int(k_str)
                delta = int(round(float(dv)))
                if not (0 <= k < len(notes_arr)):
                    continue
                base_pitch = int(notes_arr[k].get('pitch', 60))
                new_pitch = max(36, min(base_pitch + delta, 96))
                notes_arr[k]['pitch'] = new_pitch
            except Exception:
                continue

        # Reorder back to original start order (already sorted)
        # Cleanup pass to eliminate micro-notes: raise to MIN_NOTE_BEATS or merge
        cleaned_notes: List[Dict] = []
        cleaned_tokens: List[str] = []
        for i, n in enumerate(notes_arr):
            try:
                start_i = float(n.get('start_beat', 0.0))
                dur_i = float(n.get('duration_beats', 0.0))
                tok_i = tokens_arr[i] if i < len(tokens_arr) else ''
                # Determine next start for clamping
                next_start = None
                if i+1 < len(notes_arr):
                    next_start = float(notes_arr[i+1].get('start_beat', 0.0))
                # If duration already OK, keep as-is
                if dur_i >= MIN_NOTE_BEATS:
                    cleaned_notes.append(n)
                    cleaned_tokens.append(tok_i)
                    continue
                # If continuation '-', prefer merging into previous content block
                if isinstance(tok_i, str) and tok_i.strip() == '-' and cleaned_notes:
                    # Extend previous note by this tiny duration (legato)
                    prev = cleaned_notes[-1]
                    prev['duration_beats'] = float(prev.get('duration_beats', 0.0)) + dur_i
                    # Drop this micro continuation note
                    continue
                # Try to raise to MIN_NOTE_BEATS without overlap
                if next_start is not None:
                    max_dur = max(0.0, next_start - start_i - 1e-6)
                else:
                    max_dur = dur_i
                target = min(max_dur, max(MIN_NOTE_BEATS, dur_i))
                if target > 0.0:
                    n2 = dict(n)
                    n2['duration_beats'] = target
                    cleaned_notes.append(n2)
                    cleaned_tokens.append(tok_i)
                else:
                    # No room ahead; if possible, attach to previous note
                    if cleaned_notes:
                        prev = cleaned_notes[-1]
                        prev['duration_beats'] = float(prev.get('duration_beats', 0.0)) + dur_i
                        # Drop current
                        continue
                    # As last resort, keep as-is (should be rare)
                    cleaned_notes.append(n)
                    cleaned_tokens.append(tok_i)
            except Exception:
                cleaned_notes.append(n)
                cleaned_tokens.append(tokens_arr[i] if i < len(tokens_arr) else '')

        return cleaned_notes, cleaned_tokens
    except Exception:
        return notes, tokens
# --- OpenUtau UST Export (helper) ---
def _export_openutau_ust_corrected(themes: List[Dict], track_index: int, syllables_per_theme: List[List[str]], ts: Dict, bpm: float | int, output_path: str, section_length_bars: int | float | None = None) -> bool:
    """
    Write a corrected UST file for OpenUtau with proper formatting.
    Includes timing validation to ensure MIDI-UST synchronization.
    """
    try:
        ticks_per_beat = int(TICKS_PER_BEAT)
        lines = []
        
        # Timing validation data
        timing_debug = {
            "total_notes": 0,
            "total_beats": 0.0,
            "rounding_errors": [],
            "part_offsets": []
        }
        
        # UST Header
        lines.append("[#VERSION]")
        lines.append("UST Version=1.20")
        lines.append("[#SETTING]")
        lines.append(f"Tempo={float(bpm):.2f}")
        lines.append("Tracks=1")
        lines.append("ProjectName=LyricsExport")
        lines.append("Mode2=True")
        lines.append(f"TimeBase={ticks_per_beat}")
        lines.append("VoiceDir=")
        lines.append("CacheDir=")
        lines.append("Flags=")
        
        beats_per_bar = float(ts.get('beats_per_bar', 4.0))
        section_len_beats = float(section_length_bars) * beats_per_bar if section_length_bars else 8.0 * beats_per_bar
        # Guardrails to prevent micro-notes and tiny gaps
        MIN_CONTENT_BEATS = 0.6  # align with legacy exporter for singable phrasing
        SMALL_GAP_BEATS = 1.0 / 4.0
        min_content_ticks = int(round(MIN_CONTENT_BEATS * ticks_per_beat))
        small_gap_ticks = int(round(SMALL_GAP_BEATS * ticks_per_beat))
        MICRO_NOTE_BEATS = 1.0 / 16.0
        micro_note_ticks = int(round(MICRO_NOTE_BEATS * ticks_per_beat))
        
        note_blocks = []
        current_beat = 0.0
        
        # Process each theme
        target_track_name = None
        try:
            if themes and len(themes) > 0:
                tr0 = (themes[0] or {}).get('tracks', [])
                if 0 <= track_index < len(tr0):
                    t0 = tr0[track_index]
                    target_track_name = (t0.get('instrument_name') or t0.get('name') or '').strip()
        except Exception:
            target_track_name = None
        for part_idx, theme in enumerate(themes):
            tracks = theme.get('tracks', [])
            use_index = track_index
            # DEBUG: Print track info (set DEBUG_EXPORT=True at top of file to enable)
            DEBUG_EXPORT = os.environ.get('DEBUG_EXPORT', 'false').lower() == 'true'
            if DEBUG_EXPORT:
                print(f"[DEBUG UST] Part {part_idx}: tracks={len(tracks)}, use_index={use_index}")
            # Resolve by name per part to avoid index drift across parts
            if not (0 <= use_index < len(tracks)) and target_track_name:
                for ix, tr in enumerate(tracks):
                    nm = (tr.get('instrument_name') or tr.get('name') or '').strip()
                    if nm and nm == target_track_name:
                        use_index = ix
                        break
            if 0 <= use_index < len(tracks):
                track = tracks[use_index]
                notes = sorted(track.get('notes', []), key=lambda n: float(n.get('start_beat', 0.0)))
                syllables = syllables_per_theme[part_idx] if part_idx < len(syllables_per_theme) else []
                # DEBUG: Print notes and syllables info
                if DEBUG_EXPORT:
                    print(f"[DEBUG UST] Part {part_idx}: notes={len(notes)}, syllables={len(syllables)}")
                # Ensure 1:1 mapping: pad tokens to note count so content lands in UST
                try:
                    if len(syllables) < len(notes):
                        deficit = len(notes) - len(syllables)
                        if len(syllables) == 0:
                            # Start with a content vowel, then continuations
                            syllables = ['ah'] + ['-'] * max(0, deficit - 1)
                        else:
                            syllables = list(syllables) + ['-'] * deficit
                    elif len(syllables) > len(notes):
                        syllables = list(syllables)[:len(notes)]
                except Exception:
                    pass
                
                part_start = part_idx * section_len_beats
                timing_debug["part_offsets"].append({
                    "part_idx": part_idx,
                    "part_start": part_start,
                    "section_len_beats": section_len_beats
                })
                
                # Detect if note positions are already absolute (include part offset)
                is_absolute = False
                try:
                    if notes:
                        min_local_start = float(notes[0].get('start_beat', 0.0))
                        # If min start is already at or beyond part_start, treat as absolute
                        is_absolute = (min_local_start >= (part_start - 0.01))
                except Exception:
                    is_absolute = False
                
                prev_was_content = False
                for note_idx, note in enumerate(notes):
                    local_start = float(note.get('start_beat', 0.0))
                    start_beat = local_start if is_absolute else (local_start + part_start)
                    duration_beats = float(note.get('duration_beats', 1.0))
                    pitch = int(note.get('pitch', 60))
                    
                    # Get syllable - handle silence roles with minimal atmospheric content
                    syllable = syllables[note_idx] if note_idx < len(syllables) else "R"
                    if not syllable or syllable.strip() == "":
                        # For silence roles, use minimal atmospheric content instead of "R"
                        if note.get('silence_role', False):
                            syllable = "ah"  # Minimal atmospheric content
                        else:
                            syllable = "R"
                    
                    # Convert to ticks with precision tracking
                    exact_ticks = duration_beats * ticks_per_beat
                    length_ticks = int(round(exact_ticks))
                    
                    # Track rounding errors for debugging
                    rounding_error = abs(exact_ticks - length_ticks)
                    if rounding_error > 0.1:  # Significant rounding error
                        timing_debug["rounding_errors"].append({
                            "note_idx": note_idx,
                            "exact_ticks": exact_ticks,
                            "rounded_ticks": length_ticks,
                            "error": rounding_error,
                            "duration_beats": duration_beats
                        })
                    
                    # Add rest if needed (merge tiny gaps into previous note)
                    if start_beat > current_beat + 0.01:
                        rest_ticks = int(round((start_beat - current_beat) * ticks_per_beat))
                        if rest_ticks > 0:
                            if note_blocks and rest_ticks <= small_gap_ticks:
                                # attach tiny gap to previous block
                                note_blocks[-1]["Length"] = int(note_blocks[-1]["Length"] + rest_ticks)
                            else:
                                if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                    note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                                else:
                                    note_blocks.append({
                                        "Lyric": "R",
                                        "NoteNum": 60,
                                        "Length": rest_ticks
                                    })
                        current_beat = start_beat
                    
                    tok = (syllable or '').strip()

                    # Handle explicit rests / breaths tokens
                    if tok.lower() in ("r", "silence", "[br]", "br", "breath") or tok == "":
                        if length_ticks > 0:
                            if note_blocks and length_ticks <= small_gap_ticks:
                                note_blocks[-1]["Length"] = int(note_blocks[-1]["Length"] + length_ticks)
                            else:
                                if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                    note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + length_ticks)
                                else:
                                    note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": max(1, length_ticks)})
                        prev_was_content = False
                        current_beat = start_beat + duration_beats
                        
                        # Track timing data and continue
                        timing_debug["total_notes"] += 1
                        timing_debug["total_beats"] = max(timing_debug["total_beats"], current_beat)
                        continue

                    # Handle continuation '-' by extending previous content
                    if tok == '-':
                        if note_blocks and note_blocks[-1].get('Lyric') not in ('R', '-'):
                            prev_pitch = int(note_blocks[-1].get('NoteNum', pitch))
                            # Merge only if micro-duration OR same pitch as previous
                            if (length_ticks <= micro_note_ticks) or (int(pitch) == prev_pitch):
                                extend_ticks = max(1, length_ticks)
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + extend_ticks)
                            else:
                                # Keep as its own continuation note (different pitch and not micro)
                                note_blocks.append({
                                    "Lyric": '-',
                                    "NoteNum": pitch,
                                    "Length": max(1, length_ticks)
                                })
                        else:
                            # No previous content to extend → treat micro as gap, else explicit '-' note
                            if length_ticks > 0:
                                if note_blocks and length_ticks <= small_gap_ticks:
                                    note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + length_ticks)
                                else:
                                    note_blocks.append({"Lyric": '-', "NoteNum": pitch, "Length": max(1, length_ticks)})
                        prev_was_content = False
                        current_beat = start_beat + duration_beats
                        timing_debug["total_notes"] += 1
                        timing_debug["total_beats"] = max(timing_debug["total_beats"], current_beat)
                        continue

                    # Content token: enforce minimum length or convert to hold if too short
                    if length_ticks < min_content_ticks and note_blocks and note_blocks[-1].get('Lyric') not in ('R', '-'):
                        # Too short for a fresh content onset → hold previous syllable
                        note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + max(1, length_ticks))
                        prev_was_content = True
                    else:
                        # New content note with minimum singable length
                        safe_len = max(min_content_ticks, max(1, length_ticks))
                        note_blocks.append({
                            "Lyric": tok,
                            "NoteNum": pitch,
                            "Length": safe_len
                        })
                        prev_was_content = True
                    
                    # Advance timeline using effective tick length to preserve spacing
                    current_beat = start_beat + (max(1, length_ticks) / float(ticks_per_beat))
                    
                    # Track timing data
                    timing_debug["total_notes"] += 1
                    timing_debug["total_beats"] = max(timing_debug["total_beats"], current_beat)
            else:
                # If the target track is not present for this part, do not insert artificial rests.
                # Just advance the timeline reference; keep file compact and avoid fake silence blocks.
                current_beat = (part_idx + 1) * section_len_beats
        
        # Write note blocks with complete UST format
        for idx, nb in enumerate(note_blocks):
            lines.append(f"[#NOTE{idx:04d}]")
            lines.append(f"Length={int(nb['Length'])}")
            lines.append(f"Lyric={nb['Lyric']}")
            lines.append(f"NoteNum={int(nb['NoteNum'])}")
            lines.append("Intensity=100")
            lines.append("Modulation=0")
            lines.append("PreUtterance=")
            lines.append("VoiceOverlap=")
            lines.append("StartPoint=")
            lines.append("Envelope=")
            lines.append("PBType=5")
            lines.append("PitchBend=")
            lines.append("PBStart=")
            lines.append("PBS=")
            lines.append("PBE=")
            lines.append("PBW=")
            lines.append("PBY=")
            lines.append("VBR=")
            lines.append("Flags=")
        
        # Add singer section
        lines.append("[#SINGER]")
        lines.append("Name=DefaultSinger")
        lines.append("VoiceDir=")
        lines.append("[#TRACKEND]")
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        # Print timing validation report
        print(Fore.CYAN + f"UST Export Timing Report:" + Style.RESET_ALL)
        print(f"  Total notes: {timing_debug['total_notes']}")
        print(f"  Total beats: {timing_debug['total_beats']:.3f}")
        print(f"  Part offsets: {timing_debug['part_offsets']}")
        
        if timing_debug["rounding_errors"]:
            print(Fore.YELLOW + f"  Rounding errors: {len(timing_debug['rounding_errors'])}" + Style.RESET_ALL)
            for error in timing_debug["rounding_errors"][:5]:  # Show first 5
                print(f"    Note {error['note_idx']}: {error['exact_ticks']:.3f} -> {error['rounded_ticks']} (error: {error['error']:.3f})")
        else:
            print(Fore.GREEN + "  No significant rounding errors" + Style.RESET_ALL)
        
        print(Fore.GREEN + f"Exported UST: {output_path}" + Style.RESET_ALL)
        return True
        
    except Exception as e:
        print(Fore.YELLOW + f"UST export failed: {e}" + Style.RESET_ALL)
        return False
def _export_openutau_ust_for_track__legacy(themes: List[Dict], track_index: int, syllables_per_theme: List[List[str]], ts: Dict, bpm: float | int, output_path: str, section_length_bars: int | float | None = None) -> bool:
    """
    Write a minimal UST file for OpenUtau for one track across all themes.
    Simplified version to avoid duplicates and ensure proper note lengths.
    """
    try:
        # Use quarter-note beats consistently for all timing; 1 beat = 1 quarter
        ticks_per_unit = int(TICKS_PER_BEAT)
        lines = []
        lines.append("[#VERSION]")
        lines.append("UST Version=1.20")
        lines.append("[#SETTING]")
        lines.append(f"Tempo={float(bpm):.2f}")
        lines.append("Tracks=1")
        lines.append("ProjectName=LyricsExport")
        lines.append("Mode2=True")
        lines.append(f"TimeBase={TICKS_PER_BEAT}")
        lines.append("VoiceDir=")
        lines.append("CacheDir=")
        lines.append("Flags=")

        # Internal: sanitize lyric token for UST/SynthV export (aware of previous content)
        def _sanitize_ust_token(token: str, *, prev_was_content: bool) -> str:
            try:
                t = (token or '').strip()
                if not t:
                    return '-' if prev_was_content else 'R'
                # Strip brackets/quotes
                for ch in ['(', ')', '[', ']', '{', '}', '"', '“', '”', '‘', '’']:
                    t = t.replace(ch, '')
                t = t.strip()
                if not t:
                    return '-' if prev_was_content else 'R'
                # Collapse lingering hyphens
                t = t.replace('--', '-')
                if t == '-':
                    return '-'
                # Very short consonant-only artifacts → prefer sustain if continuing, else rest
                import re
                if re.fullmatch(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]", t):
                    return '-' if prev_was_content else 'R'
                return t
            except Exception:
                return '-' if prev_was_content else 'R'

        beats_per_bar = float(ts.get('beats_per_bar', 4.0))
        
        # Calculate section length (STRICT): require explicit section_length_bars
        if section_length_bars is None or float(section_length_bars) <= 0:
            raise ValueError("UST export requires section_length_bars; got None/<=0")
        section_len_beats = float(section_length_bars) * beats_per_bar
        
        note_blocks = []
        
        # Absolute timeline in beats; we emit rests to preserve spacing between parts/notes
        current_beat = 0.0
        # Cursor for absolute position (for part-precise filling)
        abs_cursor_beats = 0.0
        
        # Process each part sequentially - ensure we process ALL parts from loaded JSON
        total_parts = len(themes) if themes else 0
        # Threshold for merging micro-gaps (in denominator-beat units)
        small_gap_units = 0.5  # merge rests shorter/equal than half a denominator beat
        for part_idx in range(total_parts):
            th = themes[part_idx] if part_idx < len(themes) else {}
            trks = th.get('tracks', []) or []
            part_start = float(part_idx) * float(section_len_beats)
            # Hard part boundaries (absolute, in quarter-note beats)
            part_start_abs = float(part_idx) * float(section_len_beats)
            part_end_abs = float(part_idx + 1) * float(section_len_beats)
            # Strict: use exactly the passed track; no heuristic switching
            role_norm = None
            if 0 <= track_index < len(trks):
                try:
                    role_norm = str(trks[track_index].get('role','')).strip().lower() or None
                except Exception:
                    role_norm = None
            target_idx = track_index if (0 <= track_index < len(trks)) else -1

            if not (0 <= target_idx < len(trks)):
                # Output complete part as rest - consistent via abs_cursor_beats
                # Rest until part start
                if abs_cursor_beats < part_start_abs - 1e-6:
                    gap_to_part = part_start_abs - abs_cursor_beats
                    rest_ticks = int(round(gap_to_part * ticks_per_unit))
                    if rest_ticks > 0:
                        if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                            note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                        else:
                            note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                    abs_cursor_beats = part_start_abs
                # Rest for the complete part length
                full_part_ticks = int(round(section_len_beats * ticks_per_unit))
                if full_part_ticks > 0:
                    if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                        note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + full_part_ticks)
                    else:
                        note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": full_part_ticks})
                abs_cursor_beats = part_end_abs
                continue
            
            notes_loc = sorted(trks[target_idx].get('notes', []) or [], key=lambda n: float(n.get('start_beat', 0.0)))
            # Prefer tokens directly from the selected track to avoid index mismatches across parts
            toks_loc = []
            try:
                toks_loc = trks[target_idx].get('lyrics') or trks[target_idx].get('tokens') or []
            except Exception:
                toks_loc = []
            if not toks_loc:
                toks_loc = syllables_per_theme[part_idx] if part_idx < len(syllables_per_theme) else []

            # force_silence only when explicit flag is set (string or numeric)
            force_silence = False
            try:
                hint_theme = str(th.get('plan_hint', '') or '')
                hint_track = ''
                try:
                    hint_track = str(trks[target_idx].get('plan_hint', '') or '') if (0 <= target_idx < len(trks)) else ''
                except Exception:
                    hint_track = ''
                hint_all = (hint_theme + ' ' + hint_track).replace(' ', '').lower()
                if 'force_silence=1' in hint_all:
                    force_silence = True
                # Additionally: consider numeric flags on theme/track
                if not force_silence:
                    fs_theme = th.get('force_silence')
                    fs_track = (trks[target_idx].get('force_silence') if (0 <= target_idx < len(trks)) else None)
                    if fs_theme in (1, True, '1', 'true') or fs_track in (1, True, '1', 'true'):
                        force_silence = True
            except Exception:
                force_silence = False

            # Do NOT pad tokens with '-' to cover extra notes; extra notes will be treated as continuations

            # If no notes for this part, handle based on silence type
            if not notes_loc or force_silence:
                if force_silence or not toks_loc:
                    # Complete silence → fill exactly to part boundaries with 'R' (without double-padding)
                    # Rest until part start (if necessary)
                    if abs_cursor_beats < part_start_abs - 1e-6:
                        gap_to_part = part_start_abs - abs_cursor_beats
                        rest_ticks = int(round(gap_to_part * ticks_per_unit))
                        if rest_ticks > 0:
                            if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                            else:
                                note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                        abs_cursor_beats = part_start_abs
                    # Rest for complete part length
                    full_part_ticks = int(round((part_end_abs - abs_cursor_beats) * ticks_per_unit))
                    if full_part_ticks > 0:
                        if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                            note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + full_part_ticks)
                else:
                    note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": full_part_ticks})
                abs_cursor_beats = part_end_abs
                continue

            # STRICT: Map exactly section_len_beats per part (8 Bars)
            # (part_start_abs/part_end_abs are already defined above)

            # Ensure we are at least at this part's start
            if abs_cursor_beats < part_start_abs - 1e-6:
                gap_to_part = part_start_abs - abs_cursor_beats
                rest_ticks = int(round(gap_to_part * ticks_per_unit))
                if rest_ticks > 0:
                    if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                        note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                    else:
                        note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                abs_cursor_beats = part_start_abs

            # Notes are always part-relative in this context
            is_absolute = False

            # Process notes in part coordinates
            tok_i = 0
            prev_was_content = False
            SMALL_GAP_BEATS = 1.0/4.0
            for n in notes_loc:
                try:
                    local_start = max(0.0, float(n.get('start_beat', 0.0)))
                    d_beats = max(0.0, float(n.get('duration_beats', 0.0)))
                    # Determine tessitura center for fallback/clamping
                    try:
                        center_ref = None
                        if 0 <= target_idx < len(trks):
                            center_ref = trks[target_idx].get('tessitura_center')
                            if isinstance(center_ref, str) and center_ref.isdigit():
                                center_ref = int(center_ref)
                    except Exception:
                        center_ref = None

                    # Pitch with musical fallback: use tessitura_center if missing
                    p_raw = n.get('pitch', None)
                    if p_raw is None:
                        p = int(center_ref) if isinstance(center_ref, (int, float)) else MIDI_NOTE_C4
                    else:
                        p = int(p_raw)
                    
                    # Handle silence_role notes with minimal atmospheric content
                    if n.get('silence_role', False):
                        # Use tessitura_center for silence role notes, or A3 if not available
                        p = int(center_ref) if isinstance(center_ref, (int, float)) else 57  # A3
                        # Reduce velocity for atmospheric effect
                        velocity = max(20, min(40, int(n.get('velocity', 30))))
                    else:
                        velocity = int(n.get('velocity', 80))
                    abs_start = (local_start if is_absolute else (part_start_abs + local_start))
                    if abs_start >= part_end_abs - 1e-6:
                        break
                    # Rest until note start
                    if abs_cursor_beats < abs_start - 1e-6:
                        gap = abs_start - abs_cursor_beats
                        rest_ticks = int(round(gap * ticks_per_unit))
                        if rest_ticks > 0:
                            if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                            else:
                                note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                        abs_cursor_beats = abs_start
                    # Choose token
                    lyr_tok = toks_loc[tok_i] if tok_i < len(toks_loc) else '-'
                    tok_i += 1 if tok_i < len(toks_loc) else 0
                    raw = (str(lyr_tok) if lyr_tok is not None else '').strip()
                    # Limit duration to part end
                    d_beats = min(d_beats, max(0.0, part_end_abs - abs_start))
                    if d_beats <= 0.0:
                        continue
                    # Minimum duration 0.6 beats, then in ticks
                    note_ticks = int(round(max(0.6, d_beats) * ticks_per_unit))
                    # Dynamic clamp around center, if available
                    try:
                        if isinstance(center_ref, (int, float)):
                            lo = int(max(36, center_ref - 12))
                            hi = int(min(96, center_ref + 12))
                            clamped_pitch = max(lo, min(p, hi))
                        else:
                            clamped_pitch = max(MIDI_NOTE_C4, min(p, MIDI_NOTE_G5))
                    except Exception:
                        clamped_pitch = max(MIDI_NOTE_C4, min(p, MIDI_NOTE_G5))
                    # '-' extends previous content; otherwise new block
                    if raw == '-':
                        if note_blocks and note_blocks[-1].get('Lyric') not in ('-', 'R'):
                            note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + note_ticks)
                        else:
                            note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": note_ticks})
                        prev_was_content = False
                    else:
                        lyric_str = _sanitize_ust_token(raw, prev_was_content=prev_was_content)
                        prev_was_content = (lyric_str not in ('-', 'R'))
                        note_blocks.append({"Lyric": lyric_str, "NoteNum": clamped_pitch, "Length": note_ticks})
                    abs_cursor_beats = min(part_end_abs, abs_start + d_beats)
                except Exception:
                    continue

            # Fill part to end with rest
            if abs_cursor_beats < part_end_abs - 1e-6:
                rest_ticks_tail = int(round((part_end_abs - abs_cursor_beats) * ticks_per_unit))
                if rest_ticks_tail > 0:
                    if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                        note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks_tail)
                    else:
                        note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks_tail})
            # Always set cursor to part end to ensure correct timeline
            abs_cursor_beats = part_end_abs
        
        # If still empty, add a minimal rest
        if not note_blocks:
            note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": int(round(beats_per_bar * ticks_per_unit))})

        # Timeline is already correctly set by part boundaries - no additional padding needed

        # Write note blocks
        for idx, nb in enumerate(note_blocks):
            tag = f"[#NOTE{idx:04d}]"
            lines.append(tag)
            lines.append(f"Length={int(nb['Length'])}")
            lines.append(f"Lyric={nb['Lyric']}")
            lines.append(f"NoteNum={int(nb['NoteNum'])}")
            lines.append("Intensity=100")
            lines.append("Modulation=0")

        # Add singer and voice bank stub
        lines.append("[#SINGER]")
        lines.append("Name=DefaultSinger")
        lines.append("VoiceDir=")
        lines.append("[#TRACKEND]")

        # Diagnostics: log export parameters for bar math transparency
        try:
            total_parts = len(themes) if isinstance(themes, list) else 0
            bpb_loc = int(ts.get('beats_per_bar', 4)) if isinstance(ts, dict) else 4
            print(Style.DIM + f"[Export] parts={total_parts}, part_length_bars={section_length_bars}, beats_per_bar={bpb_loc}, total_bars={int(total_parts*float(section_length_bars))}" + Style.RESET_ALL)
        except Exception:
            pass
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(Fore.GREEN + f"Exported UST: {output_path}" + Style.RESET_ALL)
        return True
    except Exception as e:
        print(Fore.YELLOW + f"UST export failed: {e}" + Style.RESET_ALL)
        return False
    """
    Write a minimal UST file for OpenUtau for one track across all themes.
    - themes: list of parts; each has 'tracks' and selected track has 'notes' (absolute or per-part relative ok)
    - track_index: which track to export
    - syllables_per_theme: list parallel to themes; each element is syllables list 1:1 to notes of that theme's selected track
    - ts: time_signature dict for beats_per_bar
    - bpm: tempo
    - output_path: .ust target path
    """
    try:
        ticks_per_beat = TICKS_PER_BEAT
        lines = []
        lines.append("[#VERSION]")
        lines.append("UST Version=1.20")
        lines.append("[#SETTING]")
        lines.append(f"Tempo={float(bpm):.2f}")
        lines.append("Tracks=1")
        lines.append("ProjectName=LyricsExport")
        lines.append("Mode2=True")
        lines.append(f"TimeBase={TICKS_PER_BEAT}")

        # Internal: sanitize lyric token for UST/SynthV export
        def _sanitize_ust_token(token: str, *, prev_was_content: bool) -> str:
            try:
                t = (token or '').strip()
                if not t:
                    return '-' if prev_was_content else 'R'
                # Strip brackets/quotes
                for ch in ['(', ')', '[', ']', '{', '}', '"', '“', '”', '‘', '’']:
                    t = t.replace(ch, '')
                t = t.strip()
                if not t:
                    return '-' if prev_was_content else 'R'
                # Collapse lingering hyphens already stripped earlier
                t = t.replace('--', '-')
                if t == '-':
                    return '-'
                # Very short consonant-only artifacts → prefer sustain or simple vowel
                import re
                if re.fullmatch(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]", t):
                    return '-' if prev_was_content else 'R'
                return t
            except Exception:
                return '-' if prev_was_content else 'R'

        # Flatten notes across all themes (sequential order), inserting rests for gaps
        note_blocks = []
        current_index = 0
        abs_cursor_beats = 0.0
        beats_per_bar = int(ts.get('beats_per_bar', 4))
        # Determine section length (beats); prefer explicit bars, else fallback to 8 bars
        try:
            bpb = int(ts.get('beats_per_bar', 4))
        except Exception:
            bpb = 4
        # STRICT: require provided section_length_bars
        if not isinstance(section_length_bars, (int, float)) or float(section_length_bars) <= 0:
            raise ValueError("UST export requires section_length_bars; got None/<=0")
        # Use quarter-note beats: section length in beats = bars * beats_per_bar
        section_length_beats = float(section_length_bars) * float(bpb)

        # Absolute/relative: if part length is known, strictly use relative offsets
        treat_as_relative = True
        try:
            if (section_length_bars is None) and section_length_beats is not None and len(themes) >= 2:
                def _min_start(part_idx: int) -> float:
                    trks = themes[part_idx].get('tracks', [])
                    if not (0 <= track_index < len(trks)):
                        return 0.0
                    nlist = trks[track_index].get('notes', [])
                    if not nlist:
                        return 0.0
                    return min(float(n.get('start_beat', 0.0)) for n in nlist)
                abs_like = 0
                rel_like = 0
                for pi in range(1, len(themes)):
                    smin = _min_start(pi)
                    if smin >= section_length_beats * 0.5:
                        abs_like += 1
                    if smin <= section_length_beats * 0.25:
                        rel_like += 1
                treat_as_relative = (rel_like >= abs_like)
        except Exception:
            treat_as_relative = True

        # Timeline is managed by strict part boundaries - no need for total song length calculation

        # Process each part and find the vocal track
        for part_idx, th in enumerate(themes):
            trks = th.get('tracks', [])
            # Find vocal track in this part
            vocal_track = None
            for i, t in enumerate(trks):
                # 1) prefer __final_vocal__ flag
                if t.get('__final_vocal__') and (t.get('notes') or []):
                    vocal_track = t
                    break
                # 2) role == 'vocal'
                if str(t.get('role','')).lower() == 'vocal' and (t.get('notes') or []):
                    vocal_track = t
                    break
                # 3) name contains 'vocal'
                if 'vocal' in str(get_instrument_name(t)).lower() and (t.get('notes') or []):
                    vocal_track = t
                    break
            
            if not vocal_track:
                # No vocal track in this part; emit explicit rest covering this entire part
                if section_length_beats is not None and treat_as_relative:
                    part_start_abs = float(part_idx) * float(section_length_beats)
                    part_end_abs = float(part_idx + 1) * float(section_length_beats)
                    # If we're before the part, add rest up to part start
                    if abs_cursor_beats < part_start_abs - 1e-6:
                        gap_to_part = part_start_abs - abs_cursor_beats
                        rest_ticks = int(round(gap_to_part * ticks_per_beat))
                        if rest_ticks > 0:
                            if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                            else:
                                note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                        abs_cursor_beats = part_start_abs
                    # Add a full-part rest (or remaining segment) up to part end
                    if abs_cursor_beats < part_end_abs - 1e-6:
                        fill_len = part_end_abs - abs_cursor_beats
                        rest_ticks = int(round(fill_len * ticks_per_beat))
                        if rest_ticks > 0:
                            if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                            else:
                                note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                        abs_cursor_beats = part_end_abs
                continue
                
            notes = sorted(vocal_track.get('notes', []), key=lambda n: float(n.get('start_beat', 0.0)))
            sylls = syllables_per_theme[part_idx] if part_idx < len(syllables_per_theme) else []
            if not sylls:
                # fallback to track-provided tokens/lyrics if available
                try:
                    alt = vocal_track.get('lyrics', []) or vocal_track.get('tokens', []) or []
                    if isinstance(alt, list):
                        sylls = alt
                except Exception:
                    pass
            if not notes:
                # No notes in this part; emit explicit rest to preserve full part length
                if section_length_beats is not None and treat_as_relative:
                    part_start_abs = float(part_idx) * float(section_length_beats)
                    part_end_abs = float(part_idx + 1) * float(section_length_beats)
                    # If we're before the part, add rest up to part start
                    if abs_cursor_beats < part_start_abs - 1e-6:
                        gap_to_part = part_start_abs - abs_cursor_beats
                        rest_ticks = int(round(gap_to_part * ticks_per_beat))
                        if rest_ticks > 0:
                            if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                            else:
                                note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                        abs_cursor_beats = part_start_abs
                    # Add rest for the entire (remaining) part
                    if abs_cursor_beats < part_end_abs - 1e-6:
                        fill_len = part_end_abs - abs_cursor_beats
                        rest_ticks = int(round(fill_len * ticks_per_beat))
                        if rest_ticks > 0:
                            if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                                note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks)
                            else:
                                note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks})
                        abs_cursor_beats = part_end_abs
                continue
            # Reconcile token count to notes count to avoid dropping parts due to minor mismatches
            if len(sylls) != len(notes):
                try:
                    target = len(notes)
                    # Trim excess tokens, pad deficit with '-' sustains (or 'Ah' for first syllable)
                    if len(sylls) > target:
                        sylls = sylls[:target]
                    else:
                        pad_needed = target - len(sylls)
                        # Use '-' to sustain missing slots; do NOT inject vowel content
                        default_token = '-'
                        sylls = list(sylls) + [default_token] * pad_needed
                except Exception:
                    pass
            # If notes are relative per part, compute part offset
            part_offset = 0.0
            if section_length_beats is not None and treat_as_relative:
                part_offset = float(part_idx) * float(section_length_beats)
            # Define hard per-part boundaries for clamping (prevents shift if a part runs long)
            if section_length_beats is not None:
                part_start_abs = float(part_idx) * float(section_length_beats)
                part_end_abs = float(part_idx + 1) * float(section_length_beats)
            else:
                part_start_abs = None
                part_end_abs = None
            # Find first absolute start of next part (same track), used to allow cross-part sustains
            next_part_first_abs = None
            try:
                if section_length_beats is not None and (part_idx + 1) < len(themes):
                    next_trks = themes[part_idx + 1].get('tracks', [])
                    if 0 <= track_index < len(next_trks):
                        next_notes = sorted(next_trks[track_index].get('notes', []) or [], key=lambda n: float(n.get('start_beat', 0.0)))
                        if next_notes:
                            next_offset = float(part_idx + 1) * float(section_length_beats) if treat_as_relative else 0.0
                            next_part_first_abs = float(next_notes[0].get('start_beat', 0.0)) + next_offset
            except Exception:
                next_part_first_abs = None

            # Iterate notes and append rests as needed
            last_end = abs_cursor_beats
            SMALL_GAP_BEATS = 1.0/4.0
            prev_was_content = False
            continuation_count = 0
            for note, syl in zip(notes, sylls):
                start = float(note.get('start_beat', 0.0))
                dur = max(0.0, float(note.get('duration_beats', 0.0)))
                pitch = int(note.get('pitch', 60))
                abs_start = start + part_offset if section_length_beats is not None else start
                # If beyond this part's boundary, stop emitting in this part
                if part_end_abs is not None and abs_start >= part_end_abs - 1e-6:
                    break
                # Continue processing all notes - don't stop early based on song length
                # Insert rest if there is a gap (we will NOT render explicit 'R' as separate notes; merge rests implicitly)
                if abs_start > last_end + 1e-6:
                    rest_len_beats = abs_start - last_end
                    # Do not let rests cross the part boundary
                    if part_end_abs is not None:
                        rest_len_beats = min(rest_len_beats, max(0.0, part_end_abs - last_end))
                    # Cap any rest so it never extends beyond the overall song length
                    if song_total_beats:
                        remaining = song_total_beats - last_end
                        rest_len_beats = max(0.0, min(rest_len_beats, remaining))
                    # Soft ceiling for pathological rests when no global length is known
                    MAX_REST_BEATS = 2.0 * float(beats_per_bar)
                    if not song_total_beats and rest_len_beats > MAX_REST_BEATS:
                        rest_len_beats = MAX_REST_BEATS
                    # Attach tiny gaps to the previous note; output larger gaps as explicit rest notes 'R'
                    if rest_len_beats <= SMALL_GAP_BEATS and note_blocks:
                        note_blocks[-1]["Length"] = int(note_blocks[-1]["Length"] + round(rest_len_beats * ticks_per_beat))
                    else:
                        rest_ticks = int(round(rest_len_beats * ticks_per_beat))
                        if rest_ticks > 0:
                            note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": max(5, rest_ticks)})
                # Add the note or rest according to token
                # Allow sustain across part boundary; clamp only to next part's first onset or song end
                allowed_end_abs = None
                if next_part_first_abs is not None:
                    allowed_end_abs = next_part_first_abs
                if song_total_beats:
                    allowed_end_abs = (min(allowed_end_abs, song_total_beats) if allowed_end_abs is not None else song_total_beats)
                if allowed_end_abs is not None:
                    dur = min(dur, max(0.0, allowed_end_abs - abs_start))
                note_len_ticks = max(1, int(round(dur * ticks_per_beat)))
                # Enforce minimum content duration: prefer hold if continuing; keep first content even if short
                MIN_CONTENT_BEATS = 1.0/16.0
                min_content_ticks = int(round(MIN_CONTENT_BEATS * ticks_per_beat))
                lyric_token = str(syl).strip() if syl is not None else ''
                # Handle explicit silences or invalid leading continuations as rests
                if lyric_token.lower() in ("r", "silence", "[br]", "br", "breath") or (lyric_token == '-' and not prev_was_content) or lyric_token == '':
                    if note_len_ticks > 0:
                        tiny_thresh_ticks = int(round(SMALL_GAP_BEATS * ticks_per_beat))
                        if note_blocks and note_len_ticks <= tiny_thresh_ticks:
                            # attach tiny pause to the previous note
                            note_blocks[-1]["Length"] = int(note_blocks[-1]["Length"] + note_len_ticks)
                        else:
                            # larger pause as an actual rest note 'R'
                            note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": max(5, note_len_ticks)})
                    prev_was_content = False
                    continuation_count = 0
                    last_end = max(abs_start + (note_len_ticks / float(ticks_per_beat) if ticks_per_beat else 0.0), last_end)
                    continue
                # Map continuation indicators for SynthV/UTAU:
                # - '-' holds previous syllable (melisma) on continuation notes
                # - '+' is not exported; treat it as continuation and strip from output
                if lyric_token == '-':
                    lyric_str = '-'
                    continuation_count += 1
                elif lyric_token == '+':
                    continuation_count += 1
                    # Export nothing for '+'; fall through to next note
                    # Convert this '+' slot into a simple melisma hold as '-' to avoid gaps
                    lyric_str = '-'
                else:
                    # Clean inline hyphens in words (e.g., 'pier-cing' -> 'piercing') and sanitize meta tokens
                    clean = lyric_token.replace("-", "")
                    # Repair: if after cleaning the token is empty/consonant-only, map to '-' (if previous was content) else 'R'
                    lyric_str = _sanitize_ust_token(clean, prev_was_content=prev_was_content)
                    # If this will be content but the duration is too short, prefer hold only when continuing; keep first content
                    if lyric_str not in ('-', 'R') and note_len_ticks < min_content_ticks:
                        if prev_was_content:
                            lyric_str = '-'
                        else:
                            pass
                    prev_was_content = (lyric_str != '-' and lyric_str.lower() != 'r')
                    continuation_count = 0
                # Clamp pitch dynamically around tessitura_center (if available), otherwise standard C4–G5
                try:
                    center_ref = None
                    if 0 <= track_index < len(trks):
                        center_ref = trks[track_index].get('tessitura_center')
                        if isinstance(center_ref, str) and center_ref.isdigit():
                            center_ref = int(center_ref)
                    if isinstance(center_ref, (int, float)):
                        lo = int(max(36, center_ref - 12))
                        hi = int(min(96, center_ref + 12))
                        note_num = max(lo, min(pitch, hi))
                    else:
                        note_num = max(MIDI_NOTE_C4, min(pitch, MIDI_NOTE_G5))
                except Exception:
                    note_num = max(MIDI_NOTE_C4, min(pitch, MIDI_NOTE_G5))
                if note_len_ticks > 0:
                    note_blocks.append({"Lyric": lyric_str, "NoteNum": note_num, "Length": max(5, note_len_ticks)})
                    last_end = max(abs_start + (note_len_ticks / float(ticks_per_beat) if ticks_per_beat else 0.0), last_end)
                # If we reached the part end, stop this part
                if part_end_abs is not None and last_end >= part_end_abs - 1e-6:
                    break
            # Update cursor to last emitted end
            abs_cursor_beats = max(abs_cursor_beats, last_end)
            # Ensure the part fills to part end: if rest remains in part, add 'R' until part_end_abs
            if part_end_abs is not None and abs_cursor_beats < part_end_abs - 1e-6:
                gap_to_part_end = max(0.0, part_end_abs - abs_cursor_beats)
                rest_ticks_tail = int(round(gap_to_part_end * ticks_per_beat))
                if rest_ticks_tail > 0:
                    if note_blocks and note_blocks[-1].get('Lyric') == 'R':
                        note_blocks[-1]['Length'] = int(note_blocks[-1]['Length'] + rest_ticks_tail)
                    else:
                        note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": rest_ticks_tail})
                abs_cursor_beats = part_end_abs
            # Continue processing all parts - don't stop early based on song length

            

        # Coalesce micro rests and consecutive rests to reduce choppiness
        try:
            tiny_rest_ticks = int(round((1.0/4.0) * ticks_per_beat))
            coalesced = []
            for i, nb in enumerate(note_blocks):
                if nb.get('Lyric') == 'R' and nb.get('Length', 0) <= tiny_rest_ticks and coalesced:
                    # merge tiny rest into previous block
                    coalesced[-1]['Length'] = int(coalesced[-1].get('Length',0) + nb.get('Length',0))
                    continue
                if coalesced and nb.get('Lyric') == 'R' and coalesced[-1].get('Lyric') == 'R':
                    coalesced[-1]['Length'] = int(coalesced[-1].get('Length',0) + nb.get('Length',0))
                    continue
                coalesced.append(nb)
            note_blocks = coalesced
        except Exception:
            pass

        # Timeline is already correctly set by part boundaries - no additional padding needed
        
        # If still empty, add a minimal rest to make the UST loadable
        if not note_blocks:
            note_blocks.append({"Lyric": "R", "NoteNum": 60, "Length": int(round(bpb_local * ticks_per_beat))})

        # Write note blocks
        for idx, nb in enumerate(note_blocks):
            tag = f"[#NOTE{idx:04d}]"
            lines.append(tag)
            lines.append(f"Length={int(nb['Length'])}")
            lines.append(f"Lyric={nb['Lyric']}")
            lines.append(f"NoteNum={int(nb['NoteNum'])}")
            lines.append("Intensity=100")
            lines.append("Modulation=0")

        # Add singer and voice bank stub to encourage OpenUtau to import as a track
        lines.append("[#SINGER]")
        lines.append("Name=DefaultSinger")
        lines.append("VoiceDir=")
        lines.append("[#TRACKEND]")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(Fore.GREEN + f"Exported UST: {output_path}" + Style.RESET_ALL)
        return True
    except Exception as e:
        print(Fore.YELLOW + f"UST export failed: {e}" + Style.RESET_ALL)
        return False

def _export_emvoice_txt_for_track(themes: List[Dict], track_index: int, syllables_per_theme: List[List[str]], output_path_all: str, output_path_by_part: str | None = None) -> bool:
    """
    Writes plain-text files for Emvoice:
    - output_path_all: one single line with all syllables space-separated (entire song order)
    - output_path_by_part (optional): multiple lines, one per part (space-separated syllables)
    """
    try:
        # Build per-part tokens (already aligned 1:1 to notes per theme)
        per_part_tokens: List[List[str]] = []
        for part_idx, th in enumerate(themes):
            toks = syllables_per_theme[part_idx] if part_idx < len(syllables_per_theme) else []
            # Do not remap role-specific content here; rely on model tokens
            # Repair problematic tokens to avoid unintended vowels in export
            repaired = []
            prev = False
            import re
            for t in toks:
                s = (str(t) if t is not None else '').strip()
                if s == '+':
                    repaired.append('-'); prev = False; continue
                if s == '-':
                    repaired.append('-'); prev = False; continue
                for ch in ['(', ')', '[', ']', '{', '}', '"', '“', '”', '‘', '’']:
                    s = s.replace(ch, '')
                s = s.strip()
                if not s:
                    repaired.append('-' if prev else 'R'); prev = False; continue
                if re.fullmatch(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]", s):
                    repaired.append('-' if prev else 'R'); prev = False; continue
                repaired.append(s); prev = True
            per_part_tokens.append(repaired)

        # Flatten preserving part order
        flat_tokens: List[str] = []
        for toks in per_part_tokens:
            flat_tokens.extend(toks)

        with open(output_path_all, 'w', encoding='utf-8') as fa:
            fa.write(' '.join(flat_tokens))
        print(Fore.GREEN + f"Exported Emvoice TXT (whole song): {output_path_all}" + Style.RESET_ALL)

        if output_path_by_part:
            try:
                with open(output_path_by_part, 'w', encoding='utf-8') as fbp:
                    for part_idx, toks in enumerate(per_part_tokens):
                        label = themes[part_idx].get('label', f'Part_{part_idx+1}') if part_idx < len(themes) else f'Part_{part_idx+1}'
                        line = ' '.join(toks)
                        fline = f"[{label}] {line}" if isinstance(label, str) else line
                        fbp.write(fline + "\n")
                print(Fore.GREEN + f"Exported Emvoice TXT (by part): {output_path_by_part}" + Style.RESET_ALL)
            except Exception:
                pass
        return True
    except Exception as e:
        print(Fore.YELLOW + f"Emvoice TXT export failed: {e}" + Style.RESET_ALL)
        return False

# --- Build MIDI directly from UST so MIDI matches UST exactly ---
def _create_midi_from_ust(ust_path: str, bpm: float | int, ts: Dict, output_file: str) -> bool:
    try:
        with open(ust_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines()]
        notes = []  # list of tuples (lyric: str, note_num: int, length_ticks: int)
        
        # MIDI timing validation
        midi_debug = {
            "total_notes": 0,
            "total_beats": 0.0,
            "timing_errors": []
        }
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.startswith('[#NOTE'):
                length_ticks = None
                lyric = ''
                note_num = 60
                j = i + 1
                # read until next section or until fields found
                while j < len(lines) and not lines[j].startswith('[#'):
                    if lines[j].startswith('Length='):
                        try:
                            length_ticks = int(lines[j].split('=',1)[1].strip())
                        except Exception:
                            length_ticks = 0
                    elif lines[j].startswith('Lyric='):
                        lyric = lines[j].split('=',1)[1]
                    elif lines[j].startswith('NoteNum='):
                        try:
                            note_num = int(lines[j].split('=',1)[1].strip())
                        except Exception:
                            note_num = 60
                    j += 1
                if length_ticks is None:
                    length_ticks = 0
                lyr = (lyric or '').strip()
                notes.append((lyr, note_num, max(0, int(length_ticks))))
                i = j
                continue
            i += 1

        # Create MIDI
        try:
            bpb = int(ts.get('beats_per_bar', 4)) if isinstance(ts, dict) else 4
        except Exception:
            bpb = 4
        try:
            bval = int(ts.get('beat_value', 4)) if isinstance(ts, dict) else 4
        except Exception:
            bval = 4
        midi = MIDIFile(1, removeDuplicates=True, deinterleave=False)
        midi.addTempo(track=0, time=0, tempo=float(bpm))
        midi.addTimeSignature(track=0, time=0, numerator=bpb, denominator=bval, clocks_per_tick=24)

        current_ticks = 0
        prev_content_pitch = None
        MICRO_NOTE_TICKS = int(round((1.0/16.0) * float(TICKS_PER_BEAT)))
        for note_idx, (lyr, note_num, length_ticks) in enumerate(notes):
            length_ticks = max(0, int(length_ticks))
            dur_beats = float(length_ticks) / float(TICKS_PER_BEAT)
            time_beats = float(current_ticks) / float(TICKS_PER_BEAT)
            
            # Decide whether to emit MIDI based on lyric semantics
            lclean = (lyr or '').strip()
            emit_note = False
            if length_ticks > 0:
                if lclean.lower() == 'r' or lclean == '':
                    emit_note = False
                elif lclean == '-':
                    # Continuation: emit only if not micro AND pitch differs from previous content
                    if (length_ticks > MICRO_NOTE_TICKS) and (prev_content_pitch is not None) and (int(note_num) != int(prev_content_pitch)):
                        emit_note = True
                    else:
                        emit_note = False
                else:
                    emit_note = True

            # Track timing for debugging on content emissions
            if emit_note:
                midi_debug["total_notes"] += 1
                midi_debug["total_beats"] = max(midi_debug["total_beats"], time_beats + dur_beats)
                
                # Check for timing inconsistencies
                if time_beats < 0:
                    midi_debug["timing_errors"].append(f"Note {note_idx}: negative time {time_beats}")
                if dur_beats <= 0:
                    midi_debug["timing_errors"].append(f"Note {note_idx}: zero/negative duration {dur_beats}")
            
            if length_ticks > 0:
                if emit_note:
                    midi.addNote(track=0, channel=0, pitch=int(note_num), time=time_beats, duration=dur_beats, volume=100)
                    if lclean not in ('-', 'R', 'r', ''):
                        prev_content_pitch = int(note_num)
                # Advance time for all blocks (rests and holds) to preserve alignment
                current_ticks += length_ticks
        
        # Do not add artificial end markers; length is determined by last content note event

        # If there are no note events at all, do not write an empty MIDI
        try:
            has_note = any(True for lyr, _, lt in notes if ((lyr or '').strip() not in ('', 'R', 'r', '-') and lt > 0))
        except Exception:
            has_note = False
        if not has_note:
            print(Fore.YELLOW + "Warning: UST contains no playable notes; skipping MIDI write." + Style.RESET_ALL)
            return False
        with open(output_file, 'wb') as fw:
            midi.writeFile(fw)
        
        # Print MIDI timing validation report
        print(Fore.CYAN + f"MIDI Export Timing Report:" + Style.RESET_ALL)
        print(f"  Total notes: {midi_debug['total_notes']}")
        print(f"  Total beats: {midi_debug['total_beats']:.3f}")
        print(f"  Total ticks: {current_ticks}")
        
        if midi_debug["timing_errors"]:
            print(Fore.YELLOW + f"  Timing errors: {len(midi_debug['timing_errors'])}" + Style.RESET_ALL)
            for error in midi_debug["timing_errors"][:5]:  # Show first 5
                print(f"    {error}")
        else:
            print(Fore.GREEN + "  No timing errors" + Style.RESET_ALL)
        
        return True
    except Exception as e:
        print(Fore.YELLOW + f"Warning: UST→MIDI export failed: {e}" + Style.RESET_ALL)
        return False

# --- RESPONSE PARSING HELPERS (Robust JSON extraction) ---
# def _analyze_actual_music_scale(themes: List[Dict]) -> Tuple[str, str]:
#     """Analyze the actual scale used in the music by examining note patterns."""
#     # REMOVED: This function was causing incorrect scale "corrections"
#     # We now use the scale directly from the JSON as intended
#     return "C", "major"

def _validate_extracted_cfg(cfg: Dict) -> Dict:
    """Validate and clean extracted cfg data from progress files with explicit warnings."""
    validated = {}
    warnings = []
    
    # String fields - ensure they're strings and not empty (ONLY musical parameters)
    string_fields = ['key_scale', 'root_note', 'scale_type', 'genre', 'inspiration', 'lyrics_language']
    for field in string_fields:
        if field in cfg and cfg[field] is not None:
            value = str(cfg[field]).strip()
            if value:  # Only include non-empty strings
                validated[field] = value
                print(f"[DEBUG] ✅ Validated {field}: {validated[field]}")
            else:
                warnings.append(f"Empty string for '{field}' - ignoring")
        elif field in cfg:
            warnings.append(f"Null value for '{field}' - ignoring")
    
    # Numeric fields - validate ranges
    if 'bpm' in cfg and cfg['bpm'] is not None:
        try:
            bpm_val = float(cfg['bpm'])
            if 30 <= bpm_val <= 300:  # Reasonable BPM range
                validated['bpm'] = bpm_val
            else:
                warnings.append(f"BPM {bpm_val} out of range (30-300) - ignoring")
        except (ValueError, TypeError):
            warnings.append(f"Invalid BPM value '{cfg['bpm']}' - ignoring")
    
    if 'part_length' in cfg and cfg['part_length'] is not None:
        try:
            length_val = int(cfg['part_length'])
            if 1 <= length_val <= 64:  # Reasonable length range
                validated['part_length'] = length_val
            else:
                warnings.append(f"Part length {length_val} out of range (1-64) - ignoring")
        except (ValueError, TypeError):
            warnings.append(f"Invalid part_length value '{cfg['part_length']}' - ignoring")
    
    # Temperature fields should come from config.yaml, not JSON
    # (Removed temperature validation from JSON extraction)
    
    # Lyrics-specific numeric fields
    lyrics_fields = {
        'lyrics_target_words_per_bar': (0.5, 8.0),
        'lyrics_melisma_bias': (0.0, 1.0),
        'lyrics_min_word_beats': (0.5, 4.0),
        'lyrics_allow_nonsense': (0, 1)
    }
    
    for field, (min_val, max_val) in lyrics_fields.items():
        if field in cfg and cfg[field] is not None:
            try:
                val = float(cfg[field])
                if min_val <= val <= max_val:
                    validated[field] = val
                else:
                    warnings.append(f"{field} {val} out of range ({min_val}-{max_val}) - ignoring")
            except (ValueError, TypeError):
                warnings.append(f"Invalid {field} value '{cfg[field]}' - ignoring")
    
    # Time signature - validate structure
    if 'time_signature' in cfg and cfg['time_signature'] is not None:
        ts = cfg['time_signature']
        if isinstance(ts, dict) and 'beats_per_bar' in ts and 'beat_value' in ts:
            try:
                beats = int(ts['beats_per_bar'])
                beat_val = int(ts['beat_value'])
                if 1 <= beats <= 16 and beat_val in [1, 2, 4, 8, 16]:
                    validated['time_signature'] = ts
                else:
                    warnings.append(f"Invalid time signature {beats}/{beat_val} - ignoring")
            except (ValueError, TypeError):
                warnings.append(f"Invalid time signature structure '{ts}' - ignoring")
        else:
            warnings.append(f"Invalid time signature format '{ts}' - ignoring")
    
    # Print all warnings
    if warnings:
        print(Fore.YELLOW + f"JSON Validation Warnings ({len(warnings)}):" + Style.RESET_ALL)
        for warning in warnings:
            print(Fore.YELLOW + f"  - {warning}" + Style.RESET_ALL)
    
    return validated

def _get_config_value(cfg: Dict, config: Dict, key: str, default=None, warn_missing=True):
    """Unified fallback logic: cfg -> config -> default with explicit warnings"""
    
    # Technical parameters should ALWAYS come from config.yaml, not JSON
    technical_params = ['model_name', 'temperature', 'lyrics_temperature', 'max_output_tokens', 
                       'context_window_size', 'enable_hotkeys', 'stage2_invalid_retries']
    
    if key in technical_params:
        # Force technical parameters to come from config.yaml
        if warn_missing and key not in config:
            print(Fore.YELLOW + f"Warning: Technical parameter '{key}' missing in config.yaml" + Style.RESET_ALL)
        return config.get(key, default)
    
    # Musical parameters: cfg -> config -> default
    if cfg and key in cfg and cfg[key] is not None:
        return cfg[key]

    # Warn if falling back to config when cfg was expected
    if cfg and warn_missing:
        print(Fore.YELLOW + f"Warning: Missing '{key}' in cfg, using config fallback" + Style.RESET_ALL)

    return config.get(key, default)

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
    # If it already looks like a JSON object, return as-is (quick path)
    if cleaned.startswith('{') and cleaned.endswith('}'):
        return cleaned
    # Scan while respecting JSON string literals to avoid counting braces inside strings
    start = cleaned.find('{')
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return cleaned[start:i+1]
    return ""

def _sanitize_json_text_for_load(s: str) -> str:
    """Best-effort cleanup for near-JSON: normalize smart quotes and remove trailing commas.
    Returns possibly sanitized text; never raises.
    """
    try:
        t = s
        # Normalize "smart quotes" to escaped ASCII quotes inside JSON content
        t = t.replace("\u201C", '\\"').replace("\u201D", '\\"')
        t = t.replace(""", "'").replace(""", "'")
        # Normalize typographic apostrophes to plain apostrophe
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("'", "'").replace("'", "'")
        # Remove trailing commas before closing braces/brackets: , } / , ] → } / ]
        t = re.sub(r',\s*([}\]])', r'\1', t)
        return t
    except Exception:
        return s
# --- Minimal Vocal Toolbox (stage- and role-filtered) ---
def _vocal_toolbox_block(stage: int,
                         *,
                         is_chorus: bool = False,
                         is_drop: bool = False,
                         is_verse: bool = False,
                         is_prechorus: bool = False,
                         is_bridge: bool = False,
                         is_backing: bool = False,
                         is_scat: bool = False,
                         is_vowels: bool = False,
                         is_intro: bool = False,
                         is_outro: bool = False,
                         bpb: int = 4) -> str:
    """Returns a concise, Dreamtonics-/UTAU-friendly toolbox as a compact prompt block.
    - Stage 1 (Lyrics): text-related rules (word and syllable handling, hook integrity).
    - Stage 2 (Notes): note/mapping-related rules (continuation '-', gaps, lengths).
    Only rules relevant to the current stage/role are output.
    """
    try:
        lines: list[str] = []
        hdr = "VOCAL TOOLBOX (relevant):\n"
        if stage == 1:
            # Text-first guardrails
            lines.append("- Keep words intact; never split hook words.")
            lines.append("- Complement the existing arrangement; leave space where texture is dense.")
            lines.append("- Text density is flexible: it is OK to sing less (or not at all) in some bars.")
            lines.append("- Prefer whole words; split only if rhythm truly requires it; as a guide, keep ≤1 split per word outside the hook.")
            lines.append("- Favor long sustains on open vowels (ah/oh); keep i/e short.")
            if is_chorus or is_drop:
                lines.append("- Hook: unchanged, contiguous, 1 word = 1 token.")
            if is_verse:
                lines.append("- Concrete imagery, short phrases; avoid chorus wording.")
            if is_prechorus:
                lines.append("- Build tension/lead-in; only hint at the hook.")
            if is_bridge:
                lines.append("- Contrast in color/angle; be sparing with new content.")
            if is_backing:
                lines.append("- Echo/response fragments; no new content.")
            if is_scat:
                lines.append("- Only non-lexical syllables; vary the vowels.")
            if is_vowels:
                lines.append("- Open vowels for long sustains; no semantics.")
            if is_intro or is_outro:
                lines.append("- Allow minimalist language or silence.")
        else:
            # Notes-first guardrails (Dreamtonics/UTAU-compatible)
            lines.append("- '-' = continuation of the previous syllable (no gap inside a word).")
            lines.append(f"- No overlaps; connect exactly end→start; avoid gaps inside a word (≥ 1/{max(1, bpb*2)} beat only between phrases).")
            lines.append("- Prefer longer sustains on open vowels (a/ah/o).")
            lines.append("- Short notes < 1/3 beat only on clear accents; otherwise merge.")
            lines.append("- Rests only between words/phrases, never inside a word.")
            lines.append("- Complement other melodies: leave space when texture is busy; avoid crowding accents used by lead instruments.")
            lines.append("- If onset rate feels high or the texture is dense: prefer '-' sustains on open vowels rather than adding new onsets.")
            lines.append("- Optional call-and-response with other tracks; response phrases should be short, off-beat, and outside lead accents; when lead is dense, actively leave pauses.")
            lines.append("- It is fine to skip entire bars if silence serves the arrangement.")
            if is_chorus or is_drop:
                lines.append("- Start the hook on the downbeat; 1 note per hook word; do not split words.")
            if is_vowels:
                lines.append("- Few onsets, long legato arcs; pure vowel sustains.")
            if is_backing:
                lines.append("- Very short, repeatable snippets; off-beat entries; avoid lead accents; prefer 'ooh/ah' color over consonant chains.")
            if is_scat:
                lines.append("- Rhythmically percussive syllables with vowel coloring; small pitch steps; avoid long consonant clusters; minimal sustains.")
            if is_outro:
                lines.append("- At least 1 note + 1 token; prefer a long, calm fade-out.")
        if not lines:
            return ""
        return hdr + "\n".join(lines) + "\n\n"
    except Exception:
        return ""

# --- Prompt label utilities (for agile, configurable context headers) ---
def _get_prompt_labels(config: Dict) -> Dict[str, str]:
    try:
        defaults = {
            'genre': 'Genre',
            'language': 'Language',
            'key_scale': 'Key/Scale',
            'bpm': 'BPM',
            'time_signature': 'TimeSig',
            'length_bars': 'Length',
            'track': 'Track',
            'section': 'Section',
            'description': 'Description'
        }
        user_labels = config.get('prompt_labels') if isinstance(config.get('prompt_labels'), dict) else {}
        out = dict(defaults)
        for k, v in (user_labels or {}).items():
            if isinstance(k, str) and isinstance(v, (str, int, float)):
                out[k] = str(v)
        return out
    except Exception:
        return {
            'genre': 'Genre',
            'language': 'Language',
            'key_scale': 'Key/Scale',
            'bpm': 'BPM',
            'time_signature': 'TimeSig',
            'length_bars': 'Length',
            'track': 'Track',
            'section': 'Section',
            'description': 'Description'
        }

def _format_prompt_context_line(ctx: Dict[str, object], *, labels: Dict[str, str], sep: str = '; ') -> str:
    try:
        parts: list[str] = []
        for key in ['genre','language','key_scale','bpm','time_signature','length_bars','track','section','description']:
            if key in ctx:
                val = ctx.get(key)
                if val is None:
                    continue
                if isinstance(val, (list, dict)):
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                label = labels.get(key, key)
                parts.append(f"{label}={sval}")
        return sep.join(parts)
    except Exception:
        return ""

# Add new constants at the beginning of the file
AVAILABLE_LENGTHS = [4, 8, 16, 32, 64, 128]
DEFAULT_LENGTH = 16

# Initialize Colorama for console color support
init(autoreset=True)
# Cap for embedding context notes into prompts; too many causes malformed JSON on some models
MAX_NOTES_IN_CONTEXT = 500

# MIDI Constants
MIDI_NOTE_C4 = 60
MIDI_NOTE_G5 = 79
MIDI_NOTE_C3 = 48
MIDI_NOTE_C6 = 84
MIDI_MIN_NOTE = 36
MIDI_MAX_NOTE = 96

# Timing Constants
DEFAULT_RETRY_DELAY = 3
MAX_RETRY_DELAY = 30
QUOTA_RETRY_DELAY = 3600
SHORT_RETRY_DELAY = 0.5

# Token Constants
DEFAULT_MAX_TOKENS = 8192
MAX_MAX_TOKENS = 65536

# Retry Constants
DEFAULT_MAX_RETRIES = 6
MAX_FAILURE_CYCLES = 10

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
                f"h=halve context (this step), a=auto-escalate (flash→pro after {AUTO_ESCALATE_THRESHOLD} fails{escalate_state}), d=defer track, r=reset cooldowns"
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
              "; press 1/2/3/0 (model), 'h' (halve context), 'd' (defer), 'a' (auto-escalate), 's' (skip wait), 'r' (reset cooldowns)...") + Style.RESET_ALL)
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
                if ch == 'r':
                    _LAST_HOTKEY_TS['r'] = now
                    try:
                        _clear_all_cooldowns()
                        globals()['NEXT_HOURLY_PROBE_TS'] = 0.0
                        print(Fore.CYAN + "All key cooldowns cleared by user (hotkey 'r')." + Style.RESET_ALL)
                    except Exception:
                        pass
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
                if ch == 's':
                    _LAST_HOTKEY_TS['s'] = now
                    print(Fore.CYAN + "Skipping wait on user request." + Style.RESET_ALL)
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

        # --- NEW: MPE defaults ---
        if "mpe" not in config or not isinstance(config.get("mpe"), dict):
            config["mpe"] = {}
        mpe_cfg = config["mpe"]
        mpe_cfg.setdefault("enabled", 0)
        mpe_cfg.setdefault("zone", "lower")
        mpe_cfg.setdefault("master_channel", 1)
        mpe_cfg.setdefault("member_channels_start", 2)
        mpe_cfg.setdefault("member_channels_end", 16)
        mpe_cfg.setdefault("pitch_bend_range_semitones", 48)
        mpe_cfg.setdefault("max_voices", 10)
        mpe_cfg.setdefault("voice_steal_policy", "last_note")

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
        return [60, 62, 64, 65, 67, 69, 71]

# --- Lyrics-first helpers ---
def _build_temp_note_grid_for_lyrics(ts: Dict, theme_len_bars: int, target_wpb: float, downbeat_targets: List[int] | None = None, phrase_windows: List[List[int]] | None = None) -> List[Dict]:
    # Minimal grid: 1–2 onsets per bar (downbeat, optional midbeat), optional extras in phrase_windows
    try:
        beats_per_bar = int(ts.get("beats_per_bar", 4)) if isinstance(ts, dict) else 4
    except Exception:
        beats_per_bar = 4
    grid: List[Dict] = []
    # Decide slots per bar very conservatively from target_wpb
    try:
        tw = float(target_wpb or 2.0)
    except Exception:
        tw = 2.0
    slots_per_bar = 1 if tw <= 1.5 else 2
    for bar in range(int(theme_len_bars)):
        # Always place downbeat
        start = float(bar * beats_per_bar)
        # Short seed durations so that min_word_beats does not get too high
        grid.append({"start_beat": round(start, 6), "duration_beats": 0.5, "pitch": 60})
        # Optional midbeat, preferred in active/phrase bars
        if slots_per_bar >= 2:
            in_phrase = False
            try:
                if isinstance(phrase_windows, list):
                    for w in phrase_windows:
                        if isinstance(w, list) and len(w) == 2 and w[0] <= bar <= w[1]:
                            in_phrase = True
                            break
            except Exception:
                in_phrase = False
            if in_phrase or (downbeat_targets and bar in downbeat_targets):
                mid = start + beats_per_bar/2.0
                grid.append({"start_beat": round(mid, 6), "duration_beats": 0.4, "pitch": 60})
                # optionally a third slot when target words are higher (phrases only)
                if tw >= 2.3:
                    qtr = start + beats_per_bar/4.0
                    grid.append({"start_beat": round(qtr, 6), "duration_beats": 0.4, "pitch": 60})
    # Sort chronologisch
    grid.sort(key=lambda x: x["start_beat"])  
    return grid

# --- Post-processing: enforce role-based timing to avoid overly hasty mapping ---
def _enforce_role_timing_constraints(notes: List[Dict], bpb: int, role_norm: str, part_index: int, config: Dict) -> List[Dict]:
    try:
        if not isinstance(notes, list) or len(notes) == 0:
            return notes
        role = (role_norm or '').lower()
        # Base thresholds (in beats); raised to slow down phrasing as requested
        base_min = {
            'intro': 1.25,
            'whisper': 1.25,
            'spoken': 1.25,
            'phrase_spot': 1.0,
            'prechorus': 1.0,
            'chorus': 0.75,
            'vocal_fx': 0.6,
            'breaths': 1.0,
            'outro': 1.0,
            'breakdown': 1.0,
        }
        # Allow optional config override
        try:
            override = config.get('role_min_note_beats')
            if isinstance(override, dict):
                for k, v in override.items():
                    if isinstance(v, (int, float)):
                        base_min[str(k).lower()] = float(v)
        except Exception:
            pass
        min_dur = base_min.get(role, 0.75)
        # Early sections: add a gentle boost to slow down even more
        if isinstance(part_index, int) and part_index <= 1:
            min_dur *= 1.2

        # Work on a sorted copy and clamp durations without overlap
        sorted_notes = sorted([n for n in notes if isinstance(n, dict)], key=lambda x: float(x.get('start_beat', 0.0)))
        for i in range(len(sorted_notes)):
            try:
                s = float(sorted_notes[i].get('start_beat', 0.0))
                d = max(0.0, float(sorted_notes[i].get('duration_beats', 0.0)))
                # target minimal duration
                target = max(min_dur, d)
                # next note start to avoid overlap
                if i + 1 < len(sorted_notes):
                    next_s = float(sorted_notes[i+1].get('start_beat', s + target))
                    max_allowed = max(0.0, next_s - s)
                    target = min(target, max_allowed)
                sorted_notes[i]['duration_beats'] = round(target, 6)
            except Exception:
                continue
        return sorted_notes
    except Exception:
        return notes

def _synthesize_notes_from_tokens(tokens: List[str], grid_notes: List[Dict], ts: Dict, theme_len_bars: int, key_scale: str | None = None) -> List[Dict]:
    if not isinstance(tokens, list) or not isinstance(grid_notes, list):
        return []
    try:
        beats_per_bar = int(ts.get("beats_per_bar", 4)) if isinstance(ts, dict) else 4
    except Exception:
        beats_per_bar = 4
    total_beats = theme_len_bars * beats_per_bar
    
    # Convert leading '-' to 'ah' to ensure we have content onsets
    if len(tokens) > 0 and str(tokens[0]).strip() == '-':
        tokens = ['ah'] + tokens[1:]
    
    # derive syllable groups from tokens against grid
    notes_out: List[Dict] = []
    i = 0
    while i < min(len(tokens), len(grid_notes)):
        tok = str(tokens[i]).strip()
        if tok and tok != '-':
            # run length = current + following '-'
            run_len = 1
            j = i + 1
            while j < len(tokens) and j < len(grid_notes) and str(tokens[j]).strip() == '-':
                run_len += 1
                j += 1
            start = float(grid_notes[i].get('start_beat', 0.0))
            end_idx = min(i + run_len, len(grid_notes)-1)
            end = float(grid_notes[end_idx].get('start_beat', start))
            # if end equals start (last token), give a minimal duration
            dur = max(0.25, (end - start) if end > start else 0.5)
            # clamp to section
            if start + dur > total_beats:
                dur = max(0.25, total_beats - start)
            # Use tessitura_center from plan_hint if available, otherwise derive from key_scale
            tessitura_center = 60  # Default fallback
            try:
                # Try to get tessitura_center from the track's plan_hint
                if hasattr(grid_notes[i], 'get') and 'tessitura_center' in grid_notes[i]:
                    tessitura_center = int(grid_notes[i].get('tessitura_center', 60))
                else:
                    # Fallback: derive from key_scale
                    ks = str(key_scale or '').lower()
                    roots = {
                        'c': 60, 'c#': 61, 'db': 61, 'd': 62, 'd#': 63, 'eb': 63, 'e': 64,
                        'f': 65, 'f#': 66, 'gb': 66, 'g': 67, 'g#': 68, 'ab': 68, 'a': 57,  # A3 instead of A4
                        'a#': 58, 'bb': 58, 'b': 59
                    }
                    for r in roots.keys():
                        if ks.startswith(r):
                            tessitura_center = roots[r]
                            break
            except Exception:
                tessitura_center = 60
            
            # For monotone roles, use tessitura_center directly with small variation
            # Add subtle variation: ±1-2 semitones around center
            variation = (i % 3) - 1  # -1, 0, or +1 semitone variation
            pitch = max(MIDI_NOTE_C3, min(MIDI_NOTE_C6, tessitura_center + variation))
            notes_out.append({"start_beat": round(start, 6), "duration_beats": round(dur, 6), "pitch": int(pitch)})
            i = j
        else:
            i += 1
    return notes_out
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
        cfg_model_name = str(config.get("model_name", "gemini-2.5-pro"))
        if SESSION_MODEL_OVERRIDE and local_model_name == SESSION_MODEL_OVERRIDE:
            origin = "session override"
        elif local_model_name == cfg_model_name:
            origin = "config default"
        else:
            origin = "hotkey"
        print(Fore.CYAN + f"Model for this step: {local_model_name} ({origin}; config={cfg_model_name}). Press 1/2/3 to switch for THIS step only; press 0 to set current as session default." + Style.RESET_ALL)
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
            # AUTO_ESCALATE_TO_PRO disabled - keep using flash
            # if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and failure_for_escalation_count >= AUTO_ESCALATE_THRESHOLD:
            #     local_model_name = 'gemini-2.5-pro'
            #     print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {failure_for_escalation_count} failures." + Style.RESET_ALL)
            pass
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
    # Dynamic hotkey hint reflecting config model and alternate
    try:
        cfg_model_name = str(config.get("model_name", "gemini-2.5-pro"))
        alt_model_name = "gemini-2.5-flash" if cfg_model_name != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
        print(Fore.CYAN + (
            f"Hotkeys [Generate: {instrument_name}]: 1={cfg_model_name} (this step), 2={alt_model_name} (this step), 3=custom (this step), 0=set session default, "
            f"h=halve context (this step), a=auto-escalate (flash→pro after {AUTO_ESCALATE_THRESHOLD} fails [OFF]), d=defer track, r=reset cooldowns"
        ) + Style.RESET_ALL)
    except Exception:
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
                    try:
                        cfg_model_name = str(config.get("model_name", "gemini-2.5-pro"))
                        alt_model_name = "gemini-2.5-flash" if cfg_model_name != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
                        print(Fore.CYAN + f"Press 1={cfg_model_name}, 2={alt_model_name}, 3=custom (custom_model_name), s=skip wait; waiting..." + Style.RESET_ALL)
                    except Exception:
                        print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (custom_model_name), s=skip wait; waiting..." + Style.RESET_ALL)
                    end_t = time.time() + max(0.0, wait_time)
                    while time.time() < end_t:
                        if msvcrt.kbhit():
                            ch = msvcrt.getch().decode().lower()
                            if ch == '1':
                                local_model_name = str(config.get("model_name", "gemini-2.5-pro")); print(Fore.YELLOW + f"Switching to {local_model_name} (this track)." + Style.RESET_ALL); break
                            if ch == '2':
                                alt_model_name = "gemini-2.5-flash" if str(config.get("model_name", "gemini-2.5-pro")) != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
                                local_model_name = alt_model_name; print(Fore.YELLOW + f"Switching to {local_model_name} (this track)." + Style.RESET_ALL); break
                            if ch == '3':
                                custom = config.get('custom_model_name')
                                if custom:
                                    local_model_name = custom; print(Fore.YELLOW + f"Switching to {custom} (this track)." + Style.RESET_ALL); break
                            if ch == 's':
                                print(Fore.CYAN + "Skipping wait on user request." + Style.RESET_ALL); break
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
                    "max_output_tokens": config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
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
                                # AUTO_ESCALATE_TO_PRO disabled - keep using flash
                                # if AUTO_ESCALATE_TO_PRO:
                                #     local_model_name = 'gemini-2.5-pro'
                                #     max_tokens_fail_pro = 0
                                #     print(Fore.CYAN + "Auto-escalate after 6 MAX_TOKENS on flash → switching to pro for this track." + Style.RESET_ALL)
                                # else:
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
                            try:
                                cfg_model_name = str(config.get("model_name", "gemini-2.5-pro"))
                                alt_model_name = "gemini-2.5-flash" if cfg_model_name != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
                                print(f"  1) {cfg_model_name}\n  2) {alt_model_name}\n  3) custom\n  4) keep current")
                            except Exception:
                                print("  1) gemini-2.5-pro\n  2) gemini-2.5-flash\n  3) custom\n  4) keep current")
                            sel = input(Fore.GREEN + "> " + Style.RESET_ALL).strip()
                            if sel == '1':
                                local_model_name = config.get("model_name", "gemini-2.5-pro")
                                attempt_count = 0
                                continue
                            elif sel == '2':
                                alt_model_name = "gemini-2.5-flash" if str(config.get("model_name", "gemini-2.5-pro")) != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
                                local_model_name = alt_model_name
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
                try:
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
                    required_keys = {"pitch", "start_beat", "duration_beats", "velocity"}
                    validated_notes = [
                        note for note in notes_list 
                        if isinstance(note, dict) and required_keys.issubset(note.keys())
                    ]
                    if len(validated_notes) != len(notes_list):
                        invalid_count = len(notes_list) - len(validated_notes)
                        print(Fore.YELLOW + f"Warning: Skipped {invalid_count} invalid note objects" + Style.RESET_ALL)
                        
                        # --- Sustain Validation ---
                        validated_sustain = []
                        for event in sustain_events:
                            if not all(k in event for k in ["beat", "action"]):
                                print(Fore.YELLOW + f"Warning: Skipping invalid sustain event: {event}" + Style.RESET_ALL)
                                continue
                            validated_sustain.append(event)

                        print(Fore.GREEN + f"Successfully generated part for {instrument_name}." + Style.RESET_ALL)
                        return ({
                            "instrument_name": instrument_name,
                            "program_num": program_num,
                            "role": role,
                            "notes": validated_notes,
                            "sustain_pedal": validated_sustain
                        }, total_token_count)

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
                    # Track last seen quota classes
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    if qtype == "per-day":
                        globals()['LAST_PER_DAY_SEEN_TS'] = time.time()
                    elif qtype == "per-hour":
                        globals()['LAST_PER_HOUR_SEEN_TS'] = time.time()
                    # Apply cooldown based on detected window
                    # For daily quotas: retry hourly to probe reset windows
                    if qtype == "per-day":
                        # Probe hourly instead of locking 24h
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS, force=True)
                    elif qtype == "per-hour":
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                    if qtype == "per-day":
                        _set_key_cooldown(CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS, force=True)
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
                        if _all_keys_daily_exhausted():
                            # All keys are per-day exhausted -> force hourly probe cadence
                            _schedule_hourly_probe_if_needed()
                            wait_time = _seconds_until_hourly_probe()
                            print(Fore.CYAN + f"All keys daily-exhausted. Next hourly probe in ~{wait_time:.1f}s." + Style.RESET_ALL)
                        else:
                            wait_time = _seconds_until_first_available()
                            wait_time = max(5.0, min(wait_time, PER_HOUR_COOLDOWN_SECONDS))
                            print(Fore.CYAN + f"All keys cooling down. Next probe in ~{wait_time:.1f}s." + Style.RESET_ALL)
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
            final_base = build_final_song_basename(config, optimized_themes, run_timestamp, resumed=False, opt_iteration=opt_iteration_num)
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
                    "max_output_tokens": config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
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
                    # AUTO_ESCALATE_TO_PRO disabled - keep using flash
                    # if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                    #     local_model_name = 'gemini-2.5-pro'
                    #     model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                    #     print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
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
                        try:
                            cfg_model_name = str(config.get("model_name", "gemini-2.5-pro"))
                            alt_model_name = "gemini-2.5-flash" if cfg_model_name != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
                            print(Fore.CYAN + f"Press 1={cfg_model_name}, 2={alt_model_name}, 3=custom (config.custom_model_name) to switch model for THIS track; continuing attempts..." + Style.RESET_ALL)
                        except Exception:
                            print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (config.custom_model_name) to switch model for THIS track; continuing attempts..." + Style.RESET_ALL)
                    # Auto-escalate: if using flash and threshold exceeded - DISABLED
                    # if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                    #     local_model_name = 'gemini-2.5-pro'
                    #     print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
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
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    if qtype == "per-day":
                        globals()['LAST_PER_DAY_SEEN_TS'] = time.time()
                    elif qtype == "per-hour":
                        globals()['LAST_PER_HOUR_SEEN_TS'] = time.time()
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
                                # AUTO_ESCALATE_TO_PRO disabled - keep using flash
                                # if AUTO_ESCALATE_TO_PRO:
                                #     local_model_name = 'gemini-2.5-pro'
                                #     max_tokens_fail_pro = 0
                                #     print(Fore.CYAN + "Auto-escalate after 6 MAX_TOKENS on flash → switching to pro for this track (optimization)." + Style.RESET_ALL)
                                # else:
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
                                    if ch == '1': local_model_name = str(config.get("model_name", "gemini-2.5-pro")); break
                                    if ch == '2':
                                        alt_model_name = "gemini-2.5-flash" if str(config.get("model_name", "gemini-2.5-pro")) != "gemini-2.5-flash" else config.get("model_name", "gemini-2.5-pro")
                                        local_model_name = alt_model_name; break
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
                    "max_output_tokens": config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
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
                    # AUTO_ESCALATE_TO_PRO disabled - keep using flash
                    # if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                    #     local_model_name = 'gemini-2.5-pro'
                    #     model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                    #     print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
                json_payload = _extract_json_object(resp_text)
                if not json_payload:
                    json_failure_count += 1
                    if json_failure_count == 2 and sys.platform == "win32":
                        print(Fore.CYAN + "Press 1=pro, 2=flash, 3=custom (config.custom_model_name) to switch model for THIS track; continuing attempts..." + Style.RESET_ALL)
                    # AUTO_ESCALATE_TO_PRO disabled - keep using flash
                    # if AUTO_ESCALATE_TO_PRO and local_model_name == 'gemini-2.5-flash' and json_failure_count >= AUTO_ESCALATE_THRESHOLD:
                    #     local_model_name = 'gemini-2.5-pro'
                    #     model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
                    #     print(Fore.CYAN + f"Auto-escalate: switching to {local_model_name} for this track after {json_failure_count} failures." + Style.RESET_ALL)
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
                    try:
                        qt = _classify_quota_error(err)
                        KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qt
                        if qt == "per-day":
                            globals()['LAST_PER_DAY_SEEN_TS'] = time.time()
                        elif qt == "per-hour":
                            globals()['LAST_PER_HOUR_SEEN_TS'] = time.time()
                    except Exception:
                        pass
                    if len(API_KEYS) > 1:
                        new_key = get_next_api_key(); genai.configure(api_key=new_key); continue
                    base = 3; wait = min(3600, base * (2 ** quota_rotation_count) + random.uniform(0,5.0))
                    if _all_keys_daily_exhausted():
                        _schedule_hourly_probe_if_needed(); wait = _seconds_until_hourly_probe()
                    _interruptible_backoff(wait, config, context_label="Automation 429 cooldown")
                    quota_rotation_count += 1; continue
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

# --- NEW: Single-Track Prompting (Standalone) ---
def create_single_track_prompt(config: Dict, length_bars: int, instrument_name: str, program_num: int, role: str, description: str, *, mpe_enabled: bool) -> str:
    """A compact, role- and expression-focused prompt for standalone tracks (no song context)."""
    beats_per_bar = config["time_signature"]["beats_per_bar"]
    total_beats = length_bars * beats_per_bar
    scale_notes = get_scale_notes(config.get("root_note", 60), config.get("scale_type", "minor"))
    # FREE MODE: No hard role binding. Role is only an intent tag.

    a = config.get("automation_settings", {})
    use_pb = a.get("use_pitch_bend", 0) == 1
    use_cc = a.get("use_cc_automation", 0) == 1
    allowed_ccs = a.get("allowed_cc_numbers", [])

    mpe_text = ""
    if mpe_enabled:
        mpe_text = (
            "\n**MPE Focus:**\n"
            "- Prefer per-note pitch_bend curves (one curve per note).\n"
            "- Favor glides/slides on transitions and light vibrato on sustained notes.\n"
            "- Do not use track_automations for pitch bend; keep it per note.\n"
            "- Keep curve density moderate (~0.1 beat resolution) and only where musically meaningful.\n"
        )
    else:
        if use_pb or use_cc:
            mpe_text = (
                "\n**Expression (optional):**\n"
                "- You MAY use per-note pitch_bend or CC (only allowed: " + ", ".join(map(str, allowed_ccs)) + ") sparingly and musically.\n"
            )

    prompt = (
        f"You are an expert MIDI musician. Compose a single standalone track.\n\n"
        f"**Context**\n"
        f"- Genre: {config.get('genre','')}\n"
        f"- Tempo: {config.get('bpm',120)} BPM\n"
        f"- Time Signature: {beats_per_bar}/{config['time_signature'].get('beat_value',4)}\n"
        f"- Key/Scale: {config.get('key_scale','')} (Notes: {scale_notes})\n"
        f"- Track Length: {length_bars} bars ({total_beats} beats)\n"
        f"- Instrument: {instrument_name} (Program: {program_num})\n"
        f"- Intent (optional): {role or 'free'}\n\n"
        f"**Creative Direction**\n{description}\n\n"
        f"**Guidelines**\n"
        f"1. Freedom: You MAY combine chords, single-note melodies, long sustains, and overlaps.\n"
        f"2. Voice-leading & texture: Alternate between dense chord moments and sparse single notes; use overlaps for legato/tension.\n"
        f"3. Pitch material: Use the scale notes {scale_notes} predominantly, with occasional passing tones as tasteful exceptions.\n"
        f"4. Timing: 'start_beat' is relative to the start of this {length_bars}-bar clip.\n"
        f"5. Dynamics & groove: Shape phrases with velocity contours; use micro-timing subtly.\n"
        f"6. Density: Aim for < 300 notes; include intentional rests for clarity.\n"
        f"7. Sorting: sort notes by 'start_beat' ascending; use dot decimals.\n"
        f"8. Optional 'pattern_blocks' for very fast grids; notes may overlap.\n"
        f"{mpe_text}\n\n"
        f"**Output (strict JSON):**\n"
        f"Return a single JSON object with top-level keys: 'notes' (required), optional 'track_automations', 'sustain_pedal', 'pattern_blocks'.\n"
        f"Each note: pitch (0-127), start_beat (float), duration_beats (float), velocity (1-127).\n"
        f"Note-level automations (if any): 'automations': {{ 'pitch_bend': [{{type:'curve', start_beat:..., end_beat:..., start_value:..., end_value:..., bias:1.0}}], 'cc': [...] }}.\n"
        f"Output ONLY the JSON object, no prose.\n"
    )
    return prompt

def generate_single_track_data(config: Dict, length_bars: int, instrument_name: str, program_num: int, role: str, description: str, mpe_enabled: bool) -> Tuple[Dict, int]:
    """Generates a single standalone track using a compact prompt tailored to role/MPE."""
    global CURRENT_KEY_INDEX, SESSION_MODEL_OVERRIDE
    prompt = create_single_track_prompt(config, length_bars, instrument_name, program_num, role, description, mpe_enabled=mpe_enabled)
    local_model_name = SESSION_MODEL_OVERRIDE or config["model_name"]
    json_failure_count = 0
    total_token_count = 0
    print_hotkey_hint(config, context=f"Single: {instrument_name}")
    max_retries = 6
    last_resp_preview = ""
    last_error_msg = ""
    for attempt in range(max_retries):
        try:
            generation_config = {
                "temperature": config["temperature"],
                "response_mime_type": "application/json",
                "max_output_tokens": config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
            }
            model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            effective_prompt = prompt + "\nOutput: Return a single JSON object with a 'notes' key only; no prose.\n"
            response = model.generate_content(effective_prompt, safety_settings=safety_settings, generation_config=generation_config)
            # Safety block check (diagnostic)
            try:
                if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                    reason = None
                    try:
                        reason = response.prompt_feedback.block_reason.name
                    except Exception:
                        reason = str(getattr(response.prompt_feedback, 'block_reason', 'UNKNOWN'))
                    print(Fore.RED + f"Attempt {attempt+1}: Prompt was blocked by safety filter: {reason}" + Style.RESET_ALL)
                    json_failure_count += 1
                    continue
            except Exception:
                pass

            resp_text = _extract_text_from_response(response)
            if hasattr(response, 'usage_metadata'):
                total_token_count = response.usage_metadata.total_token_count or 0
            if not resp_text:
                json_failure_count += 1
                continue
            json_payload = _extract_json_object(resp_text)
            if not json_payload:
                json_failure_count += 1
                last_resp_preview = (resp_text or "")[-240:]
                continue
            try:
                data = json.loads(json_payload)
            except Exception:
                json_failure_count += 1
                last_resp_preview = (resp_text or "")[-240:]
                continue

            # --- Robust fallbacks for various response shapes ---
            def _looks_like_note(obj: Dict) -> bool:
                try:
                    return (
                        isinstance(obj, dict)
                        and 'pitch' in obj
                        and 'start_beat' in obj
                        and ('duration_beats' in obj or 'duration' in obj)
                    )
                except Exception:
                    return False

            if isinstance(data, list):
                if all(_looks_like_note(n) for n in data):
                    data = {"notes": data}
                else:
                    json_failure_count += 1
                    last_resp_preview = (resp_text or "")[-240:]
                    continue
            elif isinstance(data, dict):
                if "notes" not in data:
                    # Common wrapper keys
                    for k in ["track", "data", "midi", "result"]:
                        try:
                            if isinstance(data.get(k), dict) and isinstance(data[k].get("notes"), list):
                                data = data[k]
                                break
                        except Exception:
                            pass
                # Alternate key name
                if "notes" not in data and isinstance(data.get("events"), list):
                    data["notes"] = data.get("events")
                if "notes" not in data:
                    json_failure_count += 1
                    last_resp_preview = (resp_text or "")[-240:]
                    continue
            else:
                json_failure_count += 1
                last_resp_preview = (resp_text or "")[-240:]
                continue
            # Expand pattern blocks if present
            try:
                if isinstance(data.get("pattern_blocks"), list):
                    expanded = _expand_pattern_blocks(data.get("pattern_blocks"), length_bars, config["time_signature"]["beats_per_bar"])
                    if expanded:
                        if isinstance(data.get("notes"), list):
                            data["notes"].extend(expanded)
                        else:
                            data["notes"] = expanded
                # Ensure notes are sorted by start_beat
                if isinstance(data.get("notes"), list):
                    try:
                        data["notes"].sort(key=lambda n: float(n.get("start_beat", 0.0)))
                    except Exception:
                        pass
            except Exception:
                pass
            return {
                "instrument_name": instrument_name,
                "program_num": program_num,
                "role": role,
                "notes": data.get("notes", []),
                **({"sustain_pedal": data.get("sustain_pedal", [])} if isinstance(data.get("sustain_pedal"), list) else {}),
                **({"track_automations": data.get("track_automations", {})} if isinstance(data.get("track_automations"), dict) else {})
            }, total_token_count
        except Exception as e:
            json_failure_count += 1
            try:
                last_error_msg = str(e) or last_error_msg
            except Exception:
                pass
            # Handle 429 / quota with key rotation & cooldown
            try:
                err_text = str(e)
                if '429' in err_text or 'quota' in err_text.lower() or 'rate limit' in err_text.lower():
                    error_message = err_text
                    qtype = _classify_quota_error(error_message)
                    cooldown = PER_MINUTE_COOLDOWN_SECONDS
                    if qtype in ('per-hour', 'rate-limit'):
                        cooldown = PER_HOUR_COOLDOWN_SECONDS
                    elif qtype == 'per-day':
                        # Probe hourly instead of 24h lock
                        cooldown = PER_HOUR_COOLDOWN_SECONDS
                    _set_key_cooldown(CURRENT_KEY_INDEX, cooldown, force=(qtype=='per-day'))
                    KEY_QUOTA_TYPE[CURRENT_KEY_INDEX] = qtype
                    if qtype == 'per-day':
                        globals()['LAST_PER_DAY_SEEN_TS'] = time.time()
                    elif qtype == 'per-hour':
                        globals()['LAST_PER_HOUR_SEEN_TS'] = time.time()
                    nxt = _next_available_key(CURRENT_KEY_INDEX)
                    if nxt is not None:
                        CURRENT_KEY_INDEX = nxt
                        try:
                            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                        except Exception:
                            pass
                        print(Fore.CYAN + f"Retrying with API key #{CURRENT_KEY_INDEX+1} after quota classification: {qtype}" + Style.RESET_ALL)
                        continue
                    else:
                        if _all_keys_cooling_down():
                            if _all_keys_daily_exhausted():
                                _schedule_hourly_probe_if_needed()
                                wait_s = _seconds_until_hourly_probe()
                                print(Fore.YELLOW + f"All API keys daily-exhausted. Waiting {int(wait_s)}s before hourly probe..." + Style.RESET_ALL)
                            else:
                                wait_s = max(5.0, min(_seconds_until_first_available(), PER_HOUR_COOLDOWN_SECONDS))
                                print(Fore.YELLOW + f"All API keys cooling down ({qtype}). Waiting {int(wait_s)}s before retry..." + Style.RESET_ALL)
                            _interruptible_backoff(wait_s, config, context_label="SingleTrack 429 cooldown")
                            continue
            except Exception:
                pass
            time.sleep(1.0)
            continue
    print(Fore.RED + f"All attempts failed to produce valid JSON for single track (failures: {json_failure_count})." + Style.RESET_ALL)
    if last_resp_preview:
        print(Fore.YELLOW + "Response tail (debug): " + Style.DIM + last_resp_preview + Style.RESET_ALL)
    if last_error_msg:
        print(Fore.YELLOW + f"Last error: {last_error_msg}" + Style.RESET_ALL)
    return None, 0

# --- NEW: MPE Single-Track Optimization Prompt ---
def create_mpe_single_track_optimization_prompt(config: Dict, length_bars: int, base_track: Dict, description: str, *, mpe_enabled: bool) -> str:
    beats_per_bar = config["time_signature"]["beats_per_bar"]
    total_beats = length_bars * beats_per_bar
    scale_notes = get_scale_notes(config.get("root_note", 60), config.get("scale_type", "minor"))
    original_part_str = json.dumps({
        'instrument_name': get_instrument_name(base_track),
        'program_num': base_track.get('program_num', 0),
        'role': base_track.get('role', 'free'),
        'notes': base_track.get('notes', [])
    }, separators=(',', ':'))

    mpe_block = (
        "**MPE Optimization:**\n"
        "- Prefer per-note pitch_bend curves (glides; light vibrato on sustained notes).\n"
        "- Place curves sparingly at musical inflection points (chord changes, ornaments, suspensions).\n"
        "- Do not use track-wide pitch bends.\n"
        "- Keep density moderate (~0.1 beat resolution) and add a reset to 0 after curves.\n"
    ) if mpe_enabled else (
        "**Expression (optional):** You MAY use per-note pitch_bend and CC (allowed CCs per config) sparingly.\n"
    )

    prompt = (
        f"You are an expert MIDI musician. Refine the given standalone track to increase musical interest and expression.\n\n"
        f"**Context**\n"
        f"- Genre: {config.get('genre','')} | Tempo: {config.get('bpm',120)} BPM | Signature: {beats_per_bar}/{config['time_signature'].get('beat_value',4)} | Length: {length_bars} bars ({total_beats} beats)\n"
        f"- Key/Scale: {config.get('key_scale','')} (Notes: {scale_notes})\n\n"
        f"**Creative Direction (goal):**\n{description}\n\n"
        f"**Original Part (JSON):**\n```json\n{original_part_str}\n```\n\n"
        f"**Your Task (Conservative Changes):**\n"
        f"- Preserve the musical identity and structure.\n"
        f"- Create interesting motion via good voice-leading, occasional overlaps, and tasteful ornaments.\n"
        f"- Small, targeted note edits allowed (split/merge, subtle duration/position tweaks, < 15% note changes).\n"
        f"- Use rests deliberately; avoid clutter.\n\n"
        f"{mpe_block}\n\n"
        f"**Output (strict JSON):** Return a single object with 'notes' (required), optional 'sustain_pedal', 'track_automations', 'pattern_blocks'.\n"
        f"Each note: pitch, start_beat (relative to the part start), duration_beats, velocity.\n"
        f"Output ONLY the JSON object, no prose.\n"
    )
    return prompt

def generate_mpe_single_track_optimization_data(config: Dict, length_bars: int, base_track: Dict, description: str, mpe_enabled: bool) -> Tuple[Dict, int]:
    prompt = create_mpe_single_track_optimization_prompt(config, length_bars, base_track, description, mpe_enabled=mpe_enabled)
    local_model_name = SESSION_MODEL_OVERRIDE or config["model_name"]
    total_token_count = 0
    max_retries = 6
    for attempt in range(max_retries):
        try:
            generation_config = {
                "temperature": config["temperature"],
                "response_mime_type": "application/json",
                "max_output_tokens": config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
            }
            model = genai.GenerativeModel(model_name=local_model_name, generation_config=generation_config)
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            effective_prompt = prompt + "\nOutput: Return a single JSON object with a 'notes' key only; no prose.\n"
            response = model.generate_content(effective_prompt, safety_settings=safety_settings, generation_config=generation_config)
            resp_text = _extract_text_from_response(response)
            if hasattr(response, 'usage_metadata'):
                total_token_count = response.usage_metadata.total_token_count or 0
            if not resp_text:
                continue
            json_payload = _extract_json_object(resp_text)
            if not json_payload:
                continue
            data = json.loads(json_payload)
            if not isinstance(data, dict) or "notes" not in data:
                continue
            # Expand pattern blocks if present
            try:
                if isinstance(data.get("pattern_blocks"), list):
                    expanded = _expand_pattern_blocks(data.get("pattern_blocks"), length_bars, config["time_signature"]["beats_per_bar"])
                    if expanded:
                        if isinstance(data.get("notes"), list):
                            data["notes"].extend(expanded)
                        else:
                            data["notes"] = expanded
            except Exception:
                pass
            merged = {
                "instrument_name": get_instrument_name(base_track),
                "program_num": base_track.get("program_num", 0),
                "role": base_track.get("role", "free"),
                "notes": data.get("notes", base_track.get("notes", []))
            }
            if isinstance(data.get("sustain_pedal"), list):
                merged["sustain_pedal"] = data.get("sustain_pedal", [])
            if isinstance(data.get("track_automations"), dict):
                merged["track_automations"] = data.get("track_automations", {})
            return merged, total_token_count
        except Exception:
            continue
    return None, 0
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

        # --- MPE export parameters ---
        mpe_cfg = config.get('mpe', {}) if isinstance(config.get('mpe'), dict) else {}
        mpe_enabled_globally = int(mpe_cfg.get('enabled', 0)) == 1
        member_start = int(mpe_cfg.get('member_channels_start', 2))
        member_end = int(mpe_cfg.get('member_channels_end', 16))
        master_ch = int(mpe_cfg.get('master_channel', 1))
        pbr = int(mpe_cfg.get('pitch_bend_range_semitones', 48))
        max_voices = int(mpe_cfg.get('max_voices', 10))
        voice_policy = str(mpe_cfg.get('voice_steal_policy', 'last_note')).lower()

        def _clamp_ch(ch: int) -> int:
            return max(0, min(15, ch))

        member_start = _clamp_ch(member_start)
        member_end = _clamp_ch(member_end)
        master_ch = _clamp_ch(master_ch)
        if member_end < member_start:
            member_start, member_end = 2, 16

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

            # If this track carries lyrics, emit MIDI Lyric/Text events aligned to notes (word-first uses '-' on continuations)
            try:
                lyrics = track_data.get("lyrics")
                if isinstance(lyrics, list) and lyrics and len(lyrics) == len(track_data.get("notes", [])):
                    for ln, note in zip(lyrics, track_data.get("notes", [])):
                        try:
                            start_time = float(note.get("start_beat", 0.0)) + time_offset_beats
                            text = str(ln)
                            # Emit as Lyric metaevent when supported by MIDIUtil: addLyric(track, time, lyric)
                            try:
                                # Some MIDIUtil versions implement addLyric; if missing, fall back to addText
                                if hasattr(midi_file, 'addLyric'):
                                    midi_file.addLyric(midi_track_num, start_time, text)
                                else:
                                    midi_file.addText(midi_track_num, start_time, text)
                            except Exception:
                                midi_file.addText(midi_track_num, start_time, text)
                        except Exception:
                            continue
            except Exception:
                pass

            # Determine if this track should use MPE (compute early so channel/program changes can respect it)
            track_mpe = False
            try:
                track_mpe = bool(track_data.get('mpe_enabled')) or (mpe_enabled_globally and role in ["mpe_lead","mpe_chords","mpe_pads","mpe_arp"]) or str(role).lower().startswith('mpe_')
            except Exception:
                track_mpe = False
            # If MPE, use master channel as base channel
            base_channel = master_ch if (track_mpe and role not in ["drums","percussion","kick_and_snare"]) else channel
            try:
                program_num = int(program_num)
            except Exception:
                program_num = 0
            if program_num < 0 or program_num > 127:
                program_num = max(0, min(127, program_num))
            midi_file.addProgramChange(midi_track_num, base_channel, 0, program_num)
            # For MPE, also send ProgramChange on all member channels to ensure consistent timbre
            if track_mpe and role not in ["drums","percussion","kick_and_snare"]:
                try:
                    for mch in range(member_start, member_end + 1):
                        if mch == 9: continue
                        midi_file.addProgramChange(midi_track_num, mch, 0, program_num)
                except Exception:
                    pass
            
            # --- NEW: Process TRACK-LEVEL (Sound Design) Automations ---
            DRUM_ROLES = {"drums", "percussion", "kick_and_snare"}
            if "track_automations" in track_data and role not in DRUM_ROLES:
                # Process Pitch Bend (skip for MPE to avoid global channel bends)
                if allow_pitch_bend and "pitch_bend" in track_data["track_automations"] and not ("mpe_enabled" in track_data or str(role).lower().startswith('mpe_') or (mpe_enabled_globally and role in ["mpe_lead","mpe_chords","mpe_pads","mpe_arp"])):
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
                                current_val = max(-8192, min(8191, current_val))
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
                                cc_num = max(0, min(127, cc_num))
                                current_val = max(0, min(127, current_val))
                                current_time = cc_start_beat + t * duration
                                # If MPE track, fan out CC to all member channels; else, use the track channel
                                if ("mpe_enabled" in track_data or str(role).lower().startswith('mpe_') or (mpe_enabled_globally and role in ["mpe_lead","mpe_chords","mpe_pads","mpe_arp"])):
                                    try:
                                        for mch in mpe_member_channels:
                                            midi_file.addControllerEvent(midi_track_num, mch, current_time, cc_num, current_val)
                                    except Exception:
                                        midi_file.addControllerEvent(midi_track_num, channel, current_time, cc_num, current_val)
                                else:
                                    midi_file.addControllerEvent(midi_track_num, channel, current_time, cc_num, current_val)
                            # Enforce neutral reset (0) at curve end if needed
                            if cc_end_val != 0:
                                if ("mpe_enabled" in track_data or str(role).lower().startswith('mpe_') or (mpe_enabled_globally and role in ["mpe_lead","mpe_chords","mpe_pads","mpe_arp"])):
                                    try:
                                        for mch in mpe_member_channels:
                                            midi_file.addControllerEvent(midi_track_num, mch, cc_end_beat, cc_num, 0)
                                    except Exception:
                                        midi_file.addControllerEvent(midi_track_num, channel, cc_end_beat, cc_num, 0)
                                else:
                                    midi_file.addControllerEvent(midi_track_num, channel, cc_end_beat, cc_num, 0)
            
            # --- Sustain Pedal --- (skip for drums)
            if allow_sustain and "sustain_pedal" in track_data and role not in DRUM_ROLES:
                for sustain_event in track_data["sustain_pedal"]:
                    try:
                        sustain_time = float(sustain_event["beat"]) + time_offset_beats
                        sustain_action = sustain_event["action"].lower()
                        sustain_value = 127 if sustain_action == "down" else 0
                        if track_mpe:
                            # fan-out sustain across all member channels
                            try:
                                for mch in range(member_start, member_end + 1):
                                    if mch == 9: continue
                                    midi_file.addControllerEvent(midi_track_num, mch, sustain_time, 64, sustain_value)
                            except Exception:
                                midi_file.addControllerEvent(midi_track_num, base_channel, sustain_time, 64, sustain_value)
                        else:
                            midi_file.addControllerEvent(midi_track_num, base_channel, sustain_time, 64, sustain_value)
                    except (ValueError, TypeError, KeyError) as e:
                        print(Fore.YELLOW + f"Warning: Skipping invalid sustain event in track '{track_name}': {sustain_event}. Reason: {e}" + Style.RESET_ALL)
            
            # Initialize MPE member channels if needed (set Pitch Bend Range via RPN 0)
            mpe_member_channels = []
            if track_mpe and role not in ["drums","percussion","kick_and_snare"]:
                mpe_member_channels = [ch for ch in range(member_start, member_end + 1) if ch != 9 and ch != master_ch]
                if max_voices > 0:
                    mpe_member_channels = mpe_member_channels[:max_voices]
                # RPN 0 (Pitch Bend Sensitivity)
                try:
                    # Also set RPN on master channel for completeness
                    midi_file.addControllerEvent(midi_track_num, master_ch, 0, 101, 0)
                    midi_file.addControllerEvent(midi_track_num, master_ch, 0, 100, 0)
                    midi_file.addControllerEvent(midi_track_num, master_ch, 0, 6, max(0, min(127, pbr)))
                    midi_file.addControllerEvent(midi_track_num, master_ch, 0, 38, 0)
                    for mch in mpe_member_channels:
                        midi_file.addControllerEvent(midi_track_num, mch, 0, 101, 0)  # RPN MSB
                        midi_file.addControllerEvent(midi_track_num, mch, 0, 100, 0)  # RPN LSB
                        # Data Entry: semitones in MSB (LSB=0)
                        midi_file.addControllerEvent(midi_track_num, mch, 0, 6, max(0, min(127, pbr)))
                        midi_file.addControllerEvent(midi_track_num, mch, 0, 38, 0)
                except Exception:
                    pass

            # Simple voice allocator for MPE
            active_voices = []  # list of dicts: {ch, start, end}

            for note in track_data["notes"]:
                try:
                    pitch = int(note["pitch"])
                    start_beat = float(note["start_beat"])
                    duration_beats = float(note["duration_beats"])
                    velocity = int(note.get("velocity", 96))
                    
                    if 0 <= pitch <= 127 and 1 <= velocity <= 127 and duration_beats > 0:
                        if track_mpe and mpe_member_channels:
                            # allocate voice channel
                            ev_start = start_beat + time_offset_beats
                            ev_end = ev_start + duration_beats
                            # free finished voices
                            active_voices = [v for v in active_voices if v['end'] > ev_start - 1e-9]
                            use_ch = None
                            # find free channel
                            used = {v['ch'] for v in active_voices}
                            for mch in mpe_member_channels:
                                if mch not in used:
                                    use_ch = mch
                                    break
                            if use_ch is None:
                                # voice stealing
                                if voice_policy == 'oldest' and active_voices:
                                    steal = min(active_voices, key=lambda v: v['start'])
                                else:
                                    # last_note or fallback
                                    steal = max(active_voices, key=lambda v: v['start']) if active_voices else None
                                if steal:
                                    use_ch = steal['ch']
                                    active_voices = [v for v in active_voices if v is not steal]
                                else:
                                    use_ch = mpe_member_channels[0]
                            active_voices.append({'ch': use_ch, 'start': ev_start, 'end': ev_end})
                            midi_file.addNote(
                                track=midi_track_num,
                                channel=use_ch,
                                pitch=pitch,
                                time=ev_start,
                                duration=duration_beats,
                                volume=velocity
                            )
                        else:
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
                                        current_val = max(-8192, min(8191, current_val))
                                        current_time = pb_start_beat + t * duration
                                        # Route pitchbend either to an MPE voice or the base channel
                                        target_ch = None
                                        if track_mpe and mpe_member_channels:
                                            # find the voice that overlaps this moment
                                            for v in active_voices:
                                                if v['start'] - 1e-9 <= current_time <= v['end'] + 1e-9:
                                                    target_ch = v['ch']
                                                    break
                                            if target_ch is None and active_voices:
                                                # fallback: pick the voice with the smallest time distance
                                                target_ch = min(active_voices, key=lambda v: min(abs(current_time - v['start']), abs(current_time - v['end'])) )['ch']
                                            if target_ch is not None:
                                                midi_file.addPitchWheelEvent(midi_track_num, target_ch, current_time, current_val)
                                            # no further fallback to base channel → avoids global bend
                                        else:
                                            midi_file.addPitchWheelEvent(midi_track_num, base_channel, current_time, current_val)
                                    # Enforce neutral reset (0) at curve end if needed
                                    if pb_end_val != 0:
                                        if track_mpe and mpe_member_channels:
                                            target_ch = None
                                            for v in active_voices:
                                                if v['start'] - 1e-9 <= pb_end_beat <= v['end'] + 1e-9:
                                                    target_ch = v['ch']
                                                    break
                                            if target_ch is None and active_voices:
                                                target_ch = min(active_voices, key=lambda v: min(abs(pb_end_beat - v['start']), abs(pb_end_beat - v['end'])) )['ch']
                                            if target_ch is not None:
                                                midi_file.addPitchWheelEvent(midi_track_num, target_ch, pb_end_beat, 0)
                                        else:
                                            midi_file.addPitchWheelEvent(midi_track_num, base_channel, pb_end_beat, 0)
                                else: # Fallback for single points (legacy)
                                    pb_time = start_beat + time_offset_beats + pb.get("beat", 0)
                                    pb_value = int(pb.get("value", 0))
                                    pb_value = max(-8192, min(8191, pb_value))
                                    if track_mpe and mpe_member_channels:
                                        target_ch = channel
                                        for v in active_voices:
                                            if v['start'] - 1e-9 <= pb_time <= v['end'] + 1e-9:
                                                target_ch = v['ch']
                                                break
                                        midi_file.addPitchWheelEvent(midi_track_num, target_ch, pb_time, pb_value)
                                    else:
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
                                    cc_num = max(0, min(127, cc_num))
                                    cc_val = max(0, min(127, cc_val))
                                    if ("mpe_enabled" in track_data or str(role).lower().startswith('mpe_') or (mpe_enabled_globally and role in ["mpe_lead","mpe_chords","mpe_pads","mpe_arp"])):
                                        try:
                                            for mch in mpe_member_channels:
                                                midi_file.addControllerEvent(midi_track_num, mch, cc_time, cc_num, cc_val)
                                        except Exception:
                                            midi_file.addControllerEvent(midi_track_num, channel, cc_time, cc_num, cc_val)
                                    else:
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
                    # IMPORTANT: Do NOT shift note-level automation times by time_offset_beats here,
                    # because they are defined relative to the note's own start time.
                    # They will be converted to absolute times at emission.

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

            # Additionally clamp note-level automations (relative) so absolute times don't exceed the section
            try:
                max_len = float(section_length_beats)
                for tr in normalized_theme_data.get("tracks", []):
                    for note in tr.get("notes", []):
                        nstart = float(note.get("start_beat", 0))
                        # Clamp per-note pitch_bend curves
                        if isinstance(note.get("automations"), dict) and isinstance(note["automations"].get("pitch_bend"), list):
                            new_pbs = []
                            for pb in note["automations"]["pitch_bend"]:
                                try:
                                    if pb.get("type") == "curve":
                                        rel_sb = float(pb.get("start_beat", 0))
                                        rel_eb = float(pb.get("end_beat", 0))
                                        abs_sb = nstart + rel_sb
                                        abs_eb = nstart + rel_eb
                                        abs_sb = max(0.0, min(max_len, abs_sb))
                                        abs_eb = max(0.0, min(max_len, abs_eb))
                                        if abs_eb <= abs_sb:
                                            continue
                                        pb_out = dict(pb)
                                        pb_out["start_beat"] = max(0.0, abs_sb - nstart)
                                        pb_out["end_beat"] = max(0.0, abs_eb - nstart)
                                        new_pbs.append(pb_out)
                                    else:
                                        # single point: pb['beat'] is relative to note
                                        rel_b = float(pb.get("beat", 0))
                                        abs_b = nstart + rel_b
                                        if 0.0 <= abs_b <= max_len:
                                            new_pbs.append(pb)
                                except Exception:
                                    continue
                            note["automations"]["pitch_bend"] = new_pbs
                        # Clamp per-note CC curves similarly
                        if isinstance(note.get("automations"), dict) and isinstance(note["automations"].get("cc"), list):
                            new_ccs = []
                            for cc in note["automations"]["cc"]:
                                try:
                                    if cc.get("type") == "curve":
                                        rel_sb = float(cc.get("start_beat", 0))
                                        rel_eb = float(cc.get("end_beat", 0))
                                        abs_sb = nstart + rel_sb
                                        abs_eb = nstart + rel_eb
                                        abs_sb = max(0.0, min(max_len, abs_sb))
                                        abs_eb = max(0.0, min(max_len, abs_eb))
                                        if abs_eb <= abs_sb:
                                            continue
                                        cc_out = dict(cc)
                                        cc_out["start_beat"] = max(0.0, abs_sb - nstart)
                                        cc_out["end_beat"] = max(0.0, abs_eb - nstart)
                                        new_ccs.append(cc_out)
                                    else:
                                        rel_b = float(cc.get("beat", 0))
                                        abs_b = nstart + rel_b
                                        if 0.0 <= abs_b <= max_len:
                                            new_ccs.append(cc)
                                except Exception:
                                    continue
                            note["automations"]["cc"] = new_ccs
            except Exception:
                pass

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
    # --- NEW: Single Track Standalone Mode ---
    parser.add_argument('--single-track', action='store_true', help="Generate a single standalone track (no song context).")
    parser.add_argument('--st-name', type=str, help="Standalone track instrument name (e.g., 'Lead Synth').")
    parser.add_argument('--st-role', type=str, help="Standalone track role (e.g., lead, chords, pads, mpe_lead, mpe_chords).")
    parser.add_argument('--st-program', type=int, default=80, help="MIDI program number for the instrument (0-127).")
    parser.add_argument('--st-length', type=int, default=16, help="Length in bars for the standalone track.")
    parser.add_argument('--st-desc', type=str, help="Creative description for the standalone track.")
    parser.add_argument('--st-mpe', action='store_true', help="Enable MPE mode for this standalone track (channel-per-voice pitchbend).")
    parser.add_argument('--st-pbr', type=int, default=48, help="MPE pitch-bend range in semitones (default 48).")

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

    # --- Reset cooldown timers on fresh start ---
    try:
        _clear_all_cooldowns()
        globals()['NEXT_HOURLY_PROBE_TS'] = 0.0
    except Exception:
        pass

    # --- Validate and apply performance settings ---
    if "context_window_size" not in config:
        config["context_window_size"] = -1
    if "max_output_tokens" not in config:
        config["max_output_tokens"] = 8192

    # --- Direct Action Mode: --single-track ---
    if args.single_track:
        try:
            st_name = args.st_name or "Standalone_Instrument"
            st_role = args.st_role or "lead"
            st_prog = max(0, min(127, int(args.st_program)))
            st_len_bars = max(1, int(args.st_length))
            st_desc = (args.st_desc or "An expressive standalone musical part.").strip()
            run_timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(script_dir, f"Single_{_sanitize_filename_component(st_name)}_{st_len_bars}bars_{int(config.get('bpm',120))}bpm_{run_timestamp}.mid")

            # Use the specialized standalone prompting
            track, _ = generate_single_track_data(
                config, st_len_bars, st_name, st_prog, st_role, st_desc, mpe_enabled=(args.st_mpe or str(st_role).lower().startswith('mpe_'))
            )
            if not track:
                print(Fore.RED + "Failed to generate standalone track." + Style.RESET_ALL)
                return

            # Flag MPE on track if requested or role indicates MPE
            try:
                if args.st_mpe or str(st_role).lower().startswith('mpe_'):
                    track['mpe_enabled'] = True
            except Exception:
                pass

            # Inject per-run MPE export settings (non-destructive defaults)
            try:
                cfg_mpe = config.setdefault('mpe', {})
                if args.st_mpe or str(st_role).lower().startswith('mpe_'):
                    cfg_mpe.setdefault('enabled', 1)
                cfg_mpe.setdefault('zone', 'lower')
                cfg_mpe.setdefault('master_channel', 1)
                cfg_mpe.setdefault('member_channels_start', 2)
                cfg_mpe.setdefault('member_channels_end', 16)
                cfg_mpe.setdefault('pitch_bend_range_semitones', int(args.st_pbr or 48))
                cfg_mpe.setdefault('max_voices', 10)
                cfg_mpe.setdefault('voice_steal_policy', 'last_note')
            except Exception:
                pass

            theme_data = {"tracks": [track]}
            part_len_beats = st_len_bars * config["time_signature"]["beats_per_bar"]
            ok = create_part_midi_from_theme(theme_data, config, output_path, time_offset_beats=0, section_length_beats=part_len_beats)
            if ok:
                print(Fore.GREEN + f"Standalone track saved: {output_path}" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Failed to write standalone MIDI." + Style.RESET_ALL)
            return
        except Exception as e:
            print(Fore.RED + f"Standalone mode failed: {e}" + Style.RESET_ALL)
            return

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
    print(Style.BRIGHT + "        Song Generator - Interactive Mode")
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
            # New: Lyrics from final artifact
            menu_options[str(next_option)] = ('lyrics_from_artifact', "Generate Lyrics for a Track (from final artifact)")
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
                
                # Reload configuration when starting generation to pick up any changes in config.yaml
                if action in ['generate_again', 'generate_new']:
                    try:
                        print(Fore.CYAN + "Reloading configuration from 'config.yaml'..." + Style.RESET_ALL)
                        config = load_config(CONFIG_FILE)
                        if initialize_api_keys(config):
                            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                        else:
                            print(Fore.YELLOW + "Warning: No valid API key found after reload. API calls will fail." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"Failed to reload configuration: {str(e)}. Using in-memory config." + Style.RESET_ALL)

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
                # Ensure latest automation/call-response settings
                try:
                    print(Fore.CYAN + "Reloading configuration from 'config.yaml'..." + Style.RESET_ALL)
                    config = load_config(CONFIG_FILE)
                    if initialize_api_keys(config):
                        genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                except Exception as e:
                    print(Fore.YELLOW + f"Could not reload configuration for optimization: {e}. Proceeding with current settings." + Style.RESET_ALL)

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

                # Determine how many optimization cycles to run
                try:
                    opt_cycles = int(config.get('optimization_iterations', config.get('number_of_iterations', 1)))
                    if opt_cycles < 1:
                        opt_cycles = 1
                except Exception:
                    opt_cycles = 1

                print(Fore.CYAN + f"Optimization cycles (menu 3): {opt_cycles} (from config: optimization_iterations/number_of_iterations)" + Style.RESET_ALL)

                current_themes = themes_to_opt
                for cycle in range(opt_cycles):
                    # Compute next available opt index based on base filename
                    try:
                        base_no_opt = build_final_song_basename(config, current_themes, run_timestamp, resumed=False)
                        opt_iter = get_next_available_file_number(os.path.join(script_dir, base_no_opt + ".mid"))
                    except Exception:
                        opt_iter = get_next_available_file_number(os.path.join(script_dir, final_song_basename + ".mid"))

                    print(Fore.CYAN + f"\n--- Starting Optimization (Version {opt_iter}, cycle {cycle+1}/{opt_cycles}) ---" + Style.RESET_ALL)

                    optimized_themes = create_song_optimization(
                        config, theme_len, current_themes, script_dir,
                        opt_iter, run_timestamp, user_opt_prompt
                    )

                    if not optimized_themes:
                        print(Fore.RED + "Optimization failed for this cycle. Stopping further cycles." + Style.RESET_ALL)
                        break

                    time.sleep(2)
                    final_song_data = merge_themes_to_song_data(optimized_themes, config, theme_len)
                    last_generated_themes = optimized_themes
                    last_generated_song_data = final_song_data
                    current_themes = optimized_themes

                    # Save meta artifact for this optimized result
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
            
            elif action == 'optimize_artifact':
                print_header("Optimize Existing Song (Artifacts)")
                # Ensure latest automation/call-response settings
                try:
                    print(Fore.CYAN + "Reloading configuration from 'config.yaml'..." + Style.RESET_ALL)
                    config = load_config(CONFIG_FILE)
                    if initialize_api_keys(config):
                        genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                except Exception as e:
                    print(Fore.YELLOW + f"Could not reload configuration for artifact optimization: {e}. Proceeding with current settings." + Style.RESET_ALL)
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

                # Determine cycles from config
                try:
                    opt_cycles = int(config.get('optimization_iterations', config.get('number_of_iterations', 1)))
                    if opt_cycles < 1:
                        opt_cycles = 1
                except Exception:
                    opt_cycles = 1
                print(Fore.CYAN + f"Optimization cycles (menu 4): {opt_cycles} (from config: optimization_iterations/number_of_iterations)" + Style.RESET_ALL)

                current_themes = themes_to_opt
                for cycle in range(opt_cycles):
                    try:
                        base_no_opt = build_final_song_basename(config, current_themes, run_timestamp, resumed=False)
                        opt_iter = get_next_available_file_number(os.path.join(script_dir, base_no_opt + ".mid"))
                    except Exception:
                        opt_iter = get_next_available_file_number(os.path.join(script_dir, "final_song_" + run_timestamp + ".mid"))

                    print(Fore.CYAN + f"\n--- Starting Optimization (Version {opt_iter}, cycle {cycle+1}/{opt_cycles}) ---" + Style.RESET_ALL)
                    optimized_themes = create_song_optimization(
                        config, art_length, current_themes, script_dir,
                        opt_iter, run_timestamp, user_opt_prompt
                    )
                    if not optimized_themes:
                        print(Fore.RED + "Optimization failed for this cycle. Stopping further cycles." + Style.RESET_ALL)
                        break
                    # Save new artifact so it can be selected later (do not remove old; keep history)
                    save_final_artifact(config, optimized_themes, art_length, defs, script_dir, run_timestamp)
                    current_themes = optimized_themes
                print(Fore.GREEN + "\nOptimization complete." + Style.RESET_ALL)
            elif action == 'lyrics_from_artifact':
                print_header("Generate Lyrics for a Track (Artifact/Progress)")
                # Ensure API keys/model are configured for LLM calls
                try:
                    print(Fore.CYAN + "Reloading configuration from 'config.yaml'..." + Style.RESET_ALL)
                    config = load_config(CONFIG_FILE)
                    if initialize_api_keys(config):
                        genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                        print(Fore.GREEN + f"API keys initialized: {len(API_KEYS)} key(s) available" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "ERROR: No API keys found in config.yaml!" + Style.RESET_ALL)
                        return
                except Exception as e:
                    print(Fore.RED + f"ERROR: Could not initialize API keys for lyrics: {e}" + Style.RESET_ALL)
                    return

                # --- Resume support for lyrics generation ---
                try:
                    # Reset step-scoped model switch from any prior runs
                    globals()['REQUESTED_SWITCH_MODEL'] = None
                    # Also reset session override to use config model
                    globals()['SESSION_MODEL_OVERRIDE'] = None
                except Exception:
                    pass
                run_timestamp = time.strftime("%Y%m%d-%H%M%S")
                resume_idx = 0
                out_themes = None
                resume_data = None  # Store all resume data
                try:
                    pfiles = find_progress_files(script_dir)
                    # Only consider the most recent matching lyrics progress file
                    candidate = None
                    if pfiles:
                        for pf in pfiles:
                            pdata = _load_progress_silent(pf)
                            if not isinstance(pdata, dict):
                                continue
                            ptype = str(pdata.get('type',''))
                            pgen = str(pdata.get('generation_type',''))
                            if ptype.startswith('lyrics_generation') or pgen in ('new_vocal','existing_track'):
                                candidate = pf
                                break
                    if candidate:
                        summary = summarize_progress_file(candidate)
                        ans = input(f"{Fore.CYAN}Resume previous lyrics generation '{os.path.basename(candidate)}' ({summary})? [Y/n]: {Style.RESET_ALL}").strip().lower()
                        if ans in ('', 'y', 'yes'):
                            pdata = load_progress(candidate)
                            if isinstance(pdata, dict) and isinstance(pdata.get('themes'), list):
                                resume_data = pdata  # Store complete resume data
                                out_themes = pdata['themes']
                                try:
                                    resume_idx = int(pdata.get('current_theme_index', 0)) + 1
                                    if resume_idx < 0:
                                        resume_idx = 0
                                except Exception:
                                    resume_idx = 0
                                rt = str(pdata.get('timestamp','')).strip()
                                if rt:
                                    run_timestamp = rt
                                print(Fore.GREEN + f"Resuming lyrics generation from part {resume_idx+1}." + Style.RESET_ALL)
                        else:
                            # Ablehnung -> alten Progress verwerfen
                            try:
                                os.remove(candidate)
                                print(Fore.YELLOW + f"Discarded previous lyrics progress: {os.path.basename(candidate)}" + Style.RESET_ALL)
                            except Exception:
                                pass
                except Exception:
                    pass

                # If no resume, proceed with normal artifact selection
                if out_themes is None:
                    # Build combined selection list: Final artifacts + Progress files
                    finals = find_final_artifacts(script_dir)
                    progresses = find_progress_files(script_dir)
                    combined = []  # (path, is_final, label)
                    for p in finals:
                        try:
                            combined.append((p, True, "[F] " + summarize_artifact(p)))
                        except Exception:
                            combined.append((p, True, "[F] " + os.path.basename(p)))
                    for p in progresses:
                        try:
                            combined.append((p, False, "[P] " + summarize_progress_file(p)))
                        except Exception:
                            combined.append((p, False, "[P] " + os.path.basename(p)))

                    if not combined:
                        print(Fore.YELLOW + "No final artifacts or progress files found." + Style.RESET_ALL)
                        continue

                    for i, (_, _, label) in enumerate(combined):
                        print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {label}")
                    try:
                        sel = input(f"{Fore.GREEN}Choose item (1-{len(combined)}): {Style.RESET_ALL}").strip()
                        idx = int(sel) - 1
                    except Exception:
                        print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL)
                        continue
                    if not (0 <= idx < len(combined)):
                        print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL)
                        continue

                    selected_path, is_final, _ = combined[idx]
                else:
                    # Resume case - use dummy values
                    selected_path = None
                    is_final = False

                # Helper: comprehensive extraction from progress files for lyrics
                def _extract_from_progress_for_lyrics(pdata: Dict) -> tuple:
                    """Extract and validate all relevant configuration data from progress files."""
                    stored_config = pdata.get('config') or {}
                    cfg = {}
                    
                    # Extract ONLY musical parameters from JSON - technical parameters come from config.yaml
                    musical_fields = [
                        'key_scale', 'root_note', 'scale_type', 'bpm', 'genre', 'inspiration',
                        'lyrics_language', 'time_signature', 'part_length', 'lyrics_target_words_per_bar', 
                        'lyrics_melisma_bias', 'lyrics_min_word_beats', 'lyrics_allow_nonsense'
                    ]
                    
                    # Technical parameters that should ALWAYS come from config.yaml, not JSON:
                    # - model_name, temperature, lyrics_temperature, max_output_tokens, etc.
                    
                    for field in musical_fields:
                        if field in stored_config and stored_config[field] is not None:
                            cfg[field] = stored_config[field]
                    
                    print(f"[DEBUG] Progress data keys: {list(pdata.keys())}")
                    print(f"[DEBUG] Stored config keys: {list(stored_config.keys())}")
                    print(f"[DEBUG] Stored config key_scale: {stored_config.get('key_scale', 'NOT_FOUND')}")
                    print(f"[DEBUG] Stored config genre: {stored_config.get('genre', 'NOT_FOUND')}")
                    print(f"[DEBUG] Stored config bpm: {stored_config.get('bpm', 'NOT_FOUND')}")
                    
                    # CRITICAL: Check if key_scale is actually in stored_config
                    if 'key_scale' in stored_config:
                        print(f"[DEBUG] ✅ key_scale found in stored_config: {stored_config['key_scale']}")
                    else:
                        print(f"[DEBUG] ❌ key_scale NOT found in stored_config!")
                    
                    print(f"[DEBUG] Extracted cfg from progress BEFORE validation: {cfg}")
                    
                    # CRITICAL: Force the correct values BEFORE validation
                    if 'key_scale' in stored_config:
                        cfg['key_scale'] = stored_config['key_scale']
                        print(f"[DEBUG] 🔧 FORCED key_scale to: {cfg['key_scale']}")
                    if 'genre' in stored_config:
                        cfg['genre'] = stored_config['genre']
                        print(f"[DEBUG] 🔧 FORCED genre to: {cfg['genre']}")
                    if 'bpm' in stored_config:
                        cfg['bpm'] = stored_config['bpm']
                        print(f"[DEBUG] 🔧 FORCED bpm to: {cfg['bpm']}")
                    
                    # Validate and clean extracted data AFTER forcing values
                    cfg = _validate_extracted_cfg(cfg)
                    
                    print(f"[DEBUG] Extracted cfg from progress AFTER validation: {cfg}")
                    
                    length_bars = int(pdata.get('theme_length') or pdata.get('length') or previous_settings.get('length', DEFAULT_LENGTH))
                    themes = None
                    
                    # Try by type first
                    ptype = str(pdata.get('type') or '').lower()
                    if 'generation' in ptype:
                        themes = pdata.get('all_themes_data') or pdata.get('themes')
                    if themes is None and 'window_optimization' in ptype:
                        themes = pdata.get('themes')
                    if themes is None and 'optimization' in ptype:
                        themes = pdata.get('themes_to_optimize') or pdata.get('themes') or pdata.get('final_optimized_themes')
                    if themes is None and 'automation_enhancement' in ptype:
                        themes = pdata.get('themes')
                    # Generic fallbacks
                    if themes is None:
                        themes = pdata.get('themes') or pdata.get('all_themes_data') or pdata.get('themes_to_optimize')
                    return cfg, themes, length_bars

                if out_themes is not None:
                    # Resume case - use existing data
                    cfg = {}
                    themes = out_themes
                    theme_len_bars = 8  # Default fallback
                    num_parts = len(out_themes) if isinstance(out_themes, list) else 0
                    print(Style.DIM + f"[Resume] Using existing themes data with {num_parts} parts." + Style.RESET_ALL)
                elif is_final:
                    artifact = load_final_artifact(selected_path)
                    if not artifact:
                        continue
                    cfg = artifact.get('config', {})
                    themes = artifact.get('themes', [])
                    # STRICT: part length exclusively from artifact['length']
                    if 'length' not in artifact:
                        print(Fore.RED + "Final artifact missing 'length' (bars per part)." + Style.RESET_ALL)
                        continue
                    theme_len_bars = int(artifact['length'])
                    # Determine number of parts exclusively from theme_definitions
                    try:
                        theme_defs = artifact.get('theme_definitions')
                        num_parts = int(len(theme_defs)) if isinstance(theme_defs, list) else int(len(themes) if isinstance(themes, list) else 0)
                    except Exception:
                        num_parts = int(len(themes) if isinstance(themes, list) else 0)
                    try:
                        print(Style.DIM + f"[Export] Using part_length_bars={theme_len_bars} from final artifact (length); parts={num_parts} from theme_definitions." + Style.RESET_ALL)
                    except Exception:
                        pass
                else:
                    pdata = load_progress(selected_path)
                    if not pdata:
                        print(Fore.YELLOW + "Could not load the selected progress file." + Style.RESET_ALL)
                        continue
                    cfg, themes, _lb = _extract_from_progress_for_lyrics(pdata)
                    # STRICT: Prefer 'theme_length', otherwise 'length'; no silent fallbacks
                    if 'theme_length' in pdata and isinstance(pdata.get('theme_length'), (int, float)):
                        theme_len_bars = int(pdata.get('theme_length'))
                        src = 'theme_length'
                    elif 'length' in pdata and isinstance(pdata.get('length'), (int, float)):
                        theme_len_bars = int(pdata.get('length'))
                        src = 'length'
                    else:
                        print(Fore.RED + "Progress missing 'theme_length'/'length' (bars per part)." + Style.RESET_ALL)
                        continue
                    # Number of parts from theme_definitions (Fallback: themes)
                    try:
                        theme_defs = pdata.get('theme_definitions')
                        num_parts = int(len(theme_defs)) if isinstance(theme_defs, list) else int(len(themes) if isinstance(themes, list) else 0)
                    except Exception:
                        num_parts = int(len(themes) if isinstance(themes, list) else 0)
                    try:
                        print(Style.DIM + f"[Export] Using part_length_bars={theme_len_bars} from progress ({src}); parts={num_parts} from theme_definitions." + Style.RESET_ALL)
                    except Exception:
                        pass
                if not themes:
                    print(Fore.YELLOW + "Selected item has no themes to process." + Style.RESET_ALL)
                    continue

                # Build global track name list from all themes (collect unique tracks)
                def get_all_unique_tracks_from_themes(themes: List[Dict]) -> List[Dict]:
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
                
                base_tracks = get_all_unique_tracks_from_themes(themes)
                if not base_tracks:
                    print(Fore.YELLOW + "Artifact has no tracks in any part." + Style.RESET_ALL); continue
                
                # For resume case, determine track settings from existing data
                if out_themes is not None:
                    # Resume case - determine track settings from existing data
                    new_vocal_track_mode = True  # Assume new vocal track for resume
                    track_idx = 0  # Default fallback
                    print(Style.DIM + f"[Resume] Using existing track configuration." + Style.RESET_ALL)
                else:
                    # Normal case - ask user to choose track
                    for i, tr in enumerate(base_tracks):
                        print(f"{Fore.YELLOW}{i+1}.{Style.RESET_ALL} {get_instrument_name(tr)} ({tr.get('role','complementary')})")
                    extra_opt = len(base_tracks) + 1
                    print(f"{Fore.YELLOW}{extra_opt}.{Style.RESET_ALL} Generate NEW Vocal Track (notes + lyrics + UST)")
                    try:
                        tsel = input(f"{Fore.GREEN}Choose track (1-{len(base_tracks)} or {extra_opt} for NEW): {Style.RESET_ALL}").strip()
                        track_choice = int(tsel)
                    except Exception:
                        print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL); continue
                    if not (1 <= track_choice <= extra_opt):
                        print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL); continue
                    new_vocal_track_mode = (track_choice == extra_opt)
                    if not new_vocal_track_mode:
                        # Find the track index in the first theme by name
                        selected_track_name = get_instrument_name(base_tracks[track_choice - 1])
                        track_idx = -1
                        for i, tr in enumerate(themes[0].get('tracks', [])):
                            if get_instrument_name(tr) == selected_track_name:
                                track_idx = i
                                break
                        if track_idx == -1:
                            print(Fore.YELLOW + f"Track '{selected_track_name}' not found in first theme." + Style.RESET_ALL); continue

                # Prepare constants
                genre = cfg.get('genre', 'Unknown')
                inspiration = cfg.get('inspiration', '')
                bpm = cfg.get('bpm', 120)
                ts = cfg.get('time_signature', {"beats_per_bar": 4, "beat_value": 4})
                # theme_len_bars is set based on selected item (artifact/progress)
                beats_per_bar = int(ts.get('beats_per_bar', 4))
                section_len_beats = theme_len_bars * beats_per_bar

                # Create a copy of all themes and fill lyrics for the selected track index across parts
                if out_themes is None:
                    out_themes = json.loads(json.dumps(themes))
                # Cut to the defined number of parts
                try:
                    if isinstance(out_themes, list) and isinstance(num_parts, int) and num_parts > 0:
                        out_themes = out_themes[:num_parts]
                        print(Style.DIM + f"[Export] Trimmed themes to {len(out_themes)} part(s) based on theme_definitions." + Style.RESET_ALL)
                except Exception:
                    pass
                try:
                    if new_vocal_track_mode:
                        print(Fore.CYAN + f"Generating notes+lyrics for NEW track 'Vocal' across {len(out_themes)} part(s)..." + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"Generating lyrics (word-first) for track '{get_instrument_name(base_tracks[track_idx])}' across {len(out_themes)} part(s)..." + Style.RESET_ALL)
                except Exception:
                    pass
                # Prompt (optional) - use saved data if resuming
                if resume_data is not None:
                    user_guidance = resume_data.get('user_guidance', '')
                    print(Style.DIM + f"[Resume] Using saved guidance: '{user_guidance}'" + Style.RESET_ALL)
                else:
                    try:
                        user_guidance = input(f"{Fore.CYAN}Optional: Add a short guidance for the lyric style/content (or press Enter): {Style.RESET_ALL}").strip()
                    except Exception:
                        user_guidance = ""

                # Stage-0: Analyze user prompt into structured directives
                if new_vocal_track_mode:
                    # For NEW vocal tracks, use lyrics-first approach
                    print(Fore.CYAN + "Using lyrics-first approach for new vocal track..." + Style.RESET_ALL)
                else:
                    # For existing tracks, use word-first approach
                    print(Fore.CYAN + "Using word-first approach for existing track..." + Style.RESET_ALL)
                # Ensure themes are present (configurable artifact loading)
                try:
                    cfg0 = (cfg or config)
                    artifact_from_cfg = None
                    try:
                        apath = cfg0.get('lyrics_plan_artifact_path')
                        if isinstance(apath, str) and apath.strip():
                            artifact_from_cfg = apath.strip()
                    except Exception:
                        artifact_from_cfg = None
                    try:
                        load_latest_flag = int(cfg0.get('lyrics_plan_load_latest_final', 1))
                    except Exception:
                        load_latest_flag = 1
                    if not (isinstance(out_themes, list) and out_themes):
                        if artifact_from_cfg:
                            try:
                                with open(artifact_from_cfg, 'r', encoding='utf-8') as af:
                                    art = json.load(af)
                                cand = art.get('themes') if isinstance(art, dict) else None
                                if isinstance(cand, list) and cand:
                                    out_themes = cand
                                    try:
                                        if (not theme_len_bars) and isinstance(art.get('length'), (int, float)):
                                            theme_len_bars = int(art.get('length'))
                                    except Exception:
                                        pass
                                    print(Style.DIM + f"[Plan] Loaded artifact for themes (config): {artifact_from_cfg}" + Style.RESET_ALL)
                            except Exception:
                                pass
                        elif load_latest_flag == 1:
                            arts = find_final_artifacts(script_dir)
                            if isinstance(arts, list) and arts:
                                latest = arts[-1]
                                try:
                                    with open(latest, 'r', encoding='utf-8') as af:
                                        art = json.load(af)
                                    cand = art.get('themes') if isinstance(art, dict) else None
                                    if isinstance(cand, list) and cand:
                                        out_themes = cand
                                        try:
                                            if (not theme_len_bars) and isinstance(art.get('length'), (int, float)):
                                                theme_len_bars = int(art.get('length'))
                                        except Exception:
                                            pass
                                        print(Style.DIM + f"[Plan] Loaded latest artifact for themes: {latest}" + Style.RESET_ALL)
                                except Exception:
                                    pass
                except Exception:
                    pass
                # Build summaries: for NEW vocal, aggregate over existing tracks; otherwise summarize selected track
                try:
                    if new_vocal_track_mode:
                        summaries = _summarize_parts_aggregate(out_themes, ts, exclude_track_index=None)
                    else:
                        summaries = _summarize_vocal_parts(out_themes, track_idx, ts)
                except Exception:
                    summaries = []
                # Diagnostics for planning input
                try:
                    print(Style.DIM + f"[Plan Input] themes={len(out_themes) if isinstance(out_themes, list) else 'NA'} parts; track_idx={track_idx}; summaries={len(summaries)}" + Style.RESET_ALL)
                except Exception:
                    pass
                # Optional synthetic summaries via config flag (default off)
                try:
                    allow_synth = bool(int((cfg or config).get('lyrics_plan_allow_synthetic_summaries', 1)))
                except Exception:
                    allow_synth = True
                if not summaries:
                    # Try aggregate summaries from other tracks (NEW vocal path)
                    try:
                        agg = _summarize_parts_aggregate(out_themes, ts, exclude_track_index=track_idx)
                        if agg:
                            summaries = agg
                            print(Style.DIM + f"[Plan] Using aggregate summaries from other tracks (parts={len(summaries)})" + Style.RESET_ALL)
                    except Exception:
                        pass
                if not summaries:
                    if isinstance(out_themes, list) and out_themes:
                        # As a last resort, synthesize minimal silent summaries to allow planning to proceed
                        try:
                            print(Style.DIM + "[Plan] Summaries still empty – synthesizing minimal silent summaries (last resort)." + Style.RESET_ALL)
                        except Exception:
                            pass
                        try:
                            synth_list = []
                            for i, th in enumerate(out_themes or []):
                                label = th.get('label', f'Part_{i+1}') if isinstance(th, dict) else f'Part_{i+1}'
                                synth_list.append({"idx": i, "label": label, "num_notes": 0, "notes_density": 0.0, "avg_dur": 0.0, "max_dur": 0.0, "sustain_ratio": 0.0, "silent": True})
                            summaries = synth_list
                        except Exception:
                            summaries = []
                    if not summaries:
                        try:
                            print(Fore.RED + Style.BRIGHT + "[Plan ERROR] No summaries available for planning." + Style.RESET_ALL)
                        except Exception:
                            pass
                        raise RuntimeError("No summaries available for planning")
                try:
                    ANALYSIS_CTX = _analyze_user_prompt(cfg or config, genre, inspiration, user_guidance)
                    if isinstance(ANALYSIS_CTX, dict) and ANALYSIS_CTX:
                        print(Style.DIM + f"[Prompt-Analysis] {json.dumps({k: ANALYSIS_CTX.get(k) for k in ['hook_canonical','style_tags','repetition_policy'] if k in ANALYSIS_CTX})}" + Style.RESET_ALL)
                except Exception:
                    ANALYSIS_CTX = {}
                try:
                    # Step 0: Planning - skip if resuming
                    if resume_data is not None:
                        # Use saved plan data
                        roles = resume_data.get('roles', [])
                        plan_items = resume_data.get('plan_items', [])
                        ANALYSIS_CTX = resume_data.get('analysis_ctx', {})
                        print(Style.DIM + "[Resume] Using saved plan data (skipping Step 0)" + Style.RESET_ALL)
                    else:
                        # Step 0a: Prompt Analysis (already done above)
                        # Step 0b: Plan vocal roles
                        print(Style.DIM + "[Step 0b] Planning vocal roles..." + Style.RESET_ALL)
                        roles = _plan_vocal_roles(config, genre, inspiration, bpm, ts, summaries, ANALYSIS_CTX, user_guidance, cfg)
                        
                        # Step 0c: Generate hints with consistency validation
                        print(Style.DIM + "[Step 0c] Generating hints with consistency validation..." + Style.RESET_ALL)
                        plan_items = _generate_vocal_hints(config, genre, inspiration, bpm, ts, summaries, roles, ANALYSIS_CTX, user_guidance, cfg)
                    
                    # Print plan at call site for visibility
                    try:
                        if debug_plan:
                            print(Style.BRIGHT + "[Plan OUT]" + Style.RESET_ALL)
                            print(json.dumps({"plan": plan_items}, ensure_ascii=False, indent=2))
                            roles_dbg = [str((it or {}).get('role','')) for it in (plan_items or [])]
                            print(Style.DIM + "[Plan Roles] " + ", ".join(roles_dbg[:32]) + (" …" if len(roles_dbg)>32 else "") + Style.RESET_ALL)
                    except Exception:
                        pass
                except Exception as e:
                    # Bubble up with clear message
                    print(Fore.RED + Style.BRIGHT + f"[Plan ERROR] {e}" + Style.RESET_ALL)
                    raise
                # Always show concise hook summary from plan (if any)
                try:
                    hooks = []
                    seen = set()
                    for it in (plan_items or []):
                        hc = it.get('hook_canonical') if isinstance(it, dict) else None
                        if isinstance(hc, str):
                            h = hc.strip()
                            if h and h not in seen:
                                seen.add(h); hooks.append(h)
                    if hooks:
                        print(Style.BRIGHT + "[Plan Hook]" + Style.RESET_ALL)
                        msg = ", ".join(hooks[:3]) + (" …" if len(hooks) > 3 else "")
                        print(msg)
                except Exception:
                    pass
                # Concise summaries of other step-0 values
                try:
                    # Hook Canonical (aggregate unique)
                    hooks_can = []
                    seen_hc = set()
                    for it in (plan_items or []):
                        hc = it.get('hook_canonical') if isinstance(it, dict) else None
                        if isinstance(hc, str):
                            t = hc.strip()
                            if t and t not in seen_hc:
                                seen_hc.add(t); hooks_can.append(t)
                    if hooks_can:
                        print(Style.BRIGHT + "[Plan Hook Canonical]" + Style.RESET_ALL)
                        # Show up to 2 canonical hooks
                        print(", ".join([f'"{s}"' if '"' not in s else s for s in hooks_can[:2]]) + (" …" if len(hooks_can) > 2 else ""))
                except Exception:
                    pass
                try:
                    # Hook Themes (aggregate unique)
                    hooks = []
                    seen_h = set()
                    for it in (plan_items or []):
                        ht = it.get('hook_theme') if isinstance(it, dict) else None
                        if isinstance(ht, str):
                            t = ht.strip()
                            if t and t not in seen_h:
                                seen_h.add(t); hooks.append(t)
                    if hooks:
                        print(Style.BRIGHT + "[Plan Hook Themes]" + Style.RESET_ALL)
                        print(", ".join(hooks[:3]) + (" …" if len(hooks) > 3 else ""))
                except Exception:
                    pass
                try:
                    # Chorus lines (aggregate unique)
                    lines_seen = []
                    lines_set = set()
                    for it in (plan_items or []):
                        cls = it.get('chorus_lines') if isinstance(it, dict) else None
                        if isinstance(cls, list):
                            for s in cls:
                                if isinstance(s, str):
                                    t = s.strip()
                                    if t and t not in lines_set:
                                        lines_set.add(t); lines_seen.append(t)
                    if lines_seen:
                        print(Style.BRIGHT + "[Plan Chorus Lines]" + Style.RESET_ALL)
                        print(", ".join(lines_seen[:3]) + (" …" if len(lines_seen) > 3 else ""))
                except Exception:
                    pass
                try:
                    # Story (aggregate unique, short)
                    stories = []
                    seen_s = set()
                    for it in (plan_items or []):
                        st = it.get('story') if isinstance(it, dict) else None
                        if isinstance(st, str):
                            t = st.strip()
                            if t and t not in seen_s:
                                seen_s.add(t); stories.append(t)
                    if stories:
                        print(Style.BRIGHT + "[Plan Story]" + Style.RESET_ALL)
                        # Print one or two concise items
                        def _shorten(x: str) -> str:
                            x = x.replace('\n', ' ').strip()
                            return x if len(x) <= 140 else (x[:137] + '…')
                        print(" | ".join([_shorten(s) for s in stories[:2]]) + (" …" if len(stories) > 2 else ""))
                except Exception:
                    pass
                try:
                    # Repetition policy (most common pairs)
                    from collections import Counter
                    pairs = []
                    for it in (plan_items or []):
                        rp = it.get('repetition_policy') if isinstance(it, dict) else None
                        if isinstance(rp, dict):
                            for k, v in rp.items():
                                if isinstance(k, str):
                                    pairs.append(f"{k}={v}")
                    if pairs:
                        cnt = Counter(pairs)
                        top = ", ".join([f"{p}({c})" for p, c in cnt.most_common(3)])
                        print(Style.BRIGHT + "[Plan Repetition]" + Style.RESET_ALL)
                        print(top)
                except Exception:
                    pass
                try:
                    # Lyrics prefs (aggregate most common values)
                    from collections import Counter
                    kvs = []
                    for it in (plan_items or []):
                        lp = it.get('lyrics_prefs') if isinstance(it, dict) else None
                        if isinstance(lp, dict) and lp:
                            wpb = lp.get('target_wpb'); mb = lp.get('melisma_bias'); mwb = lp.get('min_word_beats'); an = lp.get('allow_nonsense')
                            if isinstance(wpb, (int, float)):
                                kvs.append(f"wpb={round(float(wpb),2)}")
                            if isinstance(mb, (int, float)):
                                kvs.append(f"melisma_bias={round(float(mb),2)}")
                            if isinstance(mwb, (int, float)):
                                kvs.append(f"min_word_beats={round(float(mwb),2)}")
                            if isinstance(an, (int, float)):
                                kvs.append(f"allow_nonsense={int(an)}")
                    if kvs:
                        cnt = Counter(kvs)
                        top = ", ".join([f"{p}({c})" for p, c in cnt.most_common(4)])
                        print(Style.BRIGHT + "[Plan Lyrics Prefs]" + Style.RESET_ALL)
                        print(top)
                except Exception:
                    pass
                try:
                    # Palettes (imagery/verbs) aggregate
                    imagery = []
                    verbs = []
                    seen_i = set(); seen_v = set()
                    for it in (plan_items or []):
                        ip = it.get('imagery_palette') if isinstance(it, dict) else None
                        if isinstance(ip, list):
                            for s in ip:
                                if isinstance(s, str):
                                    t = s.strip()
                                    if t and t not in seen_i:
                                        seen_i.add(t); imagery.append(t)
                        vp = it.get('verb_palette') if isinstance(it, dict) else None
                        if isinstance(vp, list):
                            for s in vp:
                                if isinstance(s, str):
                                    t = s.strip()
                                    if t and t not in seen_v:
                                        seen_v.add(t); verbs.append(t)
                    if imagery:
                        print(Style.BRIGHT + "[Plan Imagery]" + Style.RESET_ALL)
                        print(", ".join(imagery[:5]) + (" …" if len(imagery) > 5 else ""))
                    if verbs:
                        print(Style.BRIGHT + "[Plan Verbs]" + Style.RESET_ALL)
                        print(", ".join(verbs[:5]) + (" …" if len(verbs) > 5 else ""))
                except Exception:
                    pass
                try:
                    # Call-and-response / Chant spots
                    cars = []
                    chants = []
                    for it in (plan_items or []):
                        ca = it.get('call_and_response') if isinstance(it, dict) else None
                        if isinstance(ca, str) and ca.strip():
                            cars.append(ca.strip())
                        chs = it.get('chant_spots') if isinstance(it, dict) else None
                        if isinstance(chs, list):
                            for s in chs:
                                if isinstance(s, str) and s.strip():
                                    chants.append(s.strip())
                    if cars:
                        uniq = []
                        seen_c = set()
                        for c in cars:
                            if c not in seen_c:
                                seen_c.add(c); uniq.append(c)
                        print(Style.BRIGHT + "[Plan Call&Response]" + Style.RESET_ALL)
                        print(", ".join(uniq[:3]) + (" …" if len(uniq) > 3 else ""))
                    if chants:
                        print(Style.BRIGHT + "[Plan Chant Spots]" + Style.RESET_ALL)
                        print(", ".join(chants[:3]) + (" …" if len(chants) > 3 else ""))
                except Exception:
                    pass
                except Exception:
                    # hard fail: honor user's request (no fallback)
                    try:
                        print(Fore.RED + Style.BRIGHT + "[Plan ERROR] Planning failed; aborting lyric generation for this run." + Style.RESET_ALL)
                    except Exception:
                        pass
                    raise RuntimeError("Planning failed (no fallback)")
                # Final safety normalization + fallback hints at call site (ensures visible effect)
                try:
                    if plan_items:
                        # Build quick feature arrays
                        labels_lc = [str(s.get('label','')).lower() for s in summaries]
                        dens = [float(s.get('notes_density', 0.0)) for s in summaries]
                        sust = [float(s.get('sustain_ratio', 0.0)) for s in summaries]
                        # Extract roles
                        roles = []
                        for i in range(len(summaries)):
                            it = next((p for p in plan_items if int(p.get('idx',-1)) == i), None)
                            roles.append(str((it or {}).get('role','verse')).lower())
                        n = len(roles)
                        # Map keywords to roles
                        for i in range(n):
                            lab = labels_lc[i]
                            if any(k in lab for k in ('drop','chorus','hook')):
                                roles[i] = 'chorus'
                            elif any(k in lab for k in ('build','buildup','rise','pre','lift')) and roles[i] not in ('chorus','silence'):
                                roles[i] = 'prechorus'
                        # Ensure at least two choruses by density
                        ch_idx = [i for i,r in enumerate(roles) if r=='chorus']
                        if len(ch_idx) < 2 and n >= 4:
                            order = sorted(range(n), key=lambda i: dens[i], reverse=True)
                            for i in order:
                                if roles[i] != 'chorus':
                                    roles[i] = 'chorus'; ch_idx.append(i)
                                    if len(ch_idx) >= 2:
                                        break
                        # Set prechorus before each chorus when possible
                        for ci in ch_idx:
                            j = ci-1
                            if 0 <= j < n and roles[j] not in ('chorus','prechorus','silence') and not summaries[j].get('silent'):
                                roles[j] = 'prechorus'
                        # Constrain vowels usage
                        for i in range(n):
                            if roles[i]=='vowels' and sust[i] < 0.5:
                                roles[i] = 'verse'
                        # Write roles back
                        for i in range(n):
                            it = next((p for p in plan_items if int(p.get('idx',-1)) == i), None)
                            if isinstance(it, dict):
                                it['role'] = roles[i]
                        # Fallback plan_hint templates if missing
                        def _hint_for(role: str) -> str:
                            r = role.lower()
                            if r == 'silence':
                                return "Instrumental only. Maintain vocal silence."
                            if r == 'chorus':
                                return "Deliver the title-drop clearly; keep 2–4 short lines and reuse them almost verbatim; add at most one global tag word."
                            if r == 'prechorus':
                                return "Tighten phrasing and raise tension; hint the title-drop without using it; end with a lift into the chorus."
                            if r == 'verse':
                                return "Advance the story with concrete imagery and active verbs; avoid chorus wording; set up the next section."
                            if r == 'bridge':
                                return "Provide contrast with a fresh angle or twist; add color, avoid chorus wording; prepare a shift."
                            if r == 'breakdown':
                                return "Reduce density and spotlight a single image or feeling; minimal new wording; leave space."
                            if r == 'backing':
                                return "Echo or answer the lead with short, repeatable fragments; avoid introducing new content."
                            if r == 'scat':
                                return "Use musical syllables supporting the groove; vary vowels; avoid monotony and full words."
                            if r == 'vowels':
                                return "Hold open vowels on long notes; no semantics; pure sustain for lift."
                            if r == 'intro':
                                return "Set the tone with one striking image; keep language sparse and intriguing."
                            if r == 'outro':
                                return "Resolve or fade with a small callback (title/tag); minimal new wording."
                            return "Keep the lyric clear and singable; avoid clutter; support the section function."
                        for i in range(n):
                            it = next((p for p in plan_items if int(p.get('idx',-1)) == i), None)
                            if isinstance(it, dict):
                                if not str(it.get('plan_hint','')).strip():
                                    it['plan_hint'] = _hint_for(str(it.get('role','verse')))
                except Exception:
                    pass
                # Build index map before printing to avoid stale/undefined lookups
                try:
                    plan_by_idx = {int(it.get('idx', i)): it for i, it in enumerate(plan_items) if isinstance(it, dict)}
                except Exception:
                    plan_by_idx = {}
                # Print plan for user visibility
                try:
                    if plan_items:
                        print(Style.BRIGHT + "\n[Vocal Plan]" + Style.RESET_ALL)
                        for s in summaries:
                            idx = int(s.get('idx', 0))
                            lab = s.get('label', f"Part_{idx+1}")
                            it = None
                            try:
                                # Use normalized map to avoid showing stale/contradictory role/hint
                                it = plan_by_idx.get(idx)
                                if not it:
                                    it = next((p for p in plan_items if int(p.get('idx', -1)) == idx), None)
                            except Exception:
                                it = None
                            role = str((it or {}).get('role', '') or 'silence')
                            hint = str((it or {}).get('plan_hint', '') or '')
                            hook = str((it or {}).get('hook_theme', '') or '')
                            hook_can = None
                            try:
                                hc = (it or {}).get('hook_canonical')
                                hook_can = (str(hc) if isinstance(hc, str) else None)
                            except Exception:
                                hook_can = None
                            line = f"{idx+1:02d}. {lab}: role={role}"
                            if hint:
                                line += f", hint={hint}"
                            if hook:
                                line += f", hook={hook}"
                            if hook_can:
                                line += f", hook_canonical=\"{hook_can}\""
                            print(line)
                            # extra details per item (full hint + prefs)
                            try:
                                if it and hint:
                                    print(Style.DIM + "    plan_hint: " + (hint if len(hint)<=300 else (hint[:300] + "…")) + Style.RESET_ALL)
                                lp = (it or {}).get('lyrics_prefs') if isinstance((it or {}).get('lyrics_prefs'), dict) else None
                                if lp:
                                    wpb = lp.get('target_wpb'); mb = lp.get('melisma_bias'); mwb = lp.get('min_word_beats'); an = lp.get('allow_nonsense')
                                    print(Style.DIM + "    prefs: " +
                                          (f"wpb={wpb} " if wpb is not None else "") +
                                          (f"melisma_bias={mb} " if mb is not None else "") +
                                          (f"min_word_beats={mwb} " if mwb is not None else "") +
                                          (f"allow_nonsense={an}" if an is not None else "")
                                          + Style.RESET_ALL)
                            except Exception:
                                pass
                        print()
                except Exception:
                    pass


                history_lines: List[str] = []
                # Optional: limit parts per run to reduce request bursts (set 0 or omit to disable)
                parts_limit = int(config.get('lyrics_parts_per_run', 0) or 0)
                # Process ALL parts from loaded JSON (including silence parts)
                total_parts = len(out_themes) if out_themes else 0
                for part_idx in range(total_parts):
                    th = out_themes[part_idx] if part_idx < len(out_themes) else {}
                    if part_idx < resume_idx:
                        continue
                    if parts_limit > 0 and part_idx >= parts_limit:
                        try:
                            print(Fore.YELLOW + f"Stopping after {parts_limit} parts this run (lyrics_parts_per_run)." + Style.RESET_ALL)
                        except Exception:
                            pass
                        break
                    # Get theme for this part (may not exist)
                    th = out_themes[part_idx] if part_idx < len(out_themes) else {}
                    label = th.get('label', f'Part_{part_idx+1}')
                    desc = th.get('description', '')
                    # Ensure 'tracks' key exists in the theme dictionary
                    if 'tracks' not in th:
                        th['tracks'] = []
                    trks = th['tracks']
                    if new_vocal_track_mode:
                        # create a new vocal track contextually for this part
                        # Build simple anchors from existing tracks: downbeats with high activity, silence windows, phrase suggestions
                        try:
                            bpb = int(ts.get('beats_per_bar', 4))
                        except Exception:
                            bpb = 4
                        # Count onsets per bar across other tracks
                        bar_activity = {}
                        max_bar = theme_len_bars
                        for t in trks:
                            for n in (t.get('notes', []) or []):
                                try:
                                    sb = float(n.get('start_beat', 0.0))
                                    bar = int(sb // max(1, bpb))
                                    bar_activity[bar] = bar_activity.get(bar, 0) + 1
                                except Exception:
                                    continue
                        # Choose top downbeat targets
                        bars_sorted = sorted(range(max_bar), key=lambda k: bar_activity.get(k, 0), reverse=True)
                        downbeat_targets = bars_sorted[: min(4, max_bar)]
                        # Silence windows (contiguous bars with 0 activity)
                        silence_windows = []
                        cur = []
                        for b in range(max_bar):
                            if bar_activity.get(b, 0) == 0:
                                if not cur:
                                    cur = [b, b]
                                else:
                                    cur[1] = b
                            else:
                                if cur:
                                    if cur[1] >= cur[0]:
                                        silence_windows.append([cur[0], cur[1]])
                                    cur = []
                        if cur and cur[1] >= cur[0]:
                            silence_windows.append([cur[0], cur[1]])
                        # Phrase windows around active bars (simple expansion by 1 bar)
                        phrase_windows = []
                        for b in downbeat_targets:
                            s = max(0, b-1); e = min(max_bar-1, b+1)
                            phrase_windows.append([s, e])
                        anchors_text = f"[Anchors] downbeats={downbeat_targets}; phrase_windows={phrase_windows}; silence_windows={silence_windows}"
                        # Inject planned role/hints/hook into section_description to ensure prompt sees it
                        try:
                            p = plan_by_idx.get(part_idx)
                            plan_bits = []
                            if isinstance(p, dict):
                                prole = str(p.get('role', '')).strip()
                                phint = str(p.get('plan_hint', '')).strip()
                                # Soft sanitize: don't pass instrumental-only hints for non-silence roles
                                try:
                                    phl = phint.lower()
                                except Exception:
                                    phl = ''
                                if prole.lower() != 'silence' and any(k in phl for k in (
                                    'instrumental only', 'maintain vocal silence', 'no vocal', 'no vocals', 'no vocal presence', 'purely instrumental'
                                )):
                                    phint = ''
                                phook = str(p.get('hook_theme', '')).strip()
                                phook_can = str(p.get('hook_canonical','')).strip() if isinstance(p.get('hook_canonical'), str) else ''
                                if prole:
                                    plan_bits.append(f"role={prole}")
                                if phint:
                                    plan_bits.append(f"hint={phint}")
                                if phook:
                                    plan_bits.append(f"hook={phook}")
                                if phook_can:
                                    plan_bits.append(f"hook_canonical=\"{phook_can}\"")
                            plan_text = (" \n[Plan] " + ", ".join(plan_bits)) if plan_bits else ""
                        except Exception:
                            plan_text = ""
                        # Ensure role hint is explicitly embedded for the prompt (first token wins)
                        role_for_desc = ''
                        try:
                            role_for_desc = str((plan_by_idx.get(part_idx) or {}).get('role','')).strip()
                        except Exception:
                            role_for_desc = ''
                        role_line = (f"[Plan] role={role_for_desc}\n" if role_for_desc else "")
                        desc_part = (desc + "\n" + role_line + plan_text + "\n" + anchors_text).strip()
                        # Lyrics-first (two stages): 1) Free lyrics + syllables, 2) Compose notes
                        try:
                            target_wpb_seed = float((cfg or config).get('lyrics_target_words_per_bar', 2.0)) if isinstance((cfg or config).get('lyrics_target_words_per_bar'), (int, float)) else 2.0
                        except Exception:
                            target_wpb_seed = 2.0
                        grid = _build_temp_note_grid_for_lyrics(ts, theme_len_bars, target_wpb_seed, downbeat_targets, phrase_windows)
                        # Build a minimal seed context (other tracks only)
                        seed_context_basic = []
                        try:
                            for _j, _t in enumerate(trks):
                                try:
                                    _t_notes = sorted(_t.get('notes', []) or [], key=lambda n: float(n.get('start_beat', 0.0)))
                                except Exception:
                                    _t_notes = _t.get('notes', []) or []
                                seed_context_basic.append({
                                    "name": get_instrument_name(_t),
                                    "role": _t.get('role','complementary'),
                                    "notes": _t_notes
                                })
                            # Add prior vocal track history (notes + tokens) from previous parts
                            try:
                                prior_hist = []
                                for pk in range(0, part_idx):
                                    if 0 <= pk < len(out_themes):
                                        th_prev = out_themes[pk]
                                        tr_prev = None
                                        for tprev in (th_prev.get('tracks', []) or []):
                                            if tprev.get('__final_vocal__'):
                                                tr_prev = tprev; break
                                        if tr_prev is None:
                                            continue
                                        prev_notes = sorted(tr_prev.get('notes', []) or [], key=lambda n: float(n.get('start_beat', 0.0)))
                                        prev_tokens = tr_prev.get('lyrics', []) or []
                                        prior_hist.append({
                                            "part": pk+1,
                                            "label": th_prev.get('label', f'Part {pk+1}'),
                                            "notes": prev_notes,
                                            "tokens": prev_tokens
                                        })
                                if prior_hist:
                                    seed_context_basic.append({"name": "__PRIOR_VOCAL__", "role": "vocal_history", "history": prior_hist})
                            except Exception:
                                pass
                        except Exception:
                            seed_context_basic = []
                        # Seed history is the same aggregation we build later; if empty, pass ""
                        try:
                            history_text_seed = "\n".join(history_lines) if history_lines else ""
                        except Exception:
                            history_text_seed = ""
                        # STAGE 1: free-form lyrics with optional syllables (no grid binding)
                        # Role 'silence' handling controlled by config: skip or still call LLM
                        try:
                            planned_role = str((plan_by_idx.get(part_idx) or {}).get('role','')).lower()
                        except Exception:
                            planned_role = ''
                        skip_role_silence = bool((cfg or config).get('skip_role_silence_parts', 1))
                        # Hard skip also when plan_hint contains force_silence=1 (numeric) and config allows skipping
                        try:
                            planned_hint = str((plan_by_idx.get(part_idx) or {}).get('plan_hint','') or '')
                        except Exception:
                            planned_hint = ''
                        has_force_silence = ('force_silence=1' in planned_hint.replace(' ', ''))
                        
                        # Handle silence cases correctly
                        if has_force_silence:
                            # force_silence=1: Direct silence, no Step-1
                            print(Fore.CYAN + f"[Composer] '{label}': force_silence=1, treating as silence" + Style.RESET_ALL)
                            stage1 = {"words": [], "syllables": [], "arranger_note": "intentional silence: force_silence=1"}
                        elif planned_role == 'silence':
                            # role=silence: Let Step-1 decide (may confirm silence or produce content)
                            print(Fore.CYAN + f"[Composer] '{label}': role=silence, letting Step-1 decide" + Style.RESET_ALL)
                            try:
                                stage1 = _generate_lyrics_free_with_syllables(
                                    cfg or config, genre, inspiration, 'Vocal', bpm, ts,
                                    section_label=label, section_description=desc_part,
                                    context_tracks_basic=seed_context_basic, user_prompt=user_guidance,
                                    history_context=history_text_seed, theme_len_bars=theme_len_bars, cfg=cfg, part_idx=part_idx
                                )
                            except Exception as e:
                                # If Step-1 fails, accept silence for this part
                                print(f"{Fore.RED}❌ Lyrics generation failed for '{label}' (role=silence): {str(e)}" + Style.RESET_ALL)
                                stage1 = {"words": [], "syllables": [], "arranger_note": "intentional silence: stage1 failed"}
                        else:
                            # Normal role: Generate content with retry logic
                            stage1 = {}
                            words_temp = []
                            max_retries_s1 = 3
                            s1_ok = False
                            
                            for rtry_s1 in range(max_retries_s1):
                                try:
                                    stage1 = _generate_lyrics_free_with_syllables(
                                        cfg or config, genre, inspiration, 'Vocal', bpm, ts,
                                        section_label=label, section_description=desc_part,
                                        context_tracks_basic=seed_context_basic, user_prompt=user_guidance,
                                        history_context=history_text_seed, theme_len_bars=theme_len_bars, cfg=cfg, part_idx=part_idx
                                    )
                                    
                                    preview = stage1.get('preview', '')
                                    words_temp = stage1.get('words', [])

                                    if preview and isinstance(preview, str) and preview.strip() != '""' and preview.strip():
                                        # Wenn eine Vorschau existiert, MÜSSEN auch Wörter vorhanden sein.
                                        if isinstance(words_temp, list) and len(words_temp) > 0:
                                            s1_ok = True
                                            break
                                        else:
                                            print(Fore.YELLOW + f"[Composer] Stage-1 returned preview but no words for '{label}'. Retrying ({rtry_s1 + 1}/{max_retries_s1})." + Style.RESET_ALL)
                                    else:
                                        # Keine Vorschau, Stille ist beabsichtigt und OK.
                                        s1_ok = True
                                        break
                                except Exception as e:
                                    # If Stage-1 fails, retry unless this is the last attempt
                                    print(f"{Fore.RED}❌ Lyrics generation failed for '{label}' (attempt {rtry_s1 + 1}/{max_retries_s1}): {str(e)}" + Style.RESET_ALL)
                                    if rtry_s1 == max_retries_s1 - 1:
                                        stage1 = {"words": [], "syllables": [], "arranger_note": "intentional silence: stage1 failed"}
                                        s1_ok = True
                            
                            if not s1_ok:
                                print(Fore.RED + f"[Composer] Stage-1 failed for '{label}' after {max_retries_s1} retries. Treating as silence." + Style.RESET_ALL)
                                stage1 = {"words": [], "syllables": [], "arranger_note": "intentional silence: stage1 retries exhausted"}
                        # Extract key_scale from JSON for main processing - no fallback
                        if not cfg or 'key_scale' not in cfg:
                            print(f"{Fore.RED}❌ CRITICAL ERROR: key_scale not found in JSON!" + Style.RESET_ALL)
                            print(f"{Fore.RED}   This is ESSENTIAL for lyrics generation - we need the musical parameters from the JSON!" + Style.RESET_ALL)
                            continue
                        key_scale = cfg.get("key_scale")
                        
                        # Merge Stage-0 directives (no heuristics/fallbacks)
                        try:
                            if isinstance(stage1, dict):
                                # From Stage-0 analysis
                                try:
                                    ANALYSIS_CTX
                                except NameError:
                                    ANALYSIS_CTX = {}
                                if not stage1.get('hook_canonical') and isinstance(ANALYSIS_CTX, dict) and isinstance(ANALYSIS_CTX.get('hook_canonical'), str) and ANALYSIS_CTX.get('hook_canonical').strip():
                                    stage1['hook_canonical'] = ANALYSIS_CTX.get('hook_canonical').strip()
                                if (not stage1.get('chorus_lines')) and isinstance(ANALYSIS_CTX, dict) and isinstance(ANALYSIS_CTX.get('chorus_lines'), list) and ANALYSIS_CTX.get('chorus_lines'):
                                    stage1['chorus_lines'] = [str(x) for x in ANALYSIS_CTX.get('chorus_lines') if isinstance(x, str)][:4]
                                # Do not infer additional hooks from raw user prompt per user preference
                        except Exception:
                            pass
                        words = stage1.get('words') if isinstance(stage1, dict) else None
                        syllables = stage1.get('syllables') if isinstance(stage1, dict) else None
                        # If words exist but syllables missing/empty, derive simple 1:1 syllables
                        try:
                            if isinstance(words, list) and len(words) > 0 and (not isinstance(syllables, list) or len(syllables) == 0):
                                syllables = [[w] for w in words]
                        except Exception:
                            pass
                        arranger_note = stage1.get('arranger_note') if isinstance(stage1, dict) else None
                        # New: model-declared silence flag
                        try:
                            model_silence = bool(stage1.get('intentional_silence'))
                        except Exception:
                            model_silence = False
                        track_data = None
                        # Allow explicit silence for intro/outro/breakdown if provided as empty arrays
                        section_role_local = _normalize_section_role(label)
                        is_silence_allowed = section_role_local in ('intro', 'outro', 'breakdown')
                        # Only treat as intentional silence if it's explicitly marked as silence role or force_silence
                        has_intentional_silence = bool(model_silence) or (planned_role == 'silence' and isinstance(arranger_note, str) and ('intentional silence' in arranger_note.lower()))
                        silence_cfg = bool((cfg or config).get('silence_skipping_enabled', 1))
                        # Accept explicit silence only if planned silence or model explicitly requested and no words
                        if silence_cfg and (isinstance(words, list) and isinstance(syllables, list) and len(words) == 0 and len(syllables) == 0 and (planned_role == 'silence' or has_intentional_silence)):
                            track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": [], "lyrics": []}
                            try:
                                print(Style.DIM + f"[Composer] '{label}': accepted explicit silence for this section." + Style.RESET_ALL)
                            except Exception:
                                pass
                            # Add to history so the next parts see the silence context
                            try:
                                history_lines.append(f"{label}: [silence]")
                            except Exception:
                                pass
                        elif silence_cfg and (model_silence and not (isinstance(words, list) and len(words) > 0)):
                            # Model explicitly requested silence → accept directly
                            try:
                                print(Style.DIM + f"[Composer] '{label}': model-declared intentional silence." + Style.RESET_ALL)
                            except Exception:
                                pass
                            track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": [], "lyrics": []}
                            try:
                                history_lines.append(f"{label}: [silence]")
                            except Exception:
                                pass
                        elif silence_cfg and not (isinstance(words, list) and words):
                            # Check if this is a silence role - if so, don't retry
                            # Use planned_role from the vocal plan, not the label
                            is_silence_role = planned_role == 'silence'
                            
                            if is_silence_role:
                                # For silence roles, accept empty words as intentional silence
                                print(Fore.YELLOW + f"[Composer] Stage-1 returned no words for '{label}' (silence role). Treating as intentional silence." + Style.RESET_ALL)
                                words = []
                                syllables = []
                                arranger_note = "intentional silence"
                            else:
                                # For non-silence roles, retry until we get a result
                                print(Fore.YELLOW + f"[Composer] Stage-1 returned no words for '{label}' (non-silence role). Retrying until success..." + Style.RESET_ALL)
                                
                                # CRITICAL: Extract ALL required parameters from JSON first
                                if not cfg:
                                    print(f"{Fore.RED}❌ CRITICAL ERROR: No cfg data available!" + Style.RESET_ALL)
                                    continue
                                
                                # Extract musical parameters from JSON
                                key_scale = cfg.get("key_scale")
                                genre = cfg.get("genre", "electronic")
                                inspiration = cfg.get("inspiration", "")
                                bpm = cfg.get("bpm", 120)
                                ts = cfg.get("time_signature", {"beats_per_bar": 4, "beat_value": 4})
                                
                                if not key_scale:
                                    print(f"{Fore.RED}❌ CRITICAL ERROR: key_scale not found in JSON!" + Style.RESET_ALL)
                                    continue
                                
                                # Ensure API keys are initialized
                                if not API_KEYS:
                                    print(Fore.CYAN + "Initializing API keys for retry..." + Style.RESET_ALL)
                                    if not initialize_api_keys(cfg):
                                        print(Fore.RED + "No API keys available for retry!" + Style.RESET_ALL)
                                        continue
                                    genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                                
                                # Retry loop - keep trying until we get words
                                max_retries = 5
                                retry_count = 0
                                success = False
                                
                                while retry_count < max_retries and not success:
                                    retry_count += 1
                                    print(Fore.CYAN + f"[Composer] Retry attempt {retry_count}/{max_retries} for '{label}'..." + Style.RESET_ALL)
                                    
                                    try:
                                        retry_result = _generate_lyrics_free_with_syllables(
                                            cfg, genre, inspiration, 'Vocal', bpm, ts,
                                            section_label=label, section_description=desc_part,
                                            context_tracks_basic=seed_context_basic, user_prompt="Generate meaningful lyrics",
                                            history_context=history_text_seed, theme_len_bars=theme_len_bars, cfg=cfg, key_scale=key_scale, part_idx=part_idx
                                        )
                                        words = retry_result.get('words', [])
                                        syllables = retry_result.get('syllables', [])
                                        arranger_note = retry_result.get('arranger_note', '')
                                        
                                        # If syllables missing but words present, derive simple syllables 1:1
                                        if (not isinstance(syllables, list) or len(syllables) == 0) and isinstance(words, list) and len(words) > 0:
                                            try:
                                                syllables = [[w] for w in words]
                                            except Exception:
                                                syllables = []
                                        if words and syllables and len(words) > 0:
                                            # Create preview of generated lyrics
                                            words_preview = " ".join(words[:8])  # First 8 words
                                            if len(words) > 8:
                                                words_preview += "..."
                                            print(Fore.GREEN + f"[Composer] Retry {retry_count} successful for '{label}' - got {len(words)} words" + Style.RESET_ALL)
                                            print(Fore.MAGENTA + Style.BRIGHT + f"  📝 LYRICS PREVIEW: \"{words_preview}\"" + Style.RESET_ALL)
                                            success = True
                                        else:
                                            print(Fore.YELLOW + f"[Composer] Retry {retry_count} returned empty words for '{label}'" + Style.RESET_ALL)
                                            if retry_count < max_retries:
                                                print(Fore.CYAN + f"[Composer] Waiting 2 seconds before next retry..." + Style.RESET_ALL)
                                                time.sleep(2)
                                            
                                    except Exception as e:
                                        print(Fore.RED + f"[Composer] Retry {retry_count} failed for '{label}': {e}" + Style.RESET_ALL)
                                        if retry_count < max_retries:
                                            print(Fore.CYAN + f"[Composer] Waiting 2 seconds before next retry..." + Style.RESET_ALL)
                                            time.sleep(2)
                                
                                # If all retries failed, use fallback
                                if not success:
                                    print(Fore.RED + f"[Composer] All {max_retries} retries failed for '{label}'. Using fallback." + Style.RESET_ALL)
                                    words = ["flow", "through", "time"]
                                    syllables = [["flow"], ["through"], ["time"]]
                                    arranger_note = "fallback lyrics after retries"
                        elif not new_vocal_track_mode:
                            # For existing tracks, treat as intentional silence
                            try:
                                print(Fore.YELLOW + Style.BRIGHT + f"[Composer] Stage-1 returned no words for '{label}'. Treating as intentional silence for this part." + Style.RESET_ALL)
                            except Exception:
                                pass
                            track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": [], "lyrics": []}
                            try:
                                history_lines.append(f"{label}: [silence]")
                            except Exception:
                                pass
                        if not track_data:
                            # For new vocal tracks, generate lyrics first if not already done
                            if new_vocal_track_mode:
                                # Check if words and syllables are already generated
                                if 'words' not in locals() or 'syllables' not in locals():
                                    print(f"{Fore.CYAN}🎵 Generating lyrics for '{label}' (lyrics-first approach){Style.RESET_ALL}")
                                    
                                    # Extract musical parameters from JSON (not config.yaml)
                                    # DEBUG: Show what we're working with
                                    print(f"[DEBUG] cfg keys: {list(cfg.keys()) if cfg else 'None'}")
                                    print(f"[DEBUG] cfg key_scale: {cfg.get('key_scale') if cfg else 'None'}")
                                    print(f"[DEBUG] cfg genre: {cfg.get('genre') if cfg else 'None'}")
                                    
                                    # DIRECT extraction from JSON - bypass _get_config_value
                                    # Get the actual selected artifact data that was loaded
                                    progress_data = None
                                    if selected_path and os.path.exists(selected_path):
                                        progress_data = _load_progress_silent(selected_path)
                                        print(f"[DEBUG] Loading Selected Artifact: {selected_path}")
                                        print(f"[DEBUG] JSON config key_scale: {progress_data.get('config', {}).get('key_scale', 'NOT_FOUND')}")
                                        print(f"[DEBUG] JSON config root_note: {progress_data.get('config', {}).get('root_note', 'NOT_FOUND')}")
                                        print(f"[DEBUG] JSON config scale_type: {progress_data.get('config', {}).get('scale_type', 'NOT_FOUND')}")
                                    else:
                                        print(f"[ERROR] Selected artifact not found: {selected_path}")
                                        continue
                                    
                                    # CRITICAL: Extract key_scale for lyrics generation
                                    if progress_data and 'config' in progress_data:
                                        stored_config = progress_data['config']
                                        key_scale = stored_config.get('key_scale', 'C major')
                                        genre = stored_config.get('genre', 'electronic')
                                        inspiration = stored_config.get('inspiration', '')
                                        bpm = stored_config.get('bpm', 120)
                                        ts = stored_config.get('time_signature', {'beats_per_bar': 4, 'beat_value': 4})
                                        print(f"[DEBUG] EXTRACTED from JSON - key_scale: {key_scale}, genre: {genre}, bpm: {bpm}")
                                    else:
                                        # CRITICAL ERROR: We NEED the data from the selected JSON!
                                        print(f"{Fore.RED}❌ CRITICAL ERROR: Could not load progress data from JSON file!" + Style.RESET_ALL)
                                        print(f"{Fore.RED}   This is ESSENTIAL for lyrics generation - we need the musical parameters from the JSON!" + Style.RESET_ALL)
                                        print(f"{Fore.RED}   Skipping lyrics generation for '{label}'" + Style.RESET_ALL)
                                        words = []
                                        syllables = []
                                        arranger_note = "ERROR: Could not load JSON data"
                                        continue
                                    
                                    
                                    # Ensure API keys are initialized
                                    if not API_KEYS:
                                        print(Fore.CYAN + "Initializing API keys for lyrics generation..." + Style.RESET_ALL)
                                        if initialize_api_keys(cfg or config):
                                            genai.configure(api_key=API_KEYS[CURRENT_KEY_INDEX])
                                            print(Fore.GREEN + f"API keys initialized: {len(API_KEYS)} key(s) available" + Style.RESET_ALL)
                                        else:
                                            print(Fore.RED + "No API keys available for lyrics generation!" + Style.RESET_ALL)
                                            words = []
                                            syllables = []
                                            arranger_note = "API unavailable"
                                        if not words:
                                            continue
                                    
                                    try:
                                        lyrics_result = _generate_lyrics_free_with_syllables(
                                            cfg or config, genre, inspiration, 'Vocal', bpm, ts,
                                            section_label=label, section_description=desc_part,
                                            context_tracks_basic=seed_context_basic, user_prompt=user_guidance,
                                            history_context=history_text_seed, theme_len_bars=theme_len_bars, cfg=cfg, part_idx=part_idx
                                        )
                                        words = lyrics_result.get('words', [])
                                        syllables = lyrics_result.get('syllables', [])
                                        arranger_note = lyrics_result.get('arranger_note', '')
                                    except Exception as e:
                                        print(f"{Fore.YELLOW}Lyrics generation failed for '{label}': {e}{Style.RESET_ALL}")
                                        words = []
                                        syllables = []
                                        arranger_note = "intentional silence: lyrics generation failed"
                                
                                print(f"{Fore.BLUE}  🎼 Composing notes for '{label}'{Style.RESET_ALL}")
                                # Use the generated lyrics to compose notes
                                if words and syllables:
                                    # Extract musical parameters from JSON (not config.yaml)
                                    # Get the actual selected artifact data that was loaded
                                    progress_data = None
                                    if selected_path and os.path.exists(selected_path):
                                        progress_data = _load_progress_silent(selected_path)
                                        print(f"[DEBUG] Loading Selected Artifact for notes: {selected_path}")
                                        print(f"[DEBUG] JSON config key_scale: {progress_data.get('config', {}).get('key_scale', 'NOT_FOUND')}")
                                        print(f"[DEBUG] JSON config root_note: {progress_data.get('config', {}).get('root_note', 'NOT_FOUND')}")
                                        print(f"[DEBUG] JSON config scale_type: {progress_data.get('config', {}).get('scale_type', 'NOT_FOUND')}")
                                    else:
                                        print(f"[ERROR] Selected artifact not found for notes: {selected_path}")
                                        continue
                                    
                                    if progress_data and 'config' in progress_data:
                                        stored_config = progress_data['config']
                                        key_scale = stored_config.get('key_scale', 'C major')
                                        genre = stored_config.get('genre', 'electronic')
                                        inspiration = stored_config.get('inspiration', '')
                                        bpm = stored_config.get('bpm', 120)
                                        ts = stored_config.get('time_signature', {'beats_per_bar': 4, 'beat_value': 4})
                                    else:
                                        # CRITICAL ERROR: We NEED the data from the selected JSON!
                                        print(f"{Fore.RED}❌ CRITICAL ERROR: Could not load progress data from JSON file!" + Style.RESET_ALL)
                                        print(f"{Fore.RED}   This is ESSENTIAL for note composition - we need the musical parameters from the JSON!" + Style.RESET_ALL)
                                        print(f"{Fore.RED}   Skipping note composition for '{label}'" + Style.RESET_ALL)
                                        stage2 = {"notes": [], "error": "Could not load JSON data"}
                                        continue
                                    
                                    # Only define hook_canonical if it exists (for hook-based genres)
                                    # Don't use fallback - if not defined, work without it
                                    hook_canonical = stored_config.get('hook_canonical') if stored_config else None
                                    
                                    # Define other optional variables - no fallbacks for optional parameters
                                    chorus_lines = stored_config.get('chorus_lines', []) if stored_config else []
                                    section_description = desc_part
                                    
                                    stage2 = _compose_notes_for_syllables(
                                        cfg or config, genre, inspiration, 'Vocal', bpm, ts, theme_len_bars,
                                        syllables, arranger_note, seed_context_basic, key_scale, label, hook_canonical, 
                                        chorus_lines, section_description, words, cfg
                                    )
                                else:
                                    # Fallback to minimal content - use the same cfg data that was already loaded
                                    if not cfg:
                                        print(f"{Fore.RED}❌ CRITICAL ERROR: No cfg data available for fallback!" + Style.RESET_ALL)
                                        stage2 = {"notes": [], "error": "No cfg data"}
                                    else:
                                        # Extract musical parameters from the already loaded cfg
                                        key_scale = cfg.get("key_scale")
                                        genre = cfg.get("genre", "electronic")
                                        inspiration = cfg.get("inspiration", "")
                                        bpm = cfg.get("bpm", 120)
                                        ts = cfg.get("time_signature", {"beats_per_bar": 4, "beat_value": 4})
                                        
                                        if not key_scale:
                                            print(f"{Fore.RED}❌ CRITICAL ERROR: key_scale not found in cfg!" + Style.RESET_ALL)
                                            stage2 = {"notes": [], "error": "No key_scale"}
                                        else:
                                            # Only define hook_canonical if it exists (for hook-based genres)
                                            hook_canonical = cfg.get('hook_canonical')
                                            
                                            # Define other optional variables - no fallbacks for optional parameters
                                            chorus_lines = cfg.get('chorus_lines', [])
                                            section_description = desc_part
                                            
                                            stage2 = _compose_notes_for_syllables(
                                                cfg, genre, inspiration, 'Vocal', bpm, ts, theme_len_bars,
                                                [["ah"]], arranger_note, seed_context_basic, key_scale, label, hook_canonical, 
                                                chorus_lines, section_description, ["ah"], cfg
                                            )
                                # Set notes2 and tokens2 for new vocal tracks
                                if isinstance(stage2, dict) and 'notes' in stage2:
                                    notes2 = stage2.get('notes', [])
                                    tokens2 = stage2.get('tokens', []) if isinstance(stage2.get('tokens'), list) else []
                                    # Align tokens with notes by padding with '-' continuations if needed
                                    # Do not pad tokens with '-' anymore; enforce strict validity instead
                                else:
                                    notes2 = []
                                    tokens2 = []
                                # Check if we have valid content
                                valid_len = (isinstance(notes2, list) and len(notes2) > 0) or (isinstance(tokens2, list) and len(tokens2) > 0)
                                if valid_len:
                                    track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": notes2, "lyrics": tokens2, "__final_vocal__": True}
                                    try:
                                        _lp = " ".join((tokens2 or [])[:10])
                                        if isinstance(tokens2, list) and len(tokens2) > 10:
                                            _lp += "..."
                                        print(Fore.GREEN + Style.BRIGHT + f"  ✅ {label}: notes={len(notes2)} | words={len(tokens2 or [])}" + Style.RESET_ALL)
                                        if _lp:
                                            print(Fore.MAGENTA + "     📝 " + _lp + Style.RESET_ALL)
                                    except Exception:
                                        pass
                                    # Set notes and tokens for later use
                                    notes = notes2
                                    tokens = tokens2
                                else:
                                    print(f"{Fore.RED}  ❌ Failed to generate content for '{label}'{Style.RESET_ALL}")
                                    track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": [], "lyrics": [], "__final_vocal__": True}
                                    # Set empty notes and tokens for later use
                                    notes = []
                                    tokens = []
                            else:
                                # For existing tracks, use the original Stage-2 logic
                                pass
                            planned_role = plan_by_idx.get(part_idx, {}).get('role', '')
                            role_lower = str(planned_role).lower().strip()
                            # STAGE 2: compose notes for syllables (skip if planned silence or contradictory role+hint)
                            planned_hint = plan_by_idx.get(part_idx, {}).get('plan_hint', '')
                            hint_lower = str(planned_hint).lower().strip()
                            is_contradictory = (role_lower == 'chorus' and any(k in hint_lower for k in (
                                'instrumental only', 'maintain vocal silence', 'no vocal', 'no vocals', 'no vocal presence', 'purely instrumental'
                            )))
                            
                            # ======================================================================
                            # == KONSISTENZPRÜFUNG: STAGE 0 (PLAN) vs STAGE 1 (TEXT)
                            # ======================================================================
                            # Wenn Stage 1 Text generiert hat, obwohl Stage 0 'silence' geplant hat,
                            # wird die Rolle hier überschrieben, um Noten in Stage 2 zu erzwingen.
                            # Dies stellt sicher, dass generierter Text niemals ohne Noten bleibt.
                            if section_role_local == 'silence' and isinstance(words, list) and len(words) > 0:
                                print(Fore.YELLOW + Style.BRIGHT + f"[Logic Correction] Lyrics were generated for a 'silence' part ('{label}'). Overriding role to 'verse' to ensure notes are composed." + Style.RESET_ALL)
                                section_role_local = 'verse'
                                # Da wir die Rolle geändert haben, müssen wir auch 'model_silence' auf False setzen,
                                # um zu verhindern, dass die nachfolgende Logik fälschlicherweise den Silence-Pfad wählt.
                                model_silence = False
                                # Aktualisiere auch role_lower für die nachfolgende Prüfung
                                role_lower = 'verse'
                            # ======================================================================
                            
                            if role_lower == 'silence' or is_contradictory or model_silence:
                                # For silence roles, generate minimal atmospheric notes instead of complete silence
                                print(f"[DEBUG] Silence role detected: {role_lower}, generating minimal atmospheric content")
                                stage2 = _compose_notes_for_syllables(
                                    cfg or config, genre, inspiration, 'Vocal', bpm, ts, theme_len_bars,
                                    [["ah"]], arranger_note, seed_context_basic, key_scale, label, hook_canonical, 
                                    chorus_lines, section_description, ["ah"], cfg
                                )
                                # Mark as minimal silence content
                                if isinstance(stage2, dict) and 'notes' in stage2:
                                    print(f"[DEBUG] Silence role generated {len(stage2.get('notes', []))} notes")
                                    for note in stage2.get('notes', []):
                                        note['silence_role'] = True
                                        print(f"[DEBUG] Silence note: pitch={note.get('pitch')}, duration={note.get('duration_beats')}")
                            else:
                                # Extract key_scale safely
                                key_scale = cfg.get('key_scale') if cfg else config.get('key_scale', 'C major')
                                stage2 = _compose_notes_for_syllables(
                                    cfg or config, genre, inspiration, 'Vocal', bpm, ts, theme_len_bars,
                                    (stage1.get('syllables') if isinstance(stage1.get('syllables'), list) else []), arranger_note, seed_context_basic, key_scale, section_label=label,
                                    hook_canonical=(stage1.get('hook_canonical') if isinstance(stage1.get('hook_canonical'), str) else None),
                                    chorus_lines=(stage1.get('chorus_lines') if isinstance(stage1.get('chorus_lines'), list) else None),
                                    section_description=(plan_by_idx.get(part_idx, {}) or {}).get('plan_hint') or desc_part, lyrics_words=(words or []), cfg=cfg
                                )
                                # For existing tracks, continue with the original logic
                            # Enforce role-based timing to slow down overly hasty mapping
                            notes2 = stage2.get('notes') if isinstance(stage2, dict) else None
                            tokens2 = stage2.get('tokens') if isinstance(stage2, dict) else None
                            # Do not pad tokens with '-' anymore; enforce strict validity instead
                            try:
                                if isinstance(notes2, list):
                                    notes2 = _enforce_role_timing_constraints(notes2, int(ts.get('beats_per_bar',4)), _normalize_section_role(label), part_idx, cfg or config)
                            except Exception:
                                pass
                            # ==========================================================================================
                            # == VEREINFACHTE STAGE-2 VALIDIERUNG
                            # == Goldene Regel: Wenn Text existiert, müssen Noten existieren und 1:1 passen.
                            # ==========================================================================================
                            
                            # Prüfe das Ergebnis gegen die Regeln.
                            valid_len = False
                            if isinstance(words, list) and len(words) > 0:
                                # Wenn Text vorhanden ist, MÜSSEN Noten valide sein. Strenge Prüfung.
                                if (isinstance(notes2, list) and isinstance(tokens2, list) and 
                                    len(notes2) > 0 and len(notes2) == len(tokens2) and
                                    not any((str(t).strip() == '-' for t in tokens2[:1]))):
                                    valid_len = True
                            else:
                                # Wenn kein Text vorhanden ist (geplante Stille), ist ein leeres Ergebnis in Ordnung.
                                if isinstance(notes2, list) and isinstance(tokens2, list):
                                    valid_len = True
                            if valid_len:
                                # Ergebnis ist gültig, erstelle Track-Daten.
                                track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": notes2, "lyrics": tokens2, "__final_vocal__": True}
                                try:
                                    _lp2 = " ".join((tokens2 or [])[:10]) + ("..." if isinstance(tokens2, list) and len(tokens2) > 10 else "")
                                    print(Fore.GREEN + Style.BRIGHT + f"[Composer] Stage-2 OK: {label} | notes={len(notes2)} | words={len(tokens2 or [])}" + Style.RESET_ALL)
                                    if _lp2:
                                        print(Fore.MAGENTA + "     📝 " + _lp2 + Style.RESET_ALL)
                                except Exception:
                                    pass
                            else:
                                # Wenn ungültig, starte die Retry-Logik.
                                try:
                                    preview_debug = json.dumps({
                                        "words": (words[:6] if isinstance(words, list) else None),
                                        "notes2_len": (len(notes2) if isinstance(notes2, list) else type(notes2).__name__), 
                                        "tokens2_len": (len(tokens2) if isinstance(tokens2, list) else type(tokens2).__name__)
                                    })
                                    print(Fore.YELLOW + Style.BRIGHT + f"[Composer] Stage-2 invalid for '{label}'. {preview_debug}" + Style.RESET_ALL)
                                except Exception:
                                    pass
                                # Increase retries for invalid Stage-2 to ensure valid notes/tokens
                                try:
                                    max_retries_s2 = int((cfg or config).get('stage2_invalid_retries', 4))
                                except Exception:
                                    max_retries_s2 = 4
                                retry_ok = False
                                for rtry in range(max(0, max_retries_s2)):
                                    try:
                                        print(Fore.CYAN + f"[Composer] Retrying Stage-2 for '{label}' ({rtry+1}/{max_retries_s2})" + Style.RESET_ALL)
                                    except Exception:
                                        pass
                                    stage2 = _compose_notes_for_syllables(
                                        cfg or config, genre, inspiration, 'Vocal', bpm, ts, theme_len_bars,
                                        (stage1.get('syllables') if isinstance(stage1.get('syllables'), list) else []), arranger_note, context_tracks_basic=seed_context_basic, key_scale=(cfg or config).get('key_scale',''), section_label=label,
                                        hook_canonical=(stage1.get('hook_canonical') if isinstance(stage1.get('hook_canonical'), str) else None),
                                        chorus_lines=(stage1.get('chorus_lines') if isinstance(stage1.get('chorus_lines'), list) else None),
                                        section_description=(plan_by_idx.get(part_idx, {}) or {}).get('plan_hint') or desc_part, lyrics_words=(words or [])
                                    )
                                    notes2 = stage2.get('notes') if isinstance(stage2, dict) else None
                                    tokens2 = stage2.get('tokens') if isinstance(stage2, dict) else None
                                    # Prüfe das neue Ergebnis gegen dieselben Regeln wie oben.
                                    if isinstance(words, list) and len(words) > 0:
                                        if (isinstance(notes2, list) and isinstance(tokens2, list) and len(notes2) > 0 and len(notes2) == len(tokens2) and not any((str(t).strip() == '-' for t in tokens2[:1]))):
                                            retry_ok = True
                                    else:
                                        if isinstance(notes2, list) and isinstance(tokens2, list):
                                            retry_ok = True
                                    
                                    if retry_ok:
                                        # Ergebnis ist gültig.
                                        track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": notes2, "lyrics": tokens2, "__final_vocal__": True}
                                        try:
                                            _lp2 = " ".join((tokens2 or [])[:10]) + ("..." if isinstance(tokens2, list) and len(tokens2) > 10 else "")
                                            print(Fore.GREEN + Style.BRIGHT + f"[Composer] Stage-2 retry OK: {label} | notes={len(notes2)} | words={len(tokens2 or [])}" + Style.RESET_ALL)
                                            if _lp2:
                                                print(Fore.MAGENTA + "     📝 " + _lp2 + Style.RESET_ALL)
                                        except Exception:
                                            pass
                                        break
                                if not retry_ok:
                                    # Letzter Ausweg: 1:1 Synthese
                                    if isinstance(words, list) and len(words) > 0:
                                        print(Fore.YELLOW + f"[Composer] All Stage-2 retries failed for '{label}'. Synthesizing 1:1 notes as fallback." + Style.RESET_ALL)
                                        try:
                                            bpb_loc = int(ts.get('beats_per_bar', 4))
                                            total_beats = float(theme_len_bars) * float(bpb_loc)
                                            n_tok = len(words)
                                            step = total_beats / float(max(1, n_tok))
                                            grid_notes = [{"start_beat": float(i)*step} for i in range(n_tok)]
                                            kscale = (cfg or config).get('key_scale')
                                            notes2 = _synthesize_notes_from_tokens(words, grid_notes, ts, theme_len_bars, key_scale=kscale)
                                            tokens2 = words
                                            if not isinstance(notes2, list) or len(notes2) == 0: raise RuntimeError("1:1 synthesis returned empty notes")
                                            track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": notes2, "lyrics": tokens2, "__final_vocal__": True}
                                            print(Fore.GREEN + f"[Composer] Fallback synthesis OK: {label} | notes={len(notes2)} | words={len(tokens2)}" + Style.RESET_ALL)
                                        except Exception as e:
                                            print(Fore.RED + f"[Composer] Fallback synthesis failed for '{label}': {e}" + Style.RESET_ALL)
                                            track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": [], "lyrics": words, "__final_vocal__": True}
                                    else:
                                        # Für Stille-Parts oder wenn keine Worte da sind -> leerer Track
                                        track_data = {"instrument_name": 'Vocal', "program_num": 0, "role": 'vocal', "notes": [], "lyrics": [], "__final_vocal__": True}
                        trks.append(track_data)
                        # Explicitly save tracks back to out_themes to ensure persistence
                        out_themes[part_idx]['tracks'] = trks
                        target_tr = trks[-1]
                        track_idx_effective = len(trks) - 1
                        # Save progress after creating/overwriting vocal for this part
                        try:
                            progress_data = {
                                'type': 'lyrics_generation_new_track',
                                'config': cfg or config,
                                'themes': out_themes,
                                'length': theme_len_bars,
                                'current_theme_index': part_idx,
                                'current_track_index': track_idx_effective,
                                'generation_type': 'new_vocal',
                                'timestamp': run_timestamp
                            }
                            # Add plan data if available (only save once at the beginning)
                            if part_idx == 0:
                                progress_data.update({
                                    'user_guidance': user_guidance,
                                    'roles': roles,
                                    'plan_items': plan_items,
                                    'analysis_ctx': ANALYSIS_CTX
                                })
                            save_progress(progress_data, script_dir, run_timestamp)
                        except Exception:
                            pass
                    else:
                        if not (0 <= track_idx < len(trks)):
                            continue
                        target_tr = trks[track_idx]
                        track_idx_effective = track_idx
                    # If planned role is explicit silence, skip LLM/logging for this part immediately (no notes/lyrics for vocal)
                    try:
                        planned_role_now = str((plan_by_idx.get(part_idx) or {}).get('role','')).lower()
                    except Exception:
                        planned_role_now = ''
                    if planned_role_now == 'silence':
                        try:
                            print(Style.DIM + f"[Lyrics] '{label}': planned silence — generating minimal atmospheric content." + Style.RESET_ALL)
                        except Exception:
                            pass
                        try:
                            # Generate minimal atmospheric lyrics for silence roles
                            target_tr['lyrics'] = ["ah"]
                        except Exception:
                            pass
                        try:
                            history_lines.append(f"{label}: [minimal atmospheric]")
                        except Exception:
                            pass
                        continue
                    notes = sorted(target_tr.get('notes', []), key=lambda n: float(n.get('start_beat', 0.0)))
                    if not notes:
                        try:
                            print(Style.DIM + f"[Lyrics] Part {part_idx+1}/{len(out_themes)} '{label}': no notes, skipping." + Style.RESET_ALL)
                        except Exception:
                            pass
                        # Preserve context history so subsequent parts know this section was intentionally silent
                        try:
                            history_lines.append(f"{label}: [silence]")
                        except Exception:
                            pass
                        continue
                    # Provide full context for other tracks (name, role, full notes, prior lyrics) excluding the target track
                    context_basic = []
                    for j, t in enumerate(trks):
                        if j == track_idx_effective:
                            continue
                        try:
                            t_notes = sorted(t.get('notes', []) or [], key=lambda n: float(n.get('start_beat', 0.0)))
                        except Exception:
                            t_notes = t.get('notes', []) or []
                        # Build prior lyrics history for this other track across previous parts
                        lyrics_hist = []
                        try:
                            for k_idx in range(0, part_idx):
                                if 0 <= k_idx < len(out_themes):
                                    th_prev = out_themes[k_idx]
                                    trks_prev = th_prev.get('tracks', [])
                                    if 0 <= j < len(trks_prev):
                                        toks_prev = trks_prev[j].get('lyrics', []) or []
                                        if toks_prev:
                                            try:
                                                text_prev = ' '.join(_tokens_to_words(toks_prev))
                                            except Exception:
                                                text_prev = ' '.join([str(x) for x in toks_prev if isinstance(x, str)])
                                            if text_prev.strip():
                                                lyrics_hist.append({
                                                    "label": th_prev.get('label', f'Part_{k_idx+1}'),
                                                    "text": text_prev
                                                })
                        except Exception:
                            lyrics_hist = []
                        context_basic.append({
                            "name": get_instrument_name(t),
                            "role": t.get('role','complementary'),
                            "notes": t_notes,
                            **({"lyrics_history": lyrics_hist} if lyrics_hist else {})
                        })
                    try:
                        planned_role_dbg = str((plan_by_idx.get(part_idx) or {}).get('role','')).strip() or _normalize_section_role(label)
                        # Respect skip config: don't log a Lyrics:Request for parts planned as silence when skipping is enabled
                        try:
                            _skip_role_silence = bool((cfg or config).get('skip_role_silence_parts', 1))
                        except Exception:
                            _skip_role_silence = True
                        if not (_skip_role_silence and planned_role_dbg.lower() == 'silence'):
                            first_note = float(notes[0].get('start_beat', 0.0)) if notes else 0.0
                            print(f"{Fore.CYAN}  📝 Part {part_idx+1}/{len(out_themes)}: '{label}' ({planned_role_dbg}) · {len(notes)} notes{Style.RESET_ALL}")
                    except Exception:
                        pass
                    # Build textual history context from all prior parts for this track (full history)
                    try:
                        history_text = "\n".join(history_lines) if history_lines else ""
                    except Exception:
                        history_text = ""
                    # Augment description with planned role/hints/hook if available, and extract per-part prefs
                        p = plan_by_idx.get(part_idx)
                        if isinstance(p, dict):
                            prole = str(p.get('role', '')).strip()
                            phint = str(p.get('plan_hint', '')).strip()
                            # Soft sanitize: if non-silence role but hint implies instrumental-only, drop the hint
                            try:
                                phl = phint.lower()
                            except Exception:
                                phl = ''
                            if prole.lower() != 'silence' and any(k in phl for k in (
                                'instrumental only', 'maintain vocal silence', 'no vocal', 'no vocals', 'no vocal presence', 'purely instrumental'
                            )):
                                phint = ''
                            phook = str(p.get('hook_theme', '')).strip()
                            phook_can = str(p.get('hook_canonical','')).strip() if isinstance(p.get('hook_canonical'), str) else ''
                            lprefs = p.get('lyrics_prefs') if isinstance(p.get('lyrics_prefs'), dict) else None
                            pstory = str(p.get('story','')).strip() if isinstance(p.get('story'), str) else ''
                            pchorus_lines = p.get('chorus_lines') if isinstance(p.get('chorus_lines'), list) else None
                            prep_policy = p.get('repetition_policy') if isinstance(p.get('repetition_policy'), dict) else None
                            pimagery = p.get('imagery_palette') if isinstance(p.get('imagery_palette'), list) else None
                            pverbs = p.get('verb_palette') if isinstance(p.get('verb_palette'), list) else None
                            pcar = str(p.get('call_and_response','')).strip() if isinstance(p.get('call_and_response'), str) else ''
                            pchant = p.get('chant_spots') if isinstance(p.get('chant_spots'), list) else None
                            plan_bits = []
                            if prole:
                                plan_bits.append(f"role={prole}")
                            if phint:
                                plan_bits.append(f"hint={phint}")
                            if phook:
                                plan_bits.append(f"hook={phook}")
                            if phook_can:
                                plan_bits.append(f"hook_canonical=\"{phook_can}\"")
                            if pstory:
                                plan_bits.append(f"story={pstory[:140]}")
                            if isinstance(pchorus_lines, list) and pchorus_lines:
                                plan_bits.append("chorus_lines=[" + "; ".join([str(x) for x in pchorus_lines if isinstance(x, str)][:4]) + "]")
                            if isinstance(prep_policy, dict) and prep_policy:
                                try:
                                    plan_bits.append("repetition_policy={" + ", ".join([f"{k}:{v}" for k,v in prep_policy.items() if isinstance(k,str)]) + "}")
                                except Exception:
                                    pass
                            if isinstance(pimagery, list) and pimagery:
                                plan_bits.append("imagery_palette=[" + ", ".join([str(x) for x in pimagery if isinstance(x,str)][:3]) + "]")
                            if isinstance(pverbs, list) and pverbs:
                                plan_bits.append("verb_palette=[" + ", ".join([str(x) for x in pverbs if isinstance(x,str)][:3]) + "]")
                            if pcar:
                                plan_bits.append(f"call_and_response={pcar}")
                            if isinstance(pchant, list) and pchant:
                                plan_bits.append("chant_spots=[" + ", ".join([str(x) for x in pchant if isinstance(x,str)][:3]) + "]")

                            if plan_bits:
                                desc = (desc + " \n[Plan] " + ", ".join(plan_bits)).strip()
                                try:
                                    print(Style.DIM + f"[Plan→Part] {label}: " + (", ".join(plan_bits)) + Style.RESET_ALL)
                                except Exception:
                                    pass
                            # If KI prefs are provided, build a lightweight override config for this part only
                            if lprefs:
                                local_overrides = {}
                                if isinstance(lprefs.get('target_wpb'), (int, float)):
                                    local_overrides['lyrics_target_words_per_bar'] = float(lprefs.get('target_wpb'))
                                if isinstance(lprefs.get('melisma_bias'), (int, float)):
                                    val = max(0.25, min(0.55, float(lprefs.get('melisma_bias'))))
                                    local_overrides['lyrics_melisma_bias'] = val
                                if isinstance(lprefs.get('min_word_beats'), (int, float)):
                                    local_overrides['lyrics_min_word_beats'] = float(lprefs.get('min_word_beats'))
                                if isinstance(lprefs.get('allow_nonsense'), (int, float)):
                                    local_overrides['lyrics_allow_nonsense'] = int(lprefs.get('allow_nonsense'))
                                # Create a shallow merged config view
                                if local_overrides:
                                    try:
                                        cfg = {**(cfg or config), **local_overrides}
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # For existing tracks, use word-first approach (requires notes)
                    if not new_vocal_track_mode:
                        tokens = _generate_lyrics_words_with_spans(
                            cfg or config, genre, inspiration, get_instrument_name(target_tr), bpm, ts, notes,
                            section_label=label, section_description=desc, context_tracks_basic=context_basic,
                            user_prompt=user_guidance, history_context=history_text
                        )  # Save progress after tokens for this section have been generated
                        try:
                            target_tr['lyrics'] = tokens
                        except Exception:
                            pass
                        try:
                            progress_data = {
                                'type': 'lyrics_generation_existing_track',
                                'config': cfg or config,
                                'themes': out_themes,
                                'length': theme_len_bars,
                                'current_theme_index': part_idx,
                                'current_track_index': track_idx_effective,
                                'generation_type': 'existing_track',
                                'timestamp': run_timestamp
                            }
                            # Add plan data if available (only save once at the beginning)
                            if part_idx == 0:
                                progress_data.update({
                                    'user_guidance': user_guidance,
                                    'roles': roles,
                                    'plan_items': plan_items,
                                    'analysis_ctx': ANALYSIS_CTX
                                })
                            save_progress(progress_data, script_dir, run_timestamp)
                        except Exception:
                            pass
                        # Set final notes and tokens based on track type
                        if new_vocal_track_mode:
                            notes = notes2 if 'notes2' in locals() else []
                            tokens = tokens2 if 'tokens2' in locals() else []
                        else:
                            # For existing tracks, ensure tokens is defined
                            if 'tokens' not in locals():
                                tokens = []
                        
                        # Ensure tokens is always defined for both modes
                        if 'tokens' not in locals():
                            tokens = []
                                    
                    # Adaptive pass-2: if the model signals difficulty/vision, allow broader note optimization
                    try:
                        adapt_key = f"{get_instrument_name(target_tr)}|{label}"
                        meta = LYRICS_PART_META.get(adapt_key) or {}
                        pd = meta.get('placement_difficulty')
                        vision = meta.get('note_adaptation_vision')
                        # Add safety check to prevent infinite loops
                        if isinstance(pd, (int, float)) and pd >= float((cfg or config).get('lyrics_adaptation_threshold', 0.35)) and pd < 0.8:
                            print(Style.DIM + f"[Notes-Adapt] Activating adaptive optimization for '{label}' (difficulty={pd:.2f})" + Style.RESET_ALL)
                            # Temporarily relax adjustment mode for this part
                            local_cfg = {**(cfg or config)}
                            local_cfg['note_adjustment_mode'] = 'expressive'
                            if vision:
                                print(Style.DIM + f"[Notes-Adapt] vision: {vision[:200]}" + ("..." if len(vision or '')>200 else "") + Style.RESET_ALL)
                            # Disabled conservative merging/adjustments to preserve 1:1 mapping during debugging
                            # plan = _propose_lyric_note_adjustments(local_cfg, genre, inspiration, get_instrument_name(target_tr), bpm, ts, notes, tokens, label, desc, context_tracks=[t for j,t in enumerate(trks) if j != track_idx_effective])
                            # adjusted_notes, adjusted_tokens = _apply_note_adjustments_conservative(notes, tokens, plan)
                            # notes = adjusted_notes; tokens = adjusted_tokens
                        elif isinstance(pd, (int, float)) and pd >= 0.8:
                            print(Style.DIM + f"[Notes-Adapt] Difficulty too high ({pd:.2f}), skipping adaptive optimization to prevent infinite loop" + Style.RESET_ALL)
                        # If NEW track and still hard to place -> constrained re-generation of notes, then re-run words
                        if new_vocal_track_mode and isinstance(pd, (int, float)) and pd >= 0.6:
                            try:
                                print(Style.DIM + f"[Notes-Adapt] Difficulty too high ({pd:.2f}), skipping regeneration to avoid infinite loop" + Style.RESET_ALL)
                                # Instead of regenerating, just use the existing notes with simpler lyrics
                                if not tokens or len(tokens) < 3:
                                                    tokens = []  # Let the retry logic handle this
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        words_preview = ' '.join(_tokens_to_words(tokens))
                        print(f"{Fore.GREEN}  ✨ Preview: {words_preview[:200]}{'...' if len(words_preview)>200 else ''}{Style.RESET_ALL}")
                        if words_preview.strip():
                            history_lines.append(f"{label}: {words_preview}")
                    except Exception:
                        pass
                    # Optional: propose minimal note adjustments from KI; apply conservatively
                    # For new vocal tracks, notes and lyrics are already set in track_data - don't overwrite!
                    if not new_vocal_track_mode:
                        plan = {}
                        adjusted_notes, adjusted_tokens = notes, tokens
                        if not (isinstance(target_tr, dict) and target_tr.get('__final_vocal__')):
                            # Add safety check to prevent infinite loops
                            try:
                                # Disabled conservative merging/adjustments to preserve 1:1 mapping during debugging
                                # plan = _propose_lyric_note_adjustments(cfg or config, genre, inspiration, get_instrument_name(target_tr), bpm, ts, notes, tokens, label, desc, context_tracks=[t for j,t in enumerate(trks) if j != track_idx_effective])
                                # adjusted_notes, adjusted_tokens = _apply_note_adjustments_conservative(notes, tokens, plan)
                                target_tr['notes'] = notes
                                target_tr['lyrics'] = tokens
                            except Exception as e:
                                print(Style.DIM + f"[Notes-Adapt] Error in note adjustments: {e}, using original notes" + Style.RESET_ALL)
                                target_tr['notes'] = notes
                                target_tr['lyrics'] = tokens
                        else:
                            target_tr['notes'] = notes
                            target_tr['lyrics'] = tokens
                                    
                    # For new vocal tracks, the track was already added earlier (line 12873)
                    # Don't add or update again - the notes and lyrics are already correct in track_data!
                    # This section is only needed for existing track mode
                    if not new_vocal_track_mode:
                        # Update existing vocal track
                        for t in trks:
                            if t.get('__final_vocal__'):
                                t['notes'] = notes
                                t['lyrics'] = tokens
                                target_tr = t
                                break
                    # Diagnostics: show summary of applied changes
                    try:
                        plan = plan if 'plan' in locals() else {}
                        adjusted_tokens = adjusted_tokens if 'adjusted_tokens' in locals() else (target_tr.get('lyrics', []) if isinstance(target_tr, dict) else [])
                        merges_c = len(plan.get('merge_spans', [])) if isinstance(plan, dict) else 0
                        extends_c = len(plan.get('extend', {}).keys()) if isinstance(plan, dict) else 0
                        shifts_c = len(plan.get('shift', {}).keys()) if isinstance(plan, dict) else 0
                        repitches_c = len(plan.get('repitch', {}).keys()) if isinstance(plan, dict) else 0
                        total_tokens = len(adjusted_tokens) if isinstance(adjusted_tokens, list) else 0
                        melisma_c = sum(1 for t in adjusted_tokens if str(t).strip() == '-') if isinstance(adjusted_tokens, list) else 0
                        content_c = total_tokens - melisma_c
                        print(Fore.YELLOW + Style.BRIGHT + "[Lyrics:Edits] " + Style.NORMAL + Fore.WHITE + f"merges={merges_c}, extend={extends_c}, shift={shifts_c}, repitch={repitches_c} | content={content_c}, melisma={melisma_c}" + Style.RESET_ALL)
                    except Exception:
                        pass
                    try:
                        tokens = tokens if 'tokens' in locals() else (target_tr.get('lyrics', []) if isinstance(target_tr, dict) else [])
                        dash_ratio = (sum(1 for x in tokens if str(x).strip() == '-') / max(1,len(tokens))) if isinstance(tokens, list) and len(tokens) > 0 else 0.0
                        print(f"{Fore.MAGENTA}  🎵 Generated {len(tokens) if isinstance(tokens, list) else 0} tokens (melisma: {dash_ratio:.0%}){Style.RESET_ALL}")
                    except Exception:
                        pass

                # ======================================================================
                # == FINAL EXPORT (NACHDEM ALLE PARTS GENERIERT SIND)
                # ======================================================================
                if new_vocal_track_mode:
                    print("\n--- Creating UST and MIDI exports for new vocal track ---")
                    
                    final_notes = []
                    final_lyrics = []
                    theme_len_beats = float(theme_len_bars * ts.get('beats_per_bar', 4))

                    # Build virtual "themes" structure with the new vocal track data
                    # Extract vocal track from each theme in out_themes
                    vocal_themes_for_export = []
                    syllables_list = []
                    
                    for i, theme in enumerate(out_themes):
                        # Get the tracks list for this theme
                        theme_tracks = theme.get('tracks', [])
                        
                        # Find the vocal track in this theme's tracks
                        vocal_track_data = None
                        for track in theme_tracks:
                            if track.get('__final_vocal__') or track.get('instrument_name') == 'Vocal':
                                vocal_track_data = track
                                break
                        
                        # DEBUG: Check note positions
                        DEBUG_EXPORT_MAIN = os.environ.get('DEBUG_EXPORT', 'false').lower() == 'true'
                        if vocal_track_data and vocal_track_data.get('notes') and DEBUG_EXPORT_MAIN:
                            first_note = vocal_track_data['notes'][0] if vocal_track_data['notes'] else None
                            if first_note:
                                print(f"[DEBUG Export] Theme {i}: first note start_beat={first_note.get('start_beat', 0.0)}, notes_count={len(vocal_track_data['notes'])}, lyrics_count={len(vocal_track_data.get('lyrics', []))}")
                        
                        # If no vocal track found, create empty one
                        if not vocal_track_data:
                            vocal_track_data = {
                                "instrument_name": "Vocal",
                                "program_num": 0,
                                "role": "vocal",
                                "notes": [],
                                "lyrics": []
                            }
                        
                        # Normalize note positions to be relative (starting at 0.0) for UST export
                        original_notes = vocal_track_data.get('notes', [])
                        normalized_notes = []
                        if original_notes:
                            # Find the minimum start_beat to use as offset
                            min_start = min(float(note.get('start_beat', 0.0)) for note in original_notes)
                            # Normalize all notes by subtracting the offset
                            for note in original_notes:
                                norm_note = note.copy()
                                norm_note['start_beat'] = float(note.get('start_beat', 0.0)) - min_start
                                normalized_notes.append(norm_note)
                        
                        # Create a temporary track structure for this theme's vocal part
                        vocal_track = {
                            "instrument_name": "Vocal",
                            "program_num": 0,
                            "role": "vocal",
                            "notes": normalized_notes,
                            "lyrics": vocal_track_data.get('lyrics', [])
                        }
                        
                        # Create a theme structure with just this vocal track
                        theme_with_vocal = {
                            "label": theme.get('label', f'Part_{i+1}'),
                            "description": theme.get('description', ''),
                            "tracks": [vocal_track]
                        }
                        vocal_themes_for_export.append(theme_with_vocal)
                        
                        # Add lyrics to syllables list
                        syllables_list.append(vocal_track_data.get('lyrics', []))
                        
                        # Build combined final notes/lyrics for complete song
                        part_start_beat = i * theme_len_beats
                        notes_part = vocal_track_data.get('notes', [])
                        lyrics_part = vocal_track_data.get('lyrics', [])
                        
                        if notes_part and lyrics_part:
                            for note in notes_part:
                                new_note = note.copy()
                                new_note['start_beat'] += part_start_beat
                                final_notes.append(new_note)
                            final_lyrics.extend(lyrics_part)
                    
                    # Export UST file
                    
                    output_filename = f"lyrics_Vocal_{run_timestamp}.ust"
                    ust_output_path = os.path.join(script_dir, output_filename)
                    
                    _export_openutau_ust_corrected(
                        vocal_themes_for_export, 0, syllables_list, ts, bpm, ust_output_path, theme_len_bars
                    )
                    ust_path = ust_output_path
                    
                    # Export TXT files
                    txt_output_all = os.path.join(script_dir, f"lyrics_Vocal_{run_timestamp}.txt")
                    txt_output_parts = os.path.join(script_dir, f"lyrics_Vocal_{run_timestamp}_parts.txt")
                    
                    _export_emvoice_txt_for_track(
                        vocal_themes_for_export, 0, syllables_list, txt_output_all, txt_output_parts
                    )
                    txt_path = txt_output_all
                    txt_part_path = txt_output_parts
                    
                    # Export MIDI file with correct parameters
                    midi_output_path = os.path.join(script_dir, f"lyrics_Vocal_{run_timestamp}.mid")
                    midi_success = _create_midi_from_ust(ust_path, bpm, ts, midi_output_path) if ust_path else False
                    midi_path = midi_output_path if midi_success else None
                    
                    return midi_path, ust_path, txt_path, txt_part_path

                # Fallback return statement für den Fall, dass kein Modus zutrifft
                return None, None, None, None

        except Exception as e:
            print(Fore.RED + f"Error in lyrics generation: {e}" + Style.RESET_ALL)
            import traceback
            traceback.print_exc()
            return None, None, None, None
        
        # Display options
        print("1. Generate New Song")
        print("2. Optimize Existing Song")
        print("3. Resume Generation")
        print("4. Generate Lyrics for a Track (Artifact/Progress)")
        print("5. Advanced Optimization")
        print("6. Exit")
        
        choice = input(f"{Fore.GREEN}Choose an option: {Style.RESET_ALL}").strip()
        
        if choice == '6' or choice.lower() == 'q':
            print("Goodbye!")
            break
        elif choice == '4':
            # Call the actual lyrics generation function
            from song_generator import generate_lyrics_for_all_parts  # Import the function if needed
            # Call the lyrics generation flow (similar to the one in the main try block)
            try:
                # Set the action to lyrics_from_artifact to trigger the proper code path
                action = 'lyrics_from_artifact'
                # This will be handled by the main try-except block logic
                # We need to actually call the lyrics generation code
                continue  # Let the main try-except handle it
            except Exception as e:
                print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
                continue
        elif choice == '5':
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
            # If the smallest start time is at or after the expected offset, interpret as absolute
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

    # Preserve original instrument order; append additional (unlisted) instruments
    final_tracks_sorted = [merged_tracks[name] for name in instrument_order if name in merged_tracks]
    listed = set(instrument_order)
    extras = [merged_tracks[name] for name in merged_tracks.keys() if name not in listed]
    if extras:
        try:
            print(Fore.YELLOW + f"Warning: {len(extras)} track(s) not in config['instruments'] – appending to output." + Style.RESET_ALL)
        except Exception:
            pass
    final_tracks_sorted.extend(extras)

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
        # Use the robust merge logic incl. sustain & track automations
        try:
            with open(os.path.join(script_dir, "song_settings.json"), 'r') as f:
                s = json.load(f)
                length_bars = int(s.get('length', DEFAULT_LENGTH)) if isinstance(s.get('length'), int) else DEFAULT_LENGTH
        except Exception:
            length_bars = DEFAULT_LENGTH

        final_song_data = merge_themes_to_song_data(generated_themes, config, length_bars)

        final_base = build_final_song_basename(config, generated_themes, timestamp)
        final_filename = os.path.join(script_dir, f"{final_base}.mid")

        # Small delay to ensure clean file sorting by date
        time.sleep(config.get('retry_delay', DEFAULT_RETRY_DELAY))

        create_midi_from_json(final_song_data, config, final_filename)

        # Save artifact (as before)
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