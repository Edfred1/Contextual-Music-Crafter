#!/usr/bin/env python3
"""
Melody Variation Generator
Generates variations of a selected track from a MIDI file or Final JSON.
"""

import os
import sys
import json
import time
import math
import glob
from typing import List, Tuple, Dict, Optional
from colorama import Fore, Style, init

# Import framework components from song_generator
try:
    import song_generator as sg
    from song_generator import (
        _classify_quota_error, _is_key_available, _set_key_cooldown,
        _next_available_key, _all_keys_cooling_down, _seconds_until_first_available,
        _all_keys_daily_exhausted, _schedule_hourly_probe_if_needed,
        _seconds_until_hourly_probe, initialize_api_keys, get_instrument_name,
        create_midi_from_json, merge_themes_to_song_data, load_final_artifact,
        find_final_artifacts, get_scale_notes
    )
    import google.generativeai as genai
    # Reference to song_generator's globals for API key management
    API_KEYS = sg.API_KEYS
    CURRENT_KEY_INDEX = sg.CURRENT_KEY_INDEX
    KEY_COOLDOWN_UNTIL = sg.KEY_COOLDOWN_UNTIL
    KEY_QUOTA_TYPE = sg.KEY_QUOTA_TYPE
except ImportError as e:
    print(Fore.RED + f"Error importing from song_generator: {e}" + Style.RESET_ALL)
    sys.exit(1)

# Import MIDI analysis from music_analyzer
try:
    from music_analyzer import analyze_midi_file, split_tracks_into_sections
except ImportError:
    print(Fore.YELLOW + "Warning: music_analyzer not available. MIDI analysis may be limited." + Style.RESET_ALL)
    analyze_midi_file = None
    split_tracks_into_sections = None

import mido
from midiutil import MIDIFile

# Initialize colorama
init(autoreset=True)

# Constants
PER_MINUTE_COOLDOWN_SECONDS = 60
PER_HOUR_COOLDOWN_SECONDS = 3600
PER_DAY_COOLDOWN_SECONDS = 86400
MAX_RETRIES = 3
MAX_NOTES_IN_CONTEXT = 500

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(script_dir, "config.yaml")


def load_config() -> Dict:
    """Load config.yaml"""
    try:
        import yaml
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not load config.yaml: {e}" + Style.RESET_ALL)
        return {}


def _extract_text_from_response(response) -> str:
    """Extract text from LLM response, handling various response formats."""
    try:
        if hasattr(response, 'text'):
            return response.text or ""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts:
                    return getattr(parts[0], 'text', '') or ""
        return str(response) if response else ""
    except Exception:
        return ""


def _extract_json_object(raw: str) -> str:
    """Extract JSON object from text, handling markdown code blocks."""
    if not raw:
        return ""
    # Try to find JSON object boundaries
    start_idx = raw.find('{')
    if start_idx == -1:
        return ""
    # Find matching closing brace
    brace_count = 0
    for i in range(start_idx, len(raw)):
        if raw[i] == '{':
            brace_count += 1
        elif raw[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return raw[start_idx:i+1]
    return raw[start_idx:]


def _call_llm_with_retry(prompt_text: str, config: Dict, expects_json: bool = False, max_attempts: int = MAX_RETRIES) -> Tuple[str, int]:
    """Call LLM with retry logic and key rotation (from song_generator pattern)"""
    # Always use song_generator's globals directly for consistency
    # The helper functions (_next_available_key, etc.) use song_generator's globals
    
    if not sg.API_KEYS:
        print(Fore.RED + "No API keys available." + Style.RESET_ALL)
        return "", 0
    
    model_name = config.get("model_name", "gemini-2.5-pro")
    generation_config = {
        "temperature": config.get("temperature", 1.0)
    }
    if expects_json:
        generation_config["response_mime_type"] = "application/json"
    if isinstance(config.get("max_output_tokens"), int):
        generation_config["max_output_tokens"] = config.get("max_output_tokens")
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    attempts = 0
    quota_rotation_count = 0
    
    while attempts < max_attempts:
        attempts += 1
        try:
            # Use song_generator's globals directly
            if not sg.API_KEYS or sg.CURRENT_KEY_INDEX >= len(sg.API_KEYS):
                print(Fore.RED + "No API keys available." + Style.RESET_ALL)
                return "", 0
            
            genai.configure(api_key=sg.API_KEYS[sg.CURRENT_KEY_INDEX])
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            response = model.generate_content(prompt_text, safety_settings=safety_settings)
            out_text = _extract_text_from_response(response)
            tokens = int(getattr(getattr(response, "usage_metadata", None), "total_token_count", 0) or 0)
            return out_text, tokens
        except Exception as e:
            err = str(e).lower()
            if '429' in err or 'quota' in err or 'rate limit' in err:
                # Use song_generator's globals directly
                qtype = _classify_quota_error(err)
                sg.KEY_QUOTA_TYPE[sg.CURRENT_KEY_INDEX] = qtype
                
                # Set cooldown (this modifies song_generator's KEY_COOLDOWN_UNTIL)
                if qtype == 'per-day':
                    _set_key_cooldown(sg.CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                elif qtype in ('per-hour', 'rate-limit'):
                    _set_key_cooldown(sg.CURRENT_KEY_INDEX, PER_HOUR_COOLDOWN_SECONDS)
                else:
                    _set_key_cooldown(sg.CURRENT_KEY_INDEX, PER_MINUTE_COOLDOWN_SECONDS)
                
                # Try rotation (uses song_generator's globals)
                nxt = _next_available_key()
                if nxt is not None:
                    sg.CURRENT_KEY_INDEX = nxt
                    print(Fore.CYAN + f"Switching to API key #{nxt+1}..." + Style.RESET_ALL)
                    continue
                
                # All keys cooling down
                if _all_keys_cooling_down():
                    if _all_keys_daily_exhausted():
                        _schedule_hourly_probe_if_needed()
                        wait_time = _seconds_until_hourly_probe()
                    else:
                        wait_time = max(5.0, min(_seconds_until_first_available(), PER_HOUR_COOLDOWN_SECONDS))
                    print(Fore.CYAN + f"All keys cooling down. Waiting {wait_time:.1f}s..." + Style.RESET_ALL)
                    time.sleep(wait_time)
                    continue
            else:
                # Transient error
                wait_time = min(30, 3 * (2 ** max(0, attempts - 1)))
                print(Fore.YELLOW + f"Transient error, retrying in {wait_time:.1f}s..." + Style.RESET_ALL)
                time.sleep(wait_time)
                continue
    
    print(Fore.RED + f"Failed after {max_attempts} attempts." + Style.RESET_ALL)
    return "", 0


def _summarize_artifact(path: str) -> str:
    """Create a summary string for a Final JSON artifact."""
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
        length = data.get('length', '?')
        
        # Get timestamp from file or data
        try:
            ts = data.get('timestamp') or time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        except:
            ts = os.path.basename(path).replace('final_run_', '').replace('.json', '')
        
        return f"{os.path.basename(path)} | {genre} | {bpm}bpm | {key} | {num_parts} parts ({length} bars each) | {first_label}→{last_label} | {ts}"
    except Exception:
        return os.path.basename(path)


def select_input_file() -> Tuple[Optional[str], bool]:
    """Select MIDI file or Final JSON. Returns (path, is_midi)"""
    print(Fore.CYAN + "\n=== Input Selection ===" + Style.RESET_ALL)
    print("1. MIDI file (.mid, .midi)")
    print("2. Final JSON (final_run_*.json)")
    
    choice = input(f"{Fore.GREEN}Select input type (1/2): {Style.RESET_ALL}").strip()
    
    if choice == "1":
        # MIDI files
        midi_files = glob.glob("**/*.mid", recursive=True) + glob.glob("**/*.midi", recursive=True)
        if not midi_files:
            print(Fore.RED + "No MIDI files found." + Style.RESET_ALL)
            return None, False
        print(Fore.CYAN + "\nFound MIDI files:" + Style.RESET_ALL)
        for i, f in enumerate(midi_files, 1):
            print(f"  {i}. {f}")
        sel = input(f"{Fore.GREEN}Select file number: {Style.RESET_ALL}").strip()
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(midi_files):
                return midi_files[idx], True
        except:
            pass
        print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
        return None, False
    
    elif choice == "2":
        # Final JSON files
        json_files = find_final_artifacts(script_dir)
        if not json_files:
            print(Fore.RED + "No Final JSON files found." + Style.RESET_ALL)
            return None, False
        print(Fore.CYAN + "\nFound Final JSON files:" + Style.RESET_ALL)
        for i, f in enumerate(json_files[:20], 1):  # Show max 20
            summary = _summarize_artifact(f)
            print(f"  {i}. {summary}")
        sel = input(f"{Fore.GREEN}Select file number: {Style.RESET_ALL}").strip()
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(json_files):
                return json_files[idx], False
        except:
            pass
        print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
        return None, False
    
    return None, False


def load_midi_data(file_path: str, config: Dict) -> Tuple[Optional[Dict], Optional[List[Dict]], Optional[int]]:
    """Load and analyze MIDI file. Returns (config_dict, themes, bars_per_section)"""
    if not analyze_midi_file:
        print(Fore.RED + "MIDI analysis not available." + Style.RESET_ALL)
        return None, None, None
    
    print(Fore.CYAN + f"\nAnalyzing MIDI file: {file_path}" + Style.RESET_ALL)
    
    # Ask for genre and section length
    genre = input(f"{Fore.GREEN}Enter genre: {Style.RESET_ALL}").strip() or "Electronic"
    bars_str = input(f"{Fore.GREEN}Section length in bars (8/16/32, default 16): {Style.RESET_ALL}").strip() or "16"
    try:
        bars_per_section = int(bars_str)
    except:
        bars_per_section = 16
    
    # Analyze MIDI
    tracks, bpm, total_bars, ts, root_note, scale_type = analyze_midi_file(file_path)
    if not tracks:
        print(Fore.RED + "No tracks found in MIDI file." + Style.RESET_ALL)
        return None, None, None
    
    print(Fore.GREEN + f"Analyzed: {bpm:.1f} BPM, {ts['beats_per_bar']}/{ts['beat_value']}, ~{total_bars} bars, Key: {root_note} {scale_type}" + Style.RESET_ALL)
    
    # Split into sections
    themes = split_tracks_into_sections(tracks, bars_per_section, ts["beats_per_bar"])
    
    # Build config
    midi_config = {
        "genre": genre,
        "bpm": round(float(bpm)),
        "time_signature": ts,
        "key_scale": f"{root_note} {scale_type}",
        "root_note": root_note,
        "scale_type": scale_type,
        "inspiration": f"Variations of MIDI file: {os.path.basename(file_path)}",
    }
    
    return midi_config, themes, bars_per_section


def load_json_data(file_path: str) -> Tuple[Optional[Dict], Optional[List[Dict]], Optional[int]]:
    """Load Final JSON. Returns (config_dict, themes, bars_per_section)"""
    print(Fore.CYAN + f"\nLoading Final JSON: {file_path}" + Style.RESET_ALL)
    
    data = load_final_artifact(file_path)
    if not data:
        return None, None, None
    
    config = data.get("config", {})
    themes = data.get("themes", [])
    length = data.get("length", 16)
    
    if not themes:
        print(Fore.RED + "No themes found in JSON." + Style.RESET_ALL)
        return None, None, None
    
    print(Fore.GREEN + f"Loaded: {len(themes)} parts, {length} bars per part" + Style.RESET_ALL)
    return config, themes, length


def analyze_part_content(themes: List[Dict], track_name: str, bars_per_section: int, beats_per_bar: int) -> List[Dict]:
    """Analyze which parts contain notes for a specific track. Returns list of part info dicts."""
    part_info = []
    
    for i, theme in enumerate(themes):
        label = theme.get("label", f"Part_{i+1}")
        tracks = theme.get("tracks", [])
        
        # Find the track
        track = None
        for t in tracks:
            if get_instrument_name(t) == track_name:
                track = t
                break
        
        if not track:
            part_info.append({
                "index": i,
                "label": label,
                "has_notes": False,
                "note_count": 0,
                "density": 0.0,
                "bar_range": f"{i * bars_per_section + 1}-{(i + 1) * bars_per_section}",
                "start_beat": i * bars_per_section * beats_per_bar,
                "end_beat": (i + 1) * bars_per_section * beats_per_bar,
                "pitch_range": None,
                "avg_velocity": 0,
                "total_duration": 0.0,
            })
            continue
        
        notes = track.get("notes", [])
        note_count = len(notes)
        
        # Calculate density (notes per bar)
        density = note_count / bars_per_section if bars_per_section > 0 else 0.0
        
        # Find actual note range and additional metrics
        pitch_range = None
        avg_velocity = 0
        total_duration = 0.0
        
        if notes:
            starts = [float(n.get("start_beat", 0)) for n in notes]
            ends = [s + float(n.get("duration_beats", 0)) for s, n in zip(starts, notes)]
            min_start = min(starts)
            max_end = max(ends)
            
            # Pitch range
            pitches = [int(n.get("pitch", 60)) for n in notes]
            if pitches:
                min_pitch = min(pitches)
                max_pitch = max(pitches)
                pitch_range = f"{min_pitch}-{max_pitch}"
            
            # Average velocity
            velocities = [int(n.get("velocity", 100)) for n in notes]
            if velocities:
                avg_velocity = round(sum(velocities) / len(velocities), 1)
            
            # Total duration
            durations = [float(n.get("duration_beats", 0)) for n in notes]
            total_duration = round(sum(durations), 2)
            
            # Convert to bars (relative to part start)
            part_start_beats = i * bars_per_section * beats_per_bar
            min_bar = math.floor((min_start - part_start_beats) / beats_per_bar) + 1
            max_bar = math.ceil((max_end - part_start_beats) / beats_per_bar)
            min_bar = max(1, min_bar)
            max_bar = min(bars_per_section, max_bar)
            bar_range = f"{min_bar}-{max_bar}" if min_bar <= max_bar else "none"
        else:
            bar_range = "none"
        
        part_info.append({
            "index": i,
            "label": label,
            "has_notes": note_count > 0,
            "note_count": note_count,
            "density": round(density, 2),
            "bar_range": bar_range,
            "start_beat": i * bars_per_section * beats_per_bar,
            "end_beat": (i + 1) * bars_per_section * beats_per_bar,
            "pitch_range": pitch_range,
            "avg_velocity": avg_velocity,
            "total_duration": total_duration,
        })
    
    return part_info


def select_track(themes: List[Dict]) -> Optional[str]:
    """Select a track from themes. Returns track name."""
    # Collect all unique tracks
    all_tracks = {}
    for theme in themes:
        for track in theme.get("tracks", []):
            name = get_instrument_name(track)
            if name not in all_tracks:
                all_tracks[name] = {
                    "name": name,
                    "role": track.get("role", "complementary"),
                    "program": track.get("program_num", 0),
                }
    
    if not all_tracks:
        print(Fore.RED + "No tracks found." + Style.RESET_ALL)
        return None
    
    print(Fore.CYAN + "\n=== Available Tracks ===" + Style.RESET_ALL)
    track_list = list(all_tracks.values())
    for i, t in enumerate(track_list, 1):
        print(f"  {i}. {t['name']} (Role: {t['role']}, Program: {t['program']})")
    
    if len(track_list) == 1:
        print(Fore.GREEN + f"\nOnly one track found. Auto-selecting: {track_list[0]['name']}" + Style.RESET_ALL)
        return track_list[0]['name']
    
    sel = input(f"{Fore.GREEN}Select track number: {Style.RESET_ALL}").strip()
    try:
        idx = int(sel) - 1
        if 0 <= idx < len(track_list):
            return track_list[idx]['name']
    except:
        pass
    
    print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
    return None


def select_parts(part_info: List[Dict]) -> List[int]:
    """Select parts to vary. Returns list of part indices."""
    print(Fore.CYAN + "\n=== Part Analysis ===" + Style.RESET_ALL)
    
    # Separate parts with and without notes
    parts_with_notes = [p for p in part_info if p["has_notes"]]
    parts_without_notes = [p for p in part_info if not p["has_notes"]]
    
    # Display parts with notes (detailed)
    if parts_with_notes:
        print(Fore.GREEN + f"\nParts WITH notes ({len(parts_with_notes)}):" + Style.RESET_ALL)
        print(f"{'#':<4} {'Label':<25} {'Notes':<8} {'Density':<10} {'Bars':<12} {'Pitch':<12} {'Vel':<6} {'Duration':<10}")
        print("-" * 95)
        
        for p in parts_with_notes:
            idx_str = f"{p['index']+1}"
            label_str = p['label'][:24] if len(p['label']) <= 24 else p['label'][:21] + "..."
            notes_str = str(p['note_count'])
            density_str = f"{p['density']:.1f}/bar"
            bars_str = p['bar_range'] if p['bar_range'] != "none" else "all"
            pitch_str = p['pitch_range'] or "N/A"
            vel_str = f"{p['avg_velocity']:.0f}" if p['avg_velocity'] > 0 else "N/A"
            dur_str = f"{p['total_duration']:.1f}b" if p['total_duration'] > 0 else "N/A"
            
            print(f"{Fore.GREEN}{idx_str:<4}{Style.RESET_ALL} {label_str:<25} {notes_str:<8} {density_str:<10} {bars_str:<12} {pitch_str:<12} {vel_str:<6} {dur_str:<10}")
    
    # Display parts without notes (compact)
    if parts_without_notes:
        print(Fore.YELLOW + f"\nParts WITHOUT notes ({len(parts_without_notes)}):" + Style.RESET_ALL)
        part_nums = [str(p['index']+1) for p in parts_without_notes]
        # Display in rows of 10
        for i in range(0, len(part_nums), 10):
            row = part_nums[i:i+10]
            print(f"  {', '.join(row)}")
    
    # Filter indices
    parts_with_notes_indices = [p["index"] for p in parts_with_notes]
    
    if not parts_with_notes_indices:
        print(Fore.RED + "\nNo parts contain notes for this track." + Style.RESET_ALL)
        return []
    
    print(Fore.CYAN + f"\nParts with notes: {', '.join([str(i+1) for i in parts_with_notes_indices])}" + Style.RESET_ALL)
    sel = input(f"{Fore.GREEN}Select parts (e.g., '1,3,5' or 'all'): {Style.RESET_ALL}").strip().lower()
    
    if sel == "all":
        return parts_with_notes_indices
    
    try:
        indices = [int(x.strip()) - 1 for x in sel.split(",") if x.strip().isdigit()]
        # Filter to only parts with notes
        selected = [i for i in indices if i in parts_with_notes_indices]
        if selected:
            return selected
    except:
        pass
    
    print(Fore.RED + "Invalid selection. Using all parts with notes." + Style.RESET_ALL)
    return parts_with_notes_indices


def get_track_from_theme(theme: Dict, track_name: str) -> Optional[Dict]:
    """Get a specific track from a theme by name."""
    for track in theme.get("tracks", []):
        if get_instrument_name(track) == track_name:
            return track
    return None


def get_role_instructions_for_variation(role: str, config: Dict) -> str:
    """Returns role-specific instructions for variation generation."""
    role_map = {
        "drums": "**Your Role: The Rhythmic Foundation**\nCreate rhythmic variations that maintain the groove while adding interest through ghost notes, fills, accents, or polyrhythms.",
        "kick_and_snare": "**Your Role: The Core Beat**\nVary the kick and snare pattern while keeping the fundamental pulse intact. Consider ghost notes, off-beat accents, or subtle fills.",
        "percussion": "**Your Role: Rhythmic Texture**\nAdd rhythmic interest through syncopation, polyrhythms, or accent variations while complementing the main drums.",
        "bass": "**Your Role: The Groove Foundation**\nCreate variations that maintain the harmonic foundation while exploring rhythmic patterns, octaves, slides, or staccato techniques.",
        "sub_bass": "**Your Role: The Low-End Anchor**\nKeep variations simple and powerful. Consider subtle rhythmic shifts or octave jumps while maintaining the low-end presence.",
        "pads": "**Your Role: Harmonic Atmosphere**\nVary chord voicings, inversions, or rhythmic patterns while maintaining the harmonic foundation and atmospheric quality.",
        "atmosphere": "**Your Role: Sonic Environment**\nCreate evolving variations that maintain the mood while exploring different textures, rhythms, or harmonic movements.",
        "lead": "**Your Role: The Main Hook (Lead)**\nCreate melodic variations that preserve the hook's catchiness through techniques like syncopation, octaves, ornamentation, or inversion.",
        "melody": "**Your Role: The Supporting Melody**\nVary the melody through syncopation, octaves, trioles, arpeggios, or ornamentation while maintaining its supportive role.",
        "chords": "**Your Role: Harmonic Structure**\nExplore chord inversions, voicings, arpeggios, or rhythmic patterns while maintaining the harmonic progression.",
        "harmony": "**Your Role: Harmonic Support**\nVary through inversions, voicings, or rhythmic patterns while keeping the harmonic function intact.",
        "arp": "**Your Role: Rhythmic Harmony**\nCreate arpeggio variations through rhythm changes, octave shifts, or pattern modifications while maintaining the hypnotic quality.",
        "guitar": "**Your Role: Guitar**\nVary through rhythmic patterns, strumming techniques, or melodic embellishments while maintaining the guitar's character.",
        "vocal": "**Your Role: Vocal Line**\nCreate vocal variations through melisma, ornamentation, rhythm changes, or octave shifts while maintaining the vocal character.",
        "fx": "**Your Role: Sound Effects**\nVary transitional effects while maintaining their function of moving between sections.",
    }
    return role_map.get(role.lower(), f"**Your Role: {role.title()}**\nCreate musically appropriate variations that enhance the composition.")


def generate_variation_types(config: Dict, track: Dict, role: str, context_tracks: List[Dict], themes: List[Dict], selected_parts: List[int]) -> List[str]:
    """Ask AI to suggest variation types for this track. Returns list of variation type names."""
    # Build context summary
    genre = config.get("genre", "Electronic")
    key_scale = config.get("key_scale", "C major")
    bpm = config.get("bpm", 120)
    inspiration = config.get("inspiration", "")
    
    # Get notes from selected parts (compressed: head + tail for large lists)
    selected_notes = []
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            track_data = get_track_from_theme(theme, get_instrument_name(track))
            if track_data:
                notes = track_data.get("notes", [])
                if len(notes) > MAX_NOTES_IN_CONTEXT:
                    head = notes[:MAX_NOTES_IN_CONTEXT//2]
                    tail = notes[-MAX_NOTES_IN_CONTEXT//2:]
                    selected_notes.extend(head + tail)
                else:
                    selected_notes.extend(notes)
    
    # Build role-specific examples
    role_examples = {
        "melody": ["Syncopes", "Octaves", "Triolen", "Arpeggios", "Ornamentation", "Inversion"],
        "lead": ["Syncopes", "Octaves", "Triolen", "Arpeggios", "Ornamentation", "Inversion"],
        "drums": ["Ghost Notes", "Fills", "Accents", "Polyrhythms", "Off-beat Patterns"],
        "kick_and_snare": ["Ghost Notes", "Fills", "Accents", "Polyrhythms", "Off-beat Patterns"],
        "bass": ["Walking Bass", "Staccato", "Slides", "Octaves", "Syncopes"],
        "chords": ["Inversions", "Voicings", "Arpeggios", "Rhythmic Patterns"],
        "harmony": ["Inversions", "Voicings", "Arpeggios", "Rhythmic Patterns"],
    }
    
    examples = role_examples.get(role.lower(), ["Variation 1", "Variation 2", "Variation 3"])
    
    # Compress notes for prompt (use compact JSON)
    notes_sample = selected_notes[:MAX_NOTES_IN_CONTEXT] if len(selected_notes) > MAX_NOTES_IN_CONTEXT else selected_notes
    notes_json = json.dumps(notes_sample, separators=(',', ':'))
    
    prompt = (
        f"You are a music expert. Analyze this track and suggest {len(selected_parts)}-{len(selected_parts)+3} creative variation types.\n\n"
        f"**Musical Context:**\n"
        f"- Genre: {genre}\n"
        f"- Key/Scale: {key_scale}\n"
        f"- BPM: {bpm}\n"
        f"{f'- Inspiration: {inspiration}' if inspiration else ''}\n"
        f"- Track Role: {role}\n"
        f"- Instrument: {get_instrument_name(track)}\n\n"
        f"**Original Track Sample:**\n"
        f"```json\n{notes_json}\n```\n\n"
        f"**Examples for {role} role:** {', '.join(examples)}\n\n"
        f"**Task:**\n"
        f"Suggest creative variation types that:\n"
        f"1. Are musically appropriate for this role and genre\n"
        f"2. Will create interesting variations while staying true to the original musical intent\n"
        f"3. Have descriptive names (1-3 words each, e.g., \"Syncopes\", \"Octaves\", \"Triolen\")\n"
        f"4. Are diverse from each other (avoid similar techniques)\n"
        f"5. Match the energy and style of the original\n\n"
        f"Return ONLY a JSON array of strings, e.g.: [\"Syncopes\", \"Octaves\", \"Triolen\"]\n"
        f"Do not include explanations, only the JSON array."
    )
    
    text, _ = _call_llm_with_retry(prompt, config, expects_json=True)
    if not text:
        # Fallback to examples
        return examples[:min(5, len(selected_parts)+2)]
    
    try:
        cleaned = text.strip().replace("```json", "").replace("```", "")
        variation_types = json.loads(cleaned)
        if isinstance(variation_types, list) and all(isinstance(v, str) for v in variation_types):
            return variation_types[:10]  # Limit to 10
    except:
        pass
    
    return examples[:min(5, len(selected_parts)+2)]


def generate_variation(config: Dict, original_track: Dict, variation_type: str, role: str, 
                      context_tracks: List[Dict], themes: List[Dict], selected_parts: List[int],
                      bars_per_section: int, part_info: List[Dict]) -> Optional[Dict]:
    """Generate a single variation of the track."""
    genre = config.get("genre", "Electronic")
    key_scale = config.get("key_scale", "C major")
    bpm = config.get("bpm", 120)
    ts = config.get("time_signature", {"beats_per_bar": 4, "beat_value": 4})
    beats_per_bar = ts.get("beats_per_bar", 4)
    
    # Get scale notes
    try:
        root_note = config.get("root_note", 60)
        scale_type = config.get("scale_type", "major")
        scale_notes = get_scale_notes(root_note, scale_type)
    except:
        scale_notes = "C, D, E, F, G, A, B"
    
    # Build context: all tracks from selected parts (compressed)
    context_summary = []
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            for track in theme.get("tracks", []):
                track_name = get_instrument_name(track)
                if track_name != get_instrument_name(original_track):
                    notes = track.get("notes", [])
                    # Compress: head + tail for large note lists
                    if len(notes) > MAX_NOTES_IN_CONTEXT:
                        head = notes[:MAX_NOTES_IN_CONTEXT//2]
                        tail = notes[-MAX_NOTES_IN_CONTEXT//2:]
                        notes = head + tail
                    context_summary.append({
                        "part": theme.get("label", f"Part_{part_idx+1}"),
                        "instrument": track_name,
                        "role": track.get("role", "complementary"),
                        "notes_sample": notes[:MAX_NOTES_IN_CONTEXT],
                    })
    
    # Build original track data for selected parts (compressed)
    original_parts_data = []
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            track_data = get_track_from_theme(theme, get_instrument_name(original_track))
            if track_data:
                notes = track_data.get("notes", [])
                # Normalize to relative timing within part
                part_start_beats = part_idx * bars_per_section * beats_per_bar
                rel_notes = []
                for note in notes:
                    rel_note = dict(note)
                    rel_note["start_beat"] = float(note.get("start_beat", 0)) - part_start_beats
                    rel_notes.append(rel_note)
                
                # Compress if too many notes
                if len(rel_notes) > MAX_NOTES_IN_CONTEXT:
                    head = rel_notes[:MAX_NOTES_IN_CONTEXT//2]
                    tail = rel_notes[-MAX_NOTES_IN_CONTEXT//2:]
                    rel_notes = head + tail
                
                original_parts_data.append({
                    "part_label": theme.get("label", f"Part_{part_idx+1}"),
                    "part_index": part_idx,
                    "notes": rel_notes[:MAX_NOTES_IN_CONTEXT],
                })
    
    # Build transition context (previous and next parts)
    transition_context = []
    for i, part_idx in enumerate(selected_parts):
        prev_idx = part_idx - 1
        next_idx = part_idx + 1
        
        prev_notes = []
        next_notes = []
        
        if prev_idx >= 0 and prev_idx < len(themes):
            prev_track = get_track_from_theme(themes[prev_idx], get_instrument_name(original_track))
            if prev_track:
                prev_notes = prev_track.get("notes", [])[-10:]  # Last 10 notes
        
        if next_idx < len(themes):
            next_track = get_track_from_theme(themes[next_idx], get_instrument_name(original_track))
            if next_track:
                next_notes = next_track.get("notes", [])[:10]  # First 10 notes
        
        if prev_notes or next_notes:
            transition_context.append({
                "part_index": part_idx,
                "previous_part_last_notes": prev_notes,
                "next_part_first_notes": next_notes,
            })
    
    # Get role-specific instructions
    role_instructions = get_role_instructions_for_variation(role, config)
    
    # Polyphony rules based on role
    POLYPHONIC_ROLES = {"harmony", "chords", "pads", "atmosphere", "texture", "guitar"}
    EXPRESSIVE_MONOPHONIC_ROLES = {"lead", "melody", "vocal"}
    if role in POLYPHONIC_ROLES:
        polyphony_rule = "**Polyphonic:** Notes CAN overlap to create chords or harmonies."
    elif role in EXPRESSIVE_MONOPHONIC_ROLES:
        polyphony_rule = "**Expressive Monophonic:** Notes should primarily be one at a time, but short overlaps are permitted for legato phrasing."
    else:
        polyphony_rule = "**Monophonic:** Notes should NOT overlap in time (except for drums/percussion)."
    
    # Drum map for drums
    drum_map_instructions = ""
    if role in ["drums", "percussion", "kick_and_snare"]:
        drum_map_instructions = (
            "**Drum Map (Standard MIDI):**\n"
            "- Kick: 36, Snare: 38, Rimshot: 40, Closed Hi-Hat: 42, Open Hi-Hat: 46\n"
            "- Crash: 49, Ride: 51, High Tom: 50, Mid Tom: 48, Low Tom: 45\n"
            "Use appropriate velocities: ghost notes < 45, accents ≥ 100.\n\n"
        )
    
    # Compress JSON for prompt (use compact format)
    original_parts_json = json.dumps(original_parts_data, separators=(',', ':'))
    context_summary_json = json.dumps(context_summary[:10], separators=(',', ':'))
    transition_context_json = json.dumps(transition_context, separators=(',', ':'))
    
    inspiration_text = config.get("inspiration", "")
    
    prompt = (
        f"You are an expert music producer. Create a variation of this track using the '{variation_type}' technique.\n\n"
        f"**--- MUSICAL CONTEXT ---**\n"
        f"**Genre:** {genre}\n"
        f"**Key/Scale:** {key_scale} (Available notes: {scale_notes})\n"
        f"**Tempo:** {bpm} BPM\n"
        f"**Time Signature:** {ts.get('beats_per_bar', 4)}/{ts.get('beat_value', 4)}\n"
        f"**Section Length:** {bars_per_section} bars per part ({bars_per_section * beats_per_bar} beats)\n"
        f"{f'**Inspiration:** {inspiration_text}' if inspiration_text else ''}\n\n"
        f"**--- TRACK INFORMATION ---**\n"
        f"**Track Role:** {role}\n"
        f"**Instrument:** {get_instrument_name(original_track)} (MIDI Program: {original_track.get('program_num', 0)})\n"
        f"{role_instructions}\n\n"
        f"{drum_map_instructions}"
        f"**--- VARIATION TYPE ---**\n"
        f"**Technique:** {variation_type}\n"
        f"Apply this variation technique while maintaining musical coherence and the original's character.\n\n"
        f"**--- ORIGINAL TRACK (Selected Parts) ---**\n"
        f"```json\n{original_parts_json}\n```\n\n"
        f"**--- CONTEXT TRACKS (Other instruments in selected parts) ---**\n"
        f"```json\n{context_summary_json}\n```\n\n"
        f"**--- TRANSITION CONTEXT (for smooth flow between parts) ---**\n"
        f"```json\n{transition_context_json}\n```\n\n"
        f"**--- CRITICAL REQUIREMENTS ---**\n"
        f"1. Create variations ONLY for the selected parts (indices: {selected_parts})\n"
        f"2. Maintain smooth transitions between parts - the last note of one part should flow naturally into the first note of the next\n"
        f"3. Respect the original musical intent while applying the '{variation_type}' technique\n"
        f"4. Stay in key/scale: {scale_notes}\n"
        f"5. Match the rhythm and energy of the context tracks\n"
        f"6. If the original had a hard cut between parts, you may preserve it; otherwise create smooth transitions\n"
        f"7. Use relative timing within each part (start_beat 0..{bars_per_section * beats_per_bar})\n"
        f"8. {polyphony_rule}\n"
        f"9. **Motif Coherence:** Preserve the main musical motifs while transforming them through the variation technique (inversion, octave shift, rhythm augmentation, etc.)\n\n"
        f"**--- OUTPUT FORMAT: JSON ---**\n"
        f"Return a JSON object with this structure:\n"
        f"```json\n"
        f"{{\n"
        f'  "variation_name": "{variation_type}",\n'
        f'  "parts": [\n'
        f'    {{\n'
        f'      "part_index": 0,\n'
        f'      "notes": [\n'
        f'        {{"pitch": 60, "start_beat": 0.0, "duration_beats": 1.0, "velocity": 100}}\n'
        f'      ]\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n'
        f"```\n\n"
        f"Each note must have: pitch (0-127), start_beat (float, relative to part start), duration_beats (float), velocity (1-127).\n"
        f"**IMPORTANT:** Return ONLY the JSON object, no markdown, no explanations, no prose. Start with '{{' and end with '}}'."
    )
    
    for attempt in range(MAX_RETRIES):
        text, tokens = _call_llm_with_retry(prompt, config, expects_json=True)
        if not text:
            if attempt < MAX_RETRIES - 1:
                print(Fore.YELLOW + f"Retry {attempt + 1}/{MAX_RETRIES}..." + Style.RESET_ALL)
                time.sleep(2)
                continue
            return None
        
        try:
            # Extract JSON from response
            json_payload = _extract_json_object(text)
            if not json_payload:
                # Fallback: try cleaning markdown
                cleaned = text.strip().replace("```json", "").replace("```", "").strip()
                json_payload = _extract_json_object(cleaned) or cleaned
            
            result = json.loads(json_payload)
            
            if not isinstance(result, dict) or "parts" not in result:
                raise ValueError("Invalid structure: missing 'parts' key or not a dict")
            
            # Validate and convert to absolute timing
            variation_track = {
                "instrument_name": f"{get_instrument_name(original_track)}_Variation_{variation_type}",
                "program_num": original_track.get("program_num", 0),
                "role": role,
                "notes": [],
            }
            
            parts = result.get("parts", [])
            if not isinstance(parts, list):
                raise ValueError("'parts' must be a list")
            
            for part_data in parts:
                if not isinstance(part_data, dict):
                    continue
                    
                part_idx = part_data.get("part_index")
                if part_idx not in selected_parts:
                    continue
                
                part_start_beats = part_idx * bars_per_section * beats_per_bar
                notes = part_data.get("notes", [])
                
                if not isinstance(notes, list):
                    continue
                
                for note in notes:
                    if not isinstance(note, dict):
                        continue
                    
                    try:
                        # Validate and clamp values
                        pitch = int(note.get("pitch", 60))
                        pitch = max(0, min(127, pitch))  # Clamp to valid MIDI range
                        
                        rel_start = float(note.get("start_beat", 0))
                        abs_start = part_start_beats + rel_start
                        
                        duration = float(note.get("duration_beats", 1.0))
                        duration = max(0.0, duration)  # Ensure non-negative
                        
                        velocity = int(note.get("velocity", 100))
                        velocity = max(1, min(127, velocity))  # Clamp to valid MIDI range
                        
                        # Validate timing is within part bounds
                        part_end_beats = part_start_beats + (bars_per_section * beats_per_bar)
                        if abs_start < part_end_beats:  # Only add if within part
                            variation_track["notes"].append({
                                "pitch": pitch,
                                "start_beat": abs_start,
                                "duration_beats": duration,
                                "velocity": velocity,
                            })
                    except (ValueError, TypeError) as ve:
                        # Skip invalid notes but continue processing
                        continue
            
            if variation_track["notes"]:
                return variation_track
            else:
                raise ValueError("No valid notes generated for selected parts")
            
        except json.JSONDecodeError as je:
            print(Fore.YELLOW + f"JSON parse error (attempt {attempt + 1}): {je}" + Style.RESET_ALL)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
        except ValueError as ve:
            print(Fore.YELLOW + f"Validation error (attempt {attempt + 1}): {ve}" + Style.RESET_ALL)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
        except Exception as e:
            print(Fore.YELLOW + f"Parse error (attempt {attempt + 1}): {type(e).__name__}: {e}" + Style.RESET_ALL)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
    
    return None


def export_variations_midi(config: Dict, original_track: Dict, variations: List[Dict], 
                          themes: List[Dict], bars_per_section: int, output_path: str) -> bool:
    """Export original track + variations to MIDI file."""
    try:
        # Calculate total length
        total_bars = len(themes) * bars_per_section
        ts = config.get("time_signature", {"beats_per_bar": 4, "beat_value": 4})
        beats_per_bar = ts.get("beats_per_bar", 4)
        total_beats = total_bars * beats_per_bar
        
        # Create song data structure
        all_tracks = []
        
        # Add original track (only for selected parts - we'll filter in merge)
        original_track_copy = dict(original_track)
        original_track_copy["notes"] = []  # Will be filled from themes
        all_tracks.append(original_track_copy)
        
        # Add variations
        for var in variations:
            all_tracks.append(var)
        
        # Build song data
        song_data = {
            "tracks": all_tracks,
        }
        
        # Create MIDI
        return create_midi_from_json(song_data, config, output_path)
    
    except Exception as e:
        print(Fore.RED + f"Export error: {e}" + Style.RESET_ALL)
        return False


def main():
    """Main function"""
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*80)
    print("  MELODY VARIATION GENERATOR")
    print("="*80 + Style.RESET_ALL + "\n")
    
    # Load config
    config = load_config()
    if not initialize_api_keys(config):
        print(Fore.RED + "Failed to initialize API keys." + Style.RESET_ALL)
        return
    
    # Sync API keys after initialization
    global API_KEYS, CURRENT_KEY_INDEX, KEY_COOLDOWN_UNTIL, KEY_QUOTA_TYPE
    API_KEYS = sg.API_KEYS
    CURRENT_KEY_INDEX = sg.CURRENT_KEY_INDEX
    KEY_COOLDOWN_UNTIL = sg.KEY_COOLDOWN_UNTIL
    KEY_QUOTA_TYPE = sg.KEY_QUOTA_TYPE
    
    # Verify API keys are available
    if not API_KEYS:
        print(Fore.RED + "No API keys available after initialization." + Style.RESET_ALL)
        return
    
    print(Fore.CYAN + f"API keys synchronized: {len(API_KEYS)} key(s) available." + Style.RESET_ALL)
    
    # Select input
    input_path, is_midi = select_input_file()
    if not input_path:
        return
    
    # Load data
    if is_midi:
        config_data, themes, bars_per_section = load_midi_data(input_path, config)
    else:
        config_data, themes, bars_per_section = load_json_data(input_path)
    
    if not config_data or not themes or not bars_per_section:
        print(Fore.RED + "Failed to load data." + Style.RESET_ALL)
        return
    
    # Merge config
    config.update(config_data)
    
    # Select track
    track_name = select_track(themes)
    if not track_name:
        return
    
    # Get track from first theme for reference
    original_track = None
    for theme in themes:
        track = get_track_from_theme(theme, track_name)
        if track:
            original_track = track
            break
    
    if not original_track:
        print(Fore.RED + f"Track '{track_name}' not found." + Style.RESET_ALL)
        return
    
    role = original_track.get("role", "complementary")
    ts = config.get("time_signature", {"beats_per_bar": 4, "beat_value": 4})
    beats_per_bar = ts.get("beats_per_bar", 4)
    
    # Analyze parts
    part_info = analyze_part_content(themes, track_name, bars_per_section, beats_per_bar)
    
    # Select parts
    selected_parts = select_parts(part_info)
    if not selected_parts:
        return
    
    # Get number of variations
    num_vars_str = input(f"{Fore.GREEN}Number of variations to generate (default 3): {Style.RESET_ALL}").strip() or "3"
    try:
        num_variations = int(num_vars_str)
        num_variations = max(1, min(10, num_variations))  # Limit 1-10
    except:
        num_variations = 3
    
    # Generate variation types
    print(Fore.CYAN + "\n=== Generating Variation Types ===" + Style.RESET_ALL)
    variation_types = generate_variation_types(config, original_track, role, [], themes, selected_parts)
    if len(variation_types) > num_variations:
        variation_types = variation_types[:num_variations]
    elif len(variation_types) < num_variations:
        # Extend with numbered variations
        for i in range(len(variation_types), num_variations):
            variation_types.append(f"Variation_{i+1}")
    
    print(Fore.GREEN + f"Variation types: {', '.join(variation_types)}" + Style.RESET_ALL)
    
    # Generate variations
    print(Fore.CYAN + "\n=== Generating Variations ===" + Style.RESET_ALL)
    variations = []
    
    for i, var_type in enumerate(variation_types, 1):
        print(Fore.CYAN + f"\nGenerating variation {i}/{len(variation_types)}: {var_type}..." + Style.RESET_ALL)
        
        # Get context tracks for selected parts
        context_tracks = []
        for part_idx in selected_parts:
            if part_idx < len(themes):
                theme = themes[part_idx]
                for track in theme.get("tracks", []):
                    if get_instrument_name(track) != track_name:
                        context_tracks.append(track)
        
        variation = generate_variation(
            config, original_track, var_type, role, context_tracks, themes,
            selected_parts, bars_per_section, part_info
        )
        
        if variation:
            variations.append(variation)
            print(Fore.GREEN + f"✓ Generated: {var_type} ({len(variation.get('notes', []))} notes)" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + f"✗ Failed to generate: {var_type}" + Style.RESET_ALL)
    
    if not variations:
        print(Fore.RED + "\nNo variations were generated." + Style.RESET_ALL)
        return
    
    # Export MIDI
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"Variations_{track_name}_{timestamp}.mid"
    output_path = os.path.join(script_dir, output_filename)
    
    print(Fore.CYAN + f"\n=== Exporting MIDI ===" + Style.RESET_ALL)
    
    # For export, we need to include the original track with notes from selected parts only
    original_export_track = {
        "instrument_name": track_name,
        "program_num": original_track.get("program_num", 0),
        "role": role,
        "notes": [],
    }
    
    # Collect original notes from selected parts only
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            track_data = get_track_from_theme(theme, track_name)
            if track_data:
                notes = track_data.get("notes", [])
                # Ensure absolute timing
                part_start_beats = part_idx * bars_per_section * beats_per_bar
                for note in notes:
                    try:
                        start = float(note.get("start_beat", 0))
                        # Check if already absolute (if start is >= part_start, it's likely absolute)
                        if start < part_start_beats - 0.1:  # Small tolerance
                            start = start + part_start_beats
                        # Only add notes within the part boundaries
                        if part_start_beats <= start < part_start_beats + (bars_per_section * beats_per_bar):
                            original_export_track["notes"].append({
                                "pitch": int(note.get("pitch", 60)),
                                "start_beat": start,
                                "duration_beats": float(note.get("duration_beats", 1.0)),
                                "velocity": int(note.get("velocity", 100)),
                            })
                    except:
                        continue
    
    # Build final song data - only include tracks with notes
    all_tracks = []
    if original_export_track["notes"]:
        all_tracks.append(original_export_track)
    all_tracks.extend(variations)
    
    if not all_tracks:
        print(Fore.RED + "\n✗ No tracks to export (original has no notes in selected parts)." + Style.RESET_ALL)
        return
    
    song_data = {"tracks": all_tracks}
    
    if create_midi_from_json(song_data, config, output_path):
        print(Fore.GREEN + f"\n✓ Successfully exported: {output_filename}" + Style.RESET_ALL)
        print(Fore.CYAN + f"  Original track: {len(original_export_track['notes'])} notes" + Style.RESET_ALL)
        for var in variations:
            print(Fore.CYAN + f"  {var['instrument_name']}: {len(var.get('notes', []))} notes" + Style.RESET_ALL)
    else:
        print(Fore.RED + "\n✗ Export failed." + Style.RESET_ALL)


if __name__ == "__main__":
    main()

