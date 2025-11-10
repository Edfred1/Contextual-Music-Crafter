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
        find_final_artifacts, get_scale_notes, _compact_notes_json, _extract_token_limit_from_error
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
            full_error = str(e)
            
            # Check for token limit errors specifically (input token count exceeded)
            is_token_limit_error = (
                "input_token" in err or
                "input token" in err or
                ("quota" in err and "token" in err and "limit" in err)
            )
            
            # Token limit errors should be handled by the caller (they need to reduce context)
            # NOTE: Token limits are PER MINUTE (125,000 tokens/minute for free tier)
            if is_token_limit_error:
                token_limit = _extract_token_limit_from_error(full_error)
                if token_limit is not None:
                    print(Fore.YELLOW + f"Token limit exceeded: {token_limit:,} tokens/minute. This should be handled by the caller to reduce context." + Style.RESET_ALL)
                    # Return a special error that the caller can catch
                    raise ValueError(f"TOKEN_LIMIT_EXCEEDED:{token_limit}")
            
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
    
    # Get notes from selected parts (all notes, no compression)
    selected_notes = []
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            track_data = get_track_from_theme(theme, get_instrument_name(track))
            if track_data:
                notes = track_data.get("notes", [])
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
    
    # Include all notes in compact format to save tokens
    notes_json = _compact_notes_json(selected_notes)
    
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
    
    # Build context: all tracks from selected parts (all notes, no compression)
    context_summary = []
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            for track in theme.get("tracks", []):
                track_name = get_instrument_name(track)
                if track_name != get_instrument_name(original_track):
                    notes = track.get("notes", [])
                    # Include all notes without compression
                    context_summary.append({
                        "part": theme.get("label", f"Part_{part_idx+1}"),
                        "instrument": track_name,
                        "role": track.get("role", "complementary"),
                        "notes_sample": notes,
                    })
    
    # Build original track data for selected parts (all notes, no compression)
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
                
                # Include all notes without compression
                
                # Extract automations from track
                track_automations = track_data.get("track_automations", {})
                note_automations = []
                for note in rel_notes:
                    if "automations" in note:
                        note_automations.append({
                            "note_index": len([n for n in rel_notes if n.get("start_beat", 0) <= note.get("start_beat", 0)]),
                            "automations": note.get("automations", {})
                        })
                
                part_data = {
                    "part_label": theme.get("label", f"Part_{part_idx+1}"),
                    "part_index": part_idx,
                    "notes": rel_notes,
                }
                
                # Add automations if present
                if track_automations:
                    part_data["track_automations"] = track_automations
                if note_automations:
                    part_data["note_automations"] = note_automations[:20]  # Limit to 20 examples
                
                original_parts_data.append(part_data)
    
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
    
    # Check if original track has automations
    has_automations = False
    automation_instructions = ""
    for part_data in original_parts_data:
        if "track_automations" in part_data or "note_automations" in part_data:
            has_automations = True
            break
    
    if has_automations:
        automation_instructions = (
            "\n**--- AUTOMATION SUPPORT ---**\n"
            "The original track uses MIDI automations (pitch bend, CC curves). You may include automations in your variation:\n"
            "1. **Note-based automations:** Add an `\"automations\"` object inside a note:\n"
            "   ```json\n"
            "   {\"pitch\": 60, \"start_beat\": 0.0, \"duration_beats\": 1.0, \"velocity\": 100,\n"
            "    \"automations\": {\n"
            "      \"pitch_bend\": [{\"type\": \"curve\", \"start_beat\": 0.0, \"end_beat\": 0.5, \"start_value\": 0, \"end_value\": 4096, \"bias\": 1.0}],\n"
            "      \"cc\": [{\"type\": \"curve\", \"cc\": 74, \"start_beat\": 0.0, \"end_beat\": 1.0, \"start_value\": 60, \"end_value\": 127, \"bias\": 1.0}]\n"
            "    }\n"
            "   }\n"
            "   ```\n"
            "2. **Track-based automations:** Add a `\"track_automations\"` object at the part level:\n"
            "   ```json\n"
            "   {\"part_index\": 0, \"track_automations\": {\n"
            "     \"pitch_bend\": [{\"type\": \"curve\", \"start_beat\": 0.0, \"end_beat\": 8.0, \"start_value\": 0, \"end_value\": -4096, \"bias\": 1.0}],\n"
            "     \"cc\": [{\"type\": \"curve\", \"cc\": 74, \"start_beat\": 0.0, \"end_beat\": 8.0, \"start_value\": 0, \"end_value\": 127, \"bias\": 1.0}]\n"
            "   }}\n"
            "   ```\n"
            "**Pitch Bend Range:** -8192 to 8191 (0 = no bend)\n"
            "**CC Range:** 0 to 127\n"
            "**Important:** Return to neutral values (pitch bend → 0) at the end of curves unless musically intentional.\n"
            "Automations are optional - only include them if they enhance the variation musically.\n\n"
        )
    
    # Serialize JSON for prompt (compact format, all notes included)
    # Convert notes in original_parts_data to compact format
    compact_original_parts = []
    for part in original_parts_data:
        compact_part = dict(part)
        if 'notes' in compact_part:
            compact_part['notes'] = json.loads(_compact_notes_json(compact_part['notes']))
        compact_original_parts.append(compact_part)
    original_parts_json = json.dumps(compact_original_parts, separators=(',', ':'))
    
    # Convert notes in context_summary to compact format
    compact_context_summary = []
    for ctx in context_summary[:10]:
        compact_ctx = dict(ctx)
        if 'notes_sample' in compact_ctx:
            compact_ctx['notes_sample'] = json.loads(_compact_notes_json(compact_ctx['notes_sample']))
        compact_context_summary.append(compact_ctx)
    context_summary_json = json.dumps(compact_context_summary, separators=(',', ':'))
    
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
        f"{automation_instructions}"
        f"**--- VARIATION TYPE ---**\n"
        f"**Technique:** {variation_type}\n"
        f"Apply this variation technique while maintaining musical coherence and the original's character.\n\n"
        f"**--- ORIGINAL TRACK (Selected Parts) ---**\n"
        f"**Note:** Notes use compact format: s=start_beat, d=duration_beats, p=pitch, v=velocity\n"
        f"```json\n{original_parts_json}\n```\n\n"
        f"**--- CONTEXT TRACKS (Other instruments in selected parts) ---**\n"
        f"**Note:** Notes use compact format: s=start_beat, d=duration_beats, p=pitch, v=velocity\n"
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
        f'      ],\n'
        f'      "track_automations": {{"pitch_bend": [], "cc": []}}\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n'
        f"```\n\n"
        f"Each note must have: pitch (0-127), start_beat (float, relative to part start), duration_beats (float), velocity (1-127).\n"
        f"Notes may optionally include an \"automations\" object with pitch_bend and/or cc arrays.\n"
        f"Parts may optionally include a \"track_automations\" object with pitch_bend and/or cc arrays.\n"
        f"**IMPORTANT:** Return ONLY the JSON object, no markdown, no explanations, no prose. Start with '{{' and end with '}}'."
    )
    
    # Store original data for potential reduction
    original_context_summary = context_summary[:]
    original_original_parts_data = original_parts_data[:]
    
    for attempt in range(MAX_RETRIES):
        try:
            text, tokens = _call_llm_with_retry(prompt, config, expects_json=True)
            if not text:
                if attempt < MAX_RETRIES - 1:
                    print(Fore.YELLOW + f"Retry {attempt + 1}/{MAX_RETRIES}..." + Style.RESET_ALL)
                    time.sleep(2)
                    continue
                return None
        except ValueError as e:
            error_str = str(e)
            # Check if it's a token limit error
            if error_str.startswith("TOKEN_LIMIT_EXCEEDED:"):
                token_limit = int(error_str.split(":")[1])
                print(Fore.YELLOW + f"Token limit exceeded: {token_limit:,} tokens/minute. Reducing context and retrying..." + Style.RESET_ALL)
                # Reduce context by limiting parts
                if len(original_context_summary) > 1:
                    # Reduce to half the parts
                    reduced_count = max(1, len(original_context_summary) // 2)
                    context_summary = original_context_summary[-reduced_count:]
                    # Also reduce original_parts_data to match
                    if len(original_original_parts_data) > reduced_count:
                        original_parts_data = original_original_parts_data[-reduced_count:]
                    # Recreate compact JSON
                    compact_original_parts = []
                    for part in original_parts_data:
                        compact_part = dict(part)
                        if 'notes' in compact_part:
                            compact_part['notes'] = json.loads(_compact_notes_json(compact_part['notes']))
                        compact_original_parts.append(compact_part)
                    original_parts_json = json.dumps(compact_original_parts, separators=(',', ':'))
                    
                    compact_context_summary = []
                    for ctx in context_summary:
                        compact_ctx = dict(ctx)
                        if 'notes_sample' in compact_ctx:
                            compact_ctx['notes_sample'] = json.loads(_compact_notes_json(compact_ctx['notes_sample']))
                        compact_context_summary.append(compact_ctx)
                    context_summary_json = json.dumps(compact_context_summary, separators=(',', ':'))
                    
                    # Recreate prompt with reduced context (reuse the prompt template from above)
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
                        f"{automation_instructions}"
                        f"**--- VARIATION TYPE ---**\n"
                        f"**Technique:** {variation_type}\n"
                        f"Apply this variation technique while maintaining musical coherence and the original's character.\n\n"
                        f"**--- ORIGINAL TRACK (Selected Parts) ---**\n"
                        f"**Note:** Notes use compact format: s=start_beat, d=duration_beats, p=pitch, v=velocity\n"
                        f"```json\n{original_parts_json}\n```\n\n"
                        f"**--- CONTEXT TRACKS (Other instruments in selected parts) ---**\n"
                        f"**Note:** Notes use compact format: s=start_beat, d=duration_beats, p=pitch, v=velocity\n"
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
                        f'      ],\n'
                        f'      "track_automations": {{"pitch_bend": [], "cc": []}}\n'
                        f'    }}\n'
                        f'  ]\n'
                        f'}}\n'
                        f"```\n\n"
                        f"Each note must have: pitch (0-127), start_beat (float, relative to part start), duration_beats (float), velocity (1-127).\n"
                        f"Notes may optionally include an \"automations\" object with pitch_bend and/or cc arrays.\n"
                        f"Parts may optionally include a \"track_automations\" object with pitch_bend and/or cc arrays.\n"
                        f"**IMPORTANT:** Return ONLY the JSON object, no markdown, no explanations, no prose. Start with '{{' and end with '}}'."
                    )
                    print(Fore.CYAN + f"Reduced context from {len(original_context_summary)} to {len(context_summary)} parts to fit within {token_limit:,} token limit." + Style.RESET_ALL)
                    continue
                else:
                    print(Fore.RED + f"Unable to reduce context further. Token limit {token_limit:,} is too restrictive." + Style.RESET_ALL)
                    return None
            else:
                # Re-raise if it's not a token limit error
                raise
        
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
            
            # Collect track automations from all parts
            track_automations_combined = {"pitch_bend": [], "cc": []}
            
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
                        # Convert compact format to full format if needed
                        # Resume files always use full format, but LLM responses may use compact format
                        has_compact = ('s' in note or 'd' in note or 'p' in note or 'v' in note)
                        has_full = ('start_beat' in note or 'duration_beats' in note or 'pitch' in note or 'velocity' in note)
                        
                        if has_compact and not has_full:
                            # Convert from compact to full format
                            full_note = {}
                            if 's' in note:
                                full_note['start_beat'] = note['s']
                            if 'd' in note:
                                full_note['duration_beats'] = note['d']
                            if 'p' in note:
                                full_note['pitch'] = note['p']
                            if 'v' in note:
                                full_note['velocity'] = note['v']
                            # Preserve other fields (automations, etc.)
                            for key in note:
                                if key not in ['s', 'd', 'p', 'v']:
                                    full_note[key] = note[key]
                            note = full_note
                        
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
                            note_obj = {
                                "pitch": pitch,
                                "start_beat": abs_start,
                                "duration_beats": duration,
                                "velocity": velocity,
                            }
                            
                            # Extract automations from note if present
                            if "automations" in note and isinstance(note.get("automations"), dict):
                                note_obj["automations"] = note["automations"]
                            
                            variation_track["notes"].append(note_obj)
                    except (ValueError, TypeError) as ve:
                        # Skip invalid notes but continue processing
                        continue
                
                # Extract track automations from part
                if "track_automations" in part_data and isinstance(part_data["track_automations"], dict):
                    part_start_beats = part_idx * bars_per_section * beats_per_bar
                    ta = part_data["track_automations"]
                    
                    # Process pitch bend automations
                    if "pitch_bend" in ta and isinstance(ta["pitch_bend"], list):
                        for pb in ta["pitch_bend"]:
                            if isinstance(pb, dict) and pb.get("type") == "curve":
                                pb_copy = dict(pb)
                                pb_copy["start_beat"] = part_start_beats + float(pb.get("start_beat", 0))
                                pb_copy["end_beat"] = part_start_beats + float(pb.get("end_beat", 0))
                                track_automations_combined["pitch_bend"].append(pb_copy)
                    
                    # Process CC automations
                    if "cc" in ta and isinstance(ta["cc"], list):
                        for cc in ta["cc"]:
                            if isinstance(cc, dict) and cc.get("type") == "curve":
                                cc_copy = dict(cc)
                                cc_copy["start_beat"] = part_start_beats + float(cc.get("start_beat", 0))
                                cc_copy["end_beat"] = part_start_beats + float(cc.get("end_beat", 0))
                                track_automations_combined["cc"].append(cc_copy)
            
            # Add track automations if any were found
            if track_automations_combined["pitch_bend"] or track_automations_combined["cc"]:
                variation_track["track_automations"] = track_automations_combined
            
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
    
    # Collect track automations from original track (for selected parts only)
    track_automations_original = {"pitch_bend": [], "cc": []}
    
    # Collect original notes from selected parts only
    for part_idx in selected_parts:
        if part_idx < len(themes):
            theme = themes[part_idx]
            track_data = get_track_from_theme(theme, track_name)
            if track_data:
                notes = track_data.get("notes", [])
                # Ensure absolute timing
                part_start_beats = part_idx * bars_per_section * beats_per_bar
                
                # Collect track automations from this part (once per part, not per note)
                if "track_automations" in track_data:
                    ta = track_data["track_automations"]
                    if "pitch_bend" in ta and isinstance(ta["pitch_bend"], list):
                        for pb in ta["pitch_bend"]:
                            if isinstance(pb, dict) and pb.get("type") == "curve":
                                pb_copy = dict(pb)
                                # Convert to absolute timing
                                pb_start = float(pb.get("start_beat", 0))
                                pb_end = float(pb.get("end_beat", 0))
                                # Check if already absolute
                                if pb_start < part_start_beats - 0.1:
                                    pb_start = pb_start + part_start_beats
                                    pb_end = pb_end + part_start_beats
                                pb_copy["start_beat"] = pb_start
                                pb_copy["end_beat"] = pb_end
                                track_automations_original["pitch_bend"].append(pb_copy)
                    if "cc" in ta and isinstance(ta["cc"], list):
                        for cc in ta["cc"]:
                            if isinstance(cc, dict) and cc.get("type") == "curve":
                                cc_copy = dict(cc)
                                # Convert to absolute timing
                                cc_start = float(cc.get("start_beat", 0))
                                cc_end = float(cc.get("end_beat", 0))
                                # Check if already absolute
                                if cc_start < part_start_beats - 0.1:
                                    cc_start = cc_start + part_start_beats
                                    cc_end = cc_end + part_start_beats
                                cc_copy["start_beat"] = cc_start
                                cc_copy["end_beat"] = cc_end
                                track_automations_original["cc"].append(cc_copy)
                
                # Process notes
                for note in notes:
                    try:
                        start = float(note.get("start_beat", 0))
                        # Check if already absolute (if start is >= part_start, it's likely absolute)
                        if start < part_start_beats - 0.1:  # Small tolerance
                            start = start + part_start_beats
                        # Only add notes within the part boundaries
                        if part_start_beats <= start < part_start_beats + (bars_per_section * beats_per_bar):
                            note_obj = {
                                "pitch": int(note.get("pitch", 60)),
                                "start_beat": start,
                                "duration_beats": float(note.get("duration_beats", 1.0)),
                                "velocity": int(note.get("velocity", 100)),
                            }
                            
                            # Preserve note automations if present
                            if "automations" in note and isinstance(note.get("automations"), dict):
                                note_obj["automations"] = note["automations"]
                            
                            original_export_track["notes"].append(note_obj)
                    except:
                        continue
    
    # Build final song data - only include tracks with notes
    all_tracks = []
    # Add track automations if any were found
    if track_automations_original["pitch_bend"] or track_automations_original["cc"]:
        original_export_track["track_automations"] = track_automations_original
    
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

