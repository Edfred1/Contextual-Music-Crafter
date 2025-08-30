import os
import sys
import time
import re

# Reuse utilities from the main generator
try:
    from song_generator import (
        find_final_artifacts,
        load_final_artifact,
        summarize_artifact,
        find_progress_files,
        load_progress,
        summarize_progress_file,
        merge_themes_to_song_data,
        create_midi_from_json,
        build_final_song_basename,
    )
except Exception as e:
    print(f"Failed to import helpers from song_generator.py: {e}")
    sys.exit(1)


def choose_artifact_or_progress(finals, progresses):
    print("\nAvailable artifacts:")
    combined = []
    # Final artifacts first
    for path in finals:
        try:
            label = summarize_artifact(path)
        except Exception:
            label = os.path.basename(path)
        combined.append((path, f"[F] {label}"))
    # Progress artifacts next
    for path in progresses:
        try:
            label = summarize_progress_file(path)
        except Exception:
            label = os.path.basename(path)
        combined.append((path, f"[P] {label}"))

    if not combined:
        print("  (none)")
        return None, None

    for i, (_, label) in enumerate(combined, start=1):
        print(f"  {i}. {label}")
    print()

    idx = -1
    while not (0 <= idx < len(combined)):
        raw = input(f"Choose item (1-{len(combined)}): ").strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            idx = -1
    return combined[idx][0], combined[idx][1].startswith("[F]")


def extract_from_progress(pdata):
    """Best-effort extraction of (config, themes, length_bars, timestamp) from a progress JSON."""
    config = pdata.get("config")
    ts = pdata.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
    length = int(pdata.get("theme_length") or pdata.get("length") or 16)

    themes = None
    # Try common fields by type priority
    ptype = (pdata.get("type") or '').lower()
    if 'generation' in ptype:
        themes = pdata.get('all_themes_data')
    if themes is None and 'window_optimization' in ptype:
        themes = pdata.get('themes')
    if themes is None and 'optimization' in ptype:
        themes = pdata.get('themes_to_optimize') or pdata.get('themes') or pdata.get('final_optimized_themes')
    if themes is None and 'automation_enhancement' in ptype:
        themes = pdata.get('themes')
    # Generic fallbacks
    if themes is None:
        themes = pdata.get('themes') or pdata.get('all_themes_data') or pdata.get('themes_to_optimize')

    return config, themes, length, ts


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    finals = find_final_artifacts(script_dir)[:50]
    progresses = find_progress_files(script_dir)[:50]

    selected_path, is_final = choose_artifact_or_progress(finals, progresses)
    if not selected_path:
        print("No artifacts selected.")
        return

    if is_final:
        data = load_final_artifact(selected_path)
        if not data:
            print("Could not load the selected artifact.")
            return
        config = data.get("config")
        themes = data.get("themes")
        length_bars = int(data.get("length", 16))
        ts = data.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
        suffix = "_rebuilt"
    else:
        pdata = load_progress(selected_path)
        if not pdata:
            print("Could not load the selected progress file.")
            return
        config, themes, length_bars, ts = extract_from_progress(pdata)
        suffix = "_partial_rebuilt"

    if not config or not themes:
        print("Selected file is missing required fields (config/themes).")
        return

    # Build song data and output name
    try:
        song_data = merge_themes_to_song_data(themes, config, length_bars)
    except Exception as e:
        print(f"Failed to merge themes into song data: {e}")
        return

    try:
        base = build_final_song_basename(config, themes, ts, resumed=True)
        out_default = os.path.join(script_dir, f"{base}{suffix}.mid")
    except Exception:
        out_default = os.path.join(script_dir, f"final_song_{ts}_rebuilt.mid")

    custom = input(f"Output path [{out_default}]: ").strip()
    out_path = custom or out_default

    ok = create_midi_from_json(song_data, config, out_path)
    if ok:
        print(f"Successfully wrote: {out_path}")
    else:
        print("MIDI export failed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")

