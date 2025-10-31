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
        return []

    for i, (_, label) in enumerate(combined, start=1):
        print(f"  {i}. {label}")
    print()

    # Support multiple indices and ranges, e.g. "1,5,10-12"
    selected_indices = []
    while not selected_indices:
        raw = input(f"Choose item(s) 1-{len(combined)} (comma/range): ").strip()
        if not raw:
            continue
        tokens = re.split(r"[\s,]+", raw)
        tmp_indices = []
        for tok in tokens:
            if not tok:
                continue
            m = re.match(r"^(\d+)\s*-\s*(\d+)$", tok)
            if m:
                start = int(m.group(1))
                end = int(m.group(2))
                step = 1 if start <= end else -1
                for n in range(start, end + step, step):
                    tmp_indices.append(n - 1)
            else:
                try:
                    tmp_indices.append(int(tok) - 1)
                except ValueError:
                    # ignore invalid token
                    pass
        # Filter to valid range and de-duplicate while preserving order
        seen = set()
        for idx in tmp_indices:
            if 0 <= idx < len(combined) and idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)

    return [(combined[i][0], combined[i][1].startswith("[F]")) for i in selected_indices]


def extract_from_progress(pdata):
    """Best-effort extraction of (config, themes, length_bars, timestamp) from a progress JSON."""
    config = pdata.get("config")
    ts = pdata.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
    length = int(pdata.get("theme_length") or pdata.get("length") or 8)  # Consistent fallback with other modules

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

    selections = choose_artifact_or_progress(finals, progresses)
    if not selections:
        print("No artifacts selected.")
        return

    # Single selection: keep interactive output path prompt
    if len(selections) == 1:
        selected_path, is_final = selections[0]

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
        return

    # Multiple selections: batch-convert with default output paths
    successes = 0
    failures = 0
    for selected_path, is_final in selections:
        try:
            if is_final:
                data = load_final_artifact(selected_path)
                if not data:
                    print(f"Skip: Could not load artifact: {os.path.basename(selected_path)}")
                    failures += 1
                    continue
                config = data.get("config")
                themes = data.get("themes")
                length_bars = int(data.get("length", 16))
                ts = data.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
                suffix = "_rebuilt"
            else:
                pdata = load_progress(selected_path)
                if not pdata:
                    print(f"Skip: Could not load progress: {os.path.basename(selected_path)}")
                    failures += 1
                    continue
                config, themes, length_bars, ts = extract_from_progress(pdata)
                suffix = "_partial_rebuilt"

            if not config or not themes:
                print(f"Skip: Missing config/themes in {os.path.basename(selected_path)}")
                failures += 1
                continue

            try:
                song_data = merge_themes_to_song_data(themes, config, length_bars)
            except Exception as e:
                print(f"Skip: Merge failed for {os.path.basename(selected_path)}: {e}")
                failures += 1
                continue

            try:
                base = build_final_song_basename(config, themes, ts, resumed=True)
                out_default = os.path.join(script_dir, f"{base}{suffix}.mid")
            except Exception:
                out_default = os.path.join(script_dir, f"final_song_{ts}_rebuilt.mid")

            ok = create_midi_from_json(song_data, config, out_default)
            if ok:
                print(f"Successfully wrote: {out_default}")
                successes += 1
            else:
                print(f"Export failed for: {out_default}")
                failures += 1
        except Exception as e:
            print(f"Unexpected error while processing {os.path.basename(selected_path)}: {e}")
            failures += 1

    print(f"\nBatch complete. Success: {successes}, Failed: {failures}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")

