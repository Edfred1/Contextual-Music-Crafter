# 🎵 MIDI Folder Merger

Automatically merges all MIDI files in a folder into one multi-track MIDI file.

## Usage

1. Place the script and your MIDI files in the same folder
2. Run: `python midi_folder_merger.py`
3. Output: `merged_midi.mid` (all files merged, sorted by creation date)

## Features

- ✅ Supports all MIDI types (Type 0, 1, 2)
- ✅ Automatic timing conversion (different ticks per beat)
- ✅ Preserves all MIDI data (notes, CC, automation, etc.)
- ✅ Uses filenames as track names (e.g., "Evolving Minor Sequence.mid" → Track: "Evolving Minor Sequence")
- ✅ Auto-installs dependencies (mido)

## Track Naming

The tool uses your **filename** as the track name:

```
Input files:
  - Evolving Minor Sequence.mid
  - Deep Bass Line.mid
  - Ambient Pad.mid

Output tracks:
  - Track 1: "Evolving Minor Sequence"
  - Track 2: "Deep Bass Line"
  - Track 3: "Ambient Pad"
```

**Tip:** Use meaningful filenames for better context!

## Requirements

- Python 3.6+

---

**Note:** Files are processed by creation date (oldest first). The first file's tempo and time signature are used for the merged output.

