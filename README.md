# Contextual Music Crafter (CMC)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb)

> **Note: Try it in your Browser!**
> The included [Google Colab Notebook](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb) has been updated to run the full suite of scripts directly in your browser with no local installation required. Please be aware that this Colab integration is considered experimental and has not been as extensively tested as running the scripts on a local machine. For the most stable experience, we recommend a local installation.

Contextual Music Crafter (CMC) is an intelligent, context-aware MIDI music generation tool that leverages the power of Google's Gemini to compose multi-track musical pieces. Unlike simple random note generators, CMC builds songs iteratively, instrument by instrument. Each new part is intelligently composed in response to the parts that have already been written, creating cohesive and musically interesting results. CMC also includes a Synthesizer‑ready lyric pipeline aimed at Synthesizer V Studio 2 (UST export). UST files may also work in OpenUtau and similar UTAU‑compatible tools. In addition, Emvoice TXT exports are produced for simple copy/paste into Emvoice.

The entire creative direction of the music is guided through an interactive setup process, making it accessible to both developers and musicians.

## Table of Contents

- [Toolkit overview](#toolkit-overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [How to Use](#how-to-use)
  - [0. (Optional) Analyze an existing MIDI first](#0-optional-analyze-an-existing-midi-first)
  - [1. Optional Legacy Tools (single parts & quick edits)](#1-optional-legacy-tools-single-parts--quick-edits)
  - [2. Full Song Generation: The Creative Duo](#2-full-song-generation-the-creative-duo)
- [Note on legacy scripts](#note-on-legacy-scripts)
- [Advanced usage and notes](#advanced-usage-and-notes)
  - [Lyric generation – quick guide](#lyric-generation--quick-guide)
- [Music Analyzer (optional)](#music-analyzer-optional)
- [Artifact Builder (optional)](#artifact-builder-optional)
- [Further advanced notes (addendum)](#further-advanced-notes-addendum)

## 🧰 Toolkit overview

- **music_crafter.py**: Interactive planner (co‑producer). Creates/updates `config.yaml` and `song_settings.json`, then launches the generator.
- **song_generator.py**: Core engine that builds, optimizes, and resumes full songs from the plan. Includes the lyric generator. Exports SynthV‑ready UST, Emvoice TXT, and (optionally) a single‑track MIDI for vocals.
- **music_analyzer.py**: Optional pre‑step to analyze existing MIDI and derive a compact, LLM‑friendly plan.
- **artifact_builder.py**: Utility to rebuild `.mid` from final artifacts or progress JSONs without re‑running generation.
- **part_generator.py**: Legacy single‑part generator (quick loops).
- **part_extender.py**: Legacy tool to append new parts to an existing MIDI.
- **part_variator.py**: Legacy tool to create variations of selected tracks in a MIDI.

## ✨ Features

-   **Go Beyond Loops:** Generate complete songs with multiple, distinct sections (intro, verse, chorus) to tell a musical story.
-   **Guided Song Setup:** Use an interactive wizard to go from a simple idea to a full song structure, complete with AI-suggested instrumentation and creative direction for each part.
-   **Contextual Composition:** The AI listens to the existing music before adding a new part, creating cohesive arrangements that sound like they were composed with intent.
-   **Polish Your Tracks:** After generation, an AI "producer" can optimize each part to enhance groove, dynamics, and overall musicality for a more professional sound.
-   **Resumable Workflow:** Never lose your work. If a long generation job is interrupted, you can resume it right where you left off.
-   **Fine-Grained Control:** While the interactive assistant is powerful, you can always dive into the `config.yaml` and `song_settings.json` files to manually tweak every detail.

-   **SynthV‑ready vocals:** A lyric pipeline designed for Synthesizer V Studio 2: Step‑0 plans roles/hooks with a meta‑filter, Stage‑1 shapes text for singability, Stage‑2 maps notes with soft hook contiguity and micro‑note handling. Exports UST for SynthV and two Emvoice TXT files (whole/by part). UST may also load in OpenUtau.

---

**You can find several generated example MIDI files in the `MIDI Examples` folder included in this repository.**

---

## 🚀 Installation & Setup

### Step 1: Download & Install Prerequisites

- **Python & Pip**
  - Make sure you have Python 3.7 or newer installed. You can check this by opening your terminal or command prompt and typing `python --version` or `python3 --version`.
  - If you don't have Python, download it from [python.org](https://www.python.org/).
    - **⭐ Important for Windows Users:** During installation, make sure to check the box that says **"Add Python to PATH"**.

### Step 2: Get the Project Files

1. **Download the ZIP:**
   - Go to the [main repository page](https://github.com/Edfred1/Contextual-Music-Crafter).
   - Click `Code` > `Download ZIP`.
   - Unzip the file to a folder of your choice.
   - Open a terminal or command prompt in that folder.

### Step 3: Install Python Packages

- **On Windows:** Double-click and run the `install.bat` file.
- **On macOS/Linux:** Open a terminal and run:
  ```bash
  chmod +x install.sh
  ./install.sh
  ```

<details>
<summary>Manual Installation (if the script fails)</summary>

If you prefer to install manually, run this command in your terminal:
```bash
pip install -r requirements.txt
```
</details>

### Step 4: API Key & Configuration

1.  **Open `config.yaml`** in a text editor.
2.  **Set your API Key:**
    - Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Find the `api_key` field and replace `"YOUR_GOOGLE_AI_API_KEY"` with your actual key.
    ```yaml
    # API Configuration
    api_key: "YOUR_GOOGLE_AI_API_KEY"
    ```
    
3.  **Customize Your Music:**
    Adjust the parameters in `config.yaml` to define your desired musical output.

    -   `api_key`: Your Google AI API Key.
    -   `model_name`: The specific generative model to use (e.g., "gemini-2.5-flash" or "gemini-2.5-pro").
    -   `temperature`: (0.0 to 2.0) Controls creativity. Lower values are more deterministic, higher values are more experimental. Default is 1.0.
    -   `context_window_size`: Defines how many previous musical parts are sent as context for the next generation. `-1` (dynamic) is recommended for quality, `0` disables context, and a positive number (e.g., `4`) sends a fixed amount of recent parts.
    -   `inspiration`: A detailed text prompt describing the style and mood. This is the most important creative input!
    -   `genre`: The primary musical genre.
    -   `bpm`: Tempo in beats per minute.
    -   `key_scale`: The key and scale (e.g., "C minor", "F# dorian"). The `config.yaml` includes a comment with all available scales.
    -   `use_call_and_response`: (0 or 1) When enabled, melodic instruments will try to play in the gaps left by other instruments.
    -   `number_of_iterations`: The number of songs to generate in one run.
    -   `time_signature`: Set the beats per bar and the beat value.
    -   `instruments`: Define your "band." Each instrument is a list item with a `name`, a MIDI `program_num`, and a musical `role`. The order in this list determines the generation order.

    **Instrument Roles:**
    The `role` parameter guides the AI's composition for that instrument. Available roles are:
    -   `drums`: The main rhythmic foundation (kick, snare, hats).
    -   `kick_and_snare`: The foundational kick and snare pattern, adapted to the specified genre.
    -   `percussion`: Complementary rhythmic elements (bongos, congas, shakers).
    -   `sub_bass`: Very deep, fundamental bass (often a sine wave).
    -   `bass`: The main, rhythmic bassline.
    -   `pads`: Atmospheric pads, long/slow chords.
    -   `atmosphere`: Similar to pads, often more textured or evolving.
    -   `texture`: Sonic textures, not necessarily melodic or rhythmic.
    -   `chords`: Defines the chord progression.
    -   `harmony`: Similar to chords, providing harmonic structure.
    -   `arp`: An arpeggio; plays chord notes sequentially.
    -   `guitar`: Versatile for riffs, melodies, or rhythmic chords.
    -   `lead`: The main melody or riff.
    -   `melody`: A secondary or supporting melody line.
    -   `vocal`: Rhythmic vocal chops or short phrases.
    -   `fx`: Sound effects, risers, impacts, etc.
    -   `complementary`: A general-purpose role.

## 🎹 How to Use

CMC provides a suite of tools for different creative needs, from low-cost loop creation to full-scale song composition. **We strongly recommend starting with the 'Core Generators'** to understand the process and conserve your API tokens before moving to the more advanced tools.

### 0. (Optional) Analyze an existing MIDI first

If you want to start from an existing MIDI structure, run a quick analysis first and derive a compact plan you can feed into the generator. After analysis, you can directly optimize a subset of tracks or add a new context‑aware track across all parts from within the analyzer. See the section "Music Analyzer (optional)" below for details.

### 1. Optional Legacy Tools (single parts & quick edits)

If you prefer low-cost experiments or quick edits, you can use the original scripts controlled by `config.yaml`:

- `part_generator.py`: generate a single, complete musical part from scratch
- `part_extender.py`: add new parts to an existing MIDI file
- `part_variator.py`: create variations of selected tracks in a MIDI file

Each script reads your `config.yaml` and writes a new `.mid`. For most users, we recommend the full song workflow below.

### 2. Full Song Generation: The Creative Duo

For creating complete, multi-part songs, CMC uses two powerful scripts that work hand-in-hand: `music_crafter.py` (the planner) and `song_generator.py` (the builder). The builder also covers SynthV‑ready lyric generation (see the Lyric quick guide below).

#### **Step 1 (Recommended): Plan Your Song with the Creative Assistant (`music_crafter.py`)**

This interactive wizard is the best starting point for a new song. It acts as your AI co-producer to translate a simple idea into a detailed musical blueprint.

**How it works:**
1.  You provide a high-level idea (e.g., genre, style, a feeling).
2.  The script asks you a few questions.
3.  It then uses the AI to generate a complete plan, including:
    - A detailed, MIDI-focused `inspiration` prompt.
    - A full list of suggested `instruments` with appropriate roles.
    - A complete song structure in `song_settings.json`, with descriptive labels and creative direction for each part (e.g., Intro, Verse, Chorus).
4.  After you confirm the plan, it automatically calls `song_generator.py` to create the music.

**To start the wizard, run:**
```bash
python music_crafter.py
```

#### **Step 2: Build Your Song with the Song Architect (`song_generator.py`)**

This is the core engine that reads the song plan and generates the final MIDI file. While `music_crafter.py` calls this script for you, you can also run it directly. This is useful if you want to:

-   Manually create or edit your `config.yaml` and `song_settings.json` files.
-   Re-generate a song using existing settings.
-   Run an optimization pass on a previously generated song.
-   Resume an interrupted generation.

`song_generator.py` has its own interactive menu for these tasks.

**To run the song engine directly, use:**
```bash
python song_generator.py
```

> **⚠️ Important Note on Token Usage & Generation Time:**
> Generating a full song is a powerful feature that can consume a significant amount of API tokens and may take a considerable amount of time to complete, depending on complexity. **If you are using a free API key, it is possible to exhaust your daily limit during the generation of a single song.**
>
> **Don't worry, your progress will be saved!** The `song_generator.py` script is designed for this. If generation stops, you can simply resume it from its interactive menu. It will pick up exactly where it left off.
>
> **💡 A Note on Quality:** While the process can be resource-intensive, the results from this two-step workflow can be strikingly good. The detailed plan created by `music_crafter.py` gives `song_generator.py` the context it needs to produce surprisingly cohesive and creative pieces.

## 📌 Note on legacy scripts

The original standalone scripts `part_generator.py`, `part_extender.py`, and `part_variator.py` remain in this repository and continue to work with the latest changes (including the new API key logic and the extended `config.yaml`).

- Going forward, these capabilities will be gradually integrated into the two main programs: `music_crafter.py` (planning/setup) and `song_generator.py` (generation), and expanded there.
- Until the integration is complete, you can keep using the three scripts as usual. Deprecation notices and migration guidance will be provided ahead of any breaking changes.
- Recommendation: For new workflows, prefer `music_crafter.py` and `song_generator.py` to benefit from advanced features (e.g., key rotation, resume support, automation options).

## ⚙️ Advanced usage and notes

- **API keys & rotation**: The `api_key` field supports either a single string or a list of keys. When a 429/quota error occurs, the system rotates to the next key automatically. 

- **YAML comments preservation**: `music_crafter.py` reads/writes `config.yaml` using ruamel.yaml in round-trip mode to preserve comments and quoting. You can safely annotate the config; comments will persist across saves.

- **Automation settings**: The `automation_settings` block controls expressive data:
  - `use_pitch_bend` (0/1): enables pitch bend events for expressive roles.
  - `use_sustain_pedal` (0/1): enables CC64 pedal events for sustaining instruments.
  - `use_cc_automation` (0/1): enables CC curves (e.g., filter sweeps).
  - `allowed_cc_numbers`: list of allowed CC numbers when CC automation is enabled (e.g., `[1, 10, 11, 74]`).

- **Call-and-response mode**: If `use_call_and_response` is `1`, eligible roles (e.g., `bass`, `chords`, `arp`, `guitar`, `lead`, `melody`, `vocal`) will alternate between “call” and “response” to improve interplay.

- **Token usage**: `max_output_tokens` sets the upper bound of model output length per call. Higher values improve completeness but increase cost. The scripts print token usage to help you tune settings.

- **Live model switching (hotkeys)**: While generating or optimizing, you can switch models on the fly (Windows hotkeys):
  - `1` = gemini-2.5-pro, `2` = gemini-2.5-flash, `3` = custom (prompt for name), `0` = set current as session default
  - `a` = toggle auto‑escalate: when enabled, if you are on flash and a track repeatedly fails (e.g., JSON errors, empty responses, MAX_TOKENS, or transient 5xx/timeout issues), the script automatically switches to pro after 6 failures and restarts the step.
  - The hotkey banner shows whether auto‑escalate is [ON]/[OFF]. Switches take effect as soon as the current request finishes; the step restarts with the new model.

- **Quality tip (Pro vs Flash)**:
  - If you want strong one‑shot results with fewer retries, run with `gemini-2.5-pro` from the start.
  - If you prefer lower cost, run `gemini-2.5-flash` plus an optimization pass; enable auto‑escalate so that problematic tracks automatically retry with `pro` when needed. This balances cost and reliability.

- **Why the context sometimes shows “Using 4/6 previous themes”**: The generator fits context under an internal character budget (`MAX_CONTEXT_CHARS`). In dynamic mode (`context_window_size: -1`) it includes as many previous parts as fit that budget, so the fraction (e.g., 4/6) is expected and will change if you or future versions adjust this character limit. Set `context_window_size` to a positive number to force a fixed number of previous parts, or to `0` to disable history entirely.

### 🎤 Lyric generation – quick guide

- Before you run
  - Put your chorus hook in double quotes in your prompt (short, singable). Example: "Breaking news: reality took the day off".
  - Keep the prompt style‑agnostic (no artist/model/file names). Meta words will not become lyrics.
  - Optional config: `pass_raw_prompt_to_stages`
    - `0` (default): Only Step‑0 sees the raw prompt (safer, fewer meta‑leaks).
    - `1`: Also pass raw prompt to Stage‑1/2 if you want very tight stylistic steering.
  - Prompt is optional: You can press Enter and generate without any prompt. Step‑0 will still plan sections and may propose a concise `hook_canonical` from context; you can accept, modify, or ignore it later.

- Generate a NEW vocal track
  - In the menu, pick “Generate NEW Vocal Track”.
  - Check the console “[Plan Summary]”: Hook Canonical, Chorus Lines, Imagery/Verbs, etc. If the hook isn’t right, fix the quoted line in your prompt and retry.
  - The system will then:
    - Stage‑1 (lyrics): favor whole words, use `-` sustains on tiny slots, keep hook words unbroken where possible.
    - Stage‑2 (notes): soft hook contiguity (compact segments if melody is chopped), avoid new onsets on very short notes.

- Generate lyrics for an EXISTING vocal track (word‑first)
  - Pick the existing vocal track.
  - The same shaping rules apply: fewer tokens when texture is dense; `-` on tiny notes; unbroken hook words when possible.
  - For heavily chopped melodies, expect compact hook segments instead of a forced contiguous chain.

- Read results efficiently
  - “[Plan Summary]” = quick sanity check (hook(s), chorus lines, repetition, palettes).
  - “[Vocal Plan]” = per‑part role + short hint to anticipate density and function.

- Troubleshooting
  - Meta words leaked into lyrics: keep the hook quoted; remove “in the style of …” from the raw prompt, or set `pass_raw_prompt_to_stages: 0`.
  - Too many short syllables: shorten instruction lines; the system already prefers `-` on micro‑notes; dense drops benefit from fewer words.
  - Silent/very sparse parts: allowed by design (intros/outros/breakdowns may plan silence or pure vowels).

- Exports
  - New vocal tracks export UST plus two Emvoice TXT files; you can also export a single‑track MIDI for the new vocal.

- **Resuming long runs**: The song generator saves progress and supports resuming via its interactive menu. You can also resume directly with:
  ```bash
  python song_generator.py --resume-file path/to/progress_run_*.json
  ```

- **Rebuilding from artifacts**: If you only need to export a `.mid` from an existing artifact or progress JSON without re-running generation, see the section "Artifact Builder (optional)" below.

- **MIDI channel policy**: Drums/percussion use MIDI Channel 10 (index 9). Melodic instruments are assigned sequentially across remaining channels, skipping 10.

- **Output filenames**:
  - `part_generator.py`: dynamic names like `<Genre>_<Key>_<Bars>bars_<BPM>bpm_<Timestamp>.mid`.
  - `part_extender.py`: `<OriginalName>_ext_<N>.mid`.
  - `part_variator.py`: `<OriginalName>_var_<N>.mid`.
  - Lyric exports: `lyrics_<TrackName>_<Timestamp>.ust`, `..._emvoice.txt`, `..._emvoice_by_part.txt`. You can optionally export a single‑track MIDI for the new vocal.

- **Dependencies**: See `requirements.txt` (google-generativeai, midiutil, PyYAML, colorama, ruamel.yaml, mido). Standard library modules are used for everything else.

## 🧠 Music Analyzer (optional)

`music_analyzer.py` helps analyze existing MIDI files and derive a clean, consistent plan that you can apply to CMC.

- What it does:
  - Extracts tracks, note timing in beats, initial BPM and time signature from a MIDI file.
  - Builds compact, LLM‑friendly summaries of tracks to avoid oversized prompts.
  - Can propose updates to `config.yaml`/`song_settings.json` and save them for use with the generators.
  - Integrated actions after analysis:
    - Optimize selected tracks only (keeps other tracks unchanged).
    - Add a new track to the whole song (context‑aware). Modes:
      1) Manual (your detailed description), 2) Minimal guided (short auto‑prompt),
      3) Guided full spec (you provide role + minimal idea; AI expands),
      4) Auto full spec (you choose role; AI proposes name/program/description).
    - Generate a NEW MIDI from the analyzed descriptions.
    - Full optimization of the ORIGINAL imported MIDI.
- When to use:
  - Use it when you want to use an existing MIDI structure as a starting point for generation or optimization.

## 🧱 Artifact Builder (optional)

`artifact_builder.py` can rebuild `.mid` files from existing final artifacts or progress files without re‑running a full generation.

- What it does:
  - Lists final artifacts and resumable progress JSONs and lets you pick one.
  - Extracts `config`, `themes`, and `length` and performs a pure MIDI export.
- When to use:
  - Quickly render a `.mid` from a saved artifact (e.g., after manual JSON tweaks).
  - Rebuild a partial song from a progress snapshot.

## 📌 Further advanced notes (addendum)

- Additional hotkeys (Windows, during waits):
  - `h` = halve previous‑themes context for the current step only
  - `d` = defer current track to the end of the queue
  - `s` = skip current wait immediately
  - `r` = reset all API key cooldown timers (forces an immediate probe)

- Quota/backoff behavior (daily limits):
  - If all API keys are classified as per‑day exhausted, CMC switches to an hourly probing cadence and tries all keys again each hour. As soon as any key works, normal operation resumes automatically.

- Config reload behavior:
  - `song_generator.py` reloads `config.yaml` before: Generate Again, Generate New, Optimize, and Optimize Existing Song. Changes to automation settings (pitch bend, sustain pedal/CC64, CC automation) and `use_call_and_response` take effect immediately.