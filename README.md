# Contextual Music Crafter (CMC)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb)

> **⚠️ Important Note:**
> Google has removed the free tier for the Gemini API. Currently, this affects the **Pro model** (gemini-2.5-pro), which now requires a paid API key. The **Flash model** (gemini-2.5-flash) may still work with free tier access, but please be aware that API pricing and availability may change.

> **Note: Try it in your Browser!**
> The included [Google Colab Notebook](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb) has been updated to run the full suite of scripts directly in your browser with no local installation required. Please be aware that this Colab integration is considered experimental and has not been as extensively tested as running the scripts on a local machine. For the most stable experience, we recommend a local installation.

Contextual Music Crafter (CMC) is an intelligent, context-aware MIDI music generation tool that leverages the power of Google's Gemini to compose multi-track musical pieces. Unlike simple random note generators, CMC builds songs iteratively, instrument by instrument. Each new part is intelligently composed in response to the parts that have already been written, creating cohesive and musically interesting results. CMC also includes a Synthesizer‑ready lyric and vocal‑melody pipeline aimed at Synthesizer V Studio 2 (UST export). UST files may also work in OpenUtau and similar UTAU‑compatible tools. In addition, Emvoice TXT exports are produced for simple copy/paste into Emvoice.

The entire creative direction of the music is guided through an interactive setup process, making it accessible to both developers and musicians.

<a id="recent-highlights"></a>
## 🆕 Recent Highlights

- **Music Analyzer**: Analyze your multi‑track MIDI into a compact, LLM‑friendly plan/artifact with integrated actions (optimize selected tracks, add a context‑aware track, or generate a NEW MIDI from the analyzed descriptions). See [Music Analyzer (optional)](#music-analyzer-optional).
- **Lyrics & Vocal pipeline**: Generate lyrics for an existing melody or create a NEW vocal line (notes + lyrics + UST). Exports Synthesizer V UST and Emvoice TXT. See [Lyrics and Vocal Melody – quick guide](#lyrics-and-vocal-melody--quick-guide).
- **MIDI Utilities**: A standalone [MIDI Merger tool](MIDI%20merger/README_midi_merger.md) is included to combine multiple `.mid` files into one multi-track project (see `MIDI merger/` folder).

## Table of Contents

- [Recent Highlights](#recent-highlights)
- [Toolkit overview](#toolkit-overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [How to Use](#how-to-use)
  - [0. (Optional) Analyze an existing MIDI first](#0-optional-analyze-an-existing-midi-first)
  - [1. Full Song Generation: The Creative Duo](#1-full-song-generation-the-creative-duo)
- [Creative Workflows & Ideas](#creative-workflows--ideas)
- [Advanced usage and notes](#advanced-usage-and-notes)
- [Lyrics and Vocal Melody – quick guide](#lyrics-and-vocal-melody--quick-guide)
- [General notes (resume & exports)](#general-notes-resume--exports)
- [Music Analyzer (optional)](#music-analyzer-optional)
- [Melody Variation Generator (optional)](#melody-variation-generator-optional)
- [Artifact Builder (optional)](#artifact-builder-optional)
- [Further advanced notes (addendum)](#further-advanced-notes-addendum)
- [Roadmap (Ideas)](#️-roadmap-ideas)

<a id="toolkit-overview"></a>
## 🧰 Toolkit overview

- **music_crafter.py**: Interactive planner (co‑producer). Creates/updates `config.yaml` and `song_settings.json`, then launches the generator.
- **song_generator.py**: Core engine that builds, optimizes, and resumes full songs from the plan. Includes the lyric generator. Exports SynthV‑ready UST, Emvoice TXT, and (optionally) a single‑track MIDI for vocals.
- **music_analyzer.py**: MIDI analysis tool that can analyze existing files, optimize selected tracks, add new context-aware tracks, and generate new MIDI from analyzed descriptions.
- **melody_variation_generator.py**: Generates AI‑powered variations of a selected track from MIDI files or Final JSON artifacts. Creates multiple creative variations (e.g., syncopation, octaves, fills) while maintaining musical coherence.
- **artifact_builder.py**: Utility to rebuild `.mid` from final artifacts or progress JSONs without re‑running generation.
- **MIDI merger/** (folder): Standalone tool to merge multiple MIDI files into one multi-track project. See [MIDI merger README](MIDI%20merger/README_midi_merger.md).

<a id="features"></a>
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

<a id="installation--setup"></a>
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

<a id="how-to-use"></a>
## 🎹 How to Use

CMC provides a suite of tools for different creative needs, from low-cost loop creation to full-scale song composition. **We strongly recommend starting with the 'Core Generators'** to understand the process and conserve your API tokens before moving to the more advanced tools.

### 0. (Optional) Analyze an existing MIDI first

If you want to start from an existing MIDI structure, run a quick analysis first and derive a compact plan you can feed into the generator. After analysis, you can directly optimize a subset of tracks or add a new context‑aware track across all parts from within the analyzer. See the section "Music Analyzer (optional)" below for details.

### 1. Full Song Generation: The Creative Duo

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

<a id="creative-workflows--ideas"></a>
## 🎨 Creative Workflows & Ideas

CMC works best as a creativity amplifier: a sketchbook, exploration engine, and co‑producer. The ideas below reference existing sections to avoid repetition, while giving you practical routes to try.

- **Start from your own MIDI (Analyzer → Generator)**
  - Create a multi‑track MIDI in your DAW.
  - Run the analyzer to derive a compact, LLM‑friendly plan and artifact (see [Music Analyzer (optional)](#music-analyzer-optional)):
    ```bash
    python music_analyzer.py
    ```
  - Then use the song engine (see [How to Use](#how-to-use)) to either add a new, context‑aware instrument across all parts or run an optimization cycle to refine what’s already there:
    ```bash
    python song_generator.py
    ```
  - Outcome: you preserve your song’s identity while exploring fresh harmonic, rhythmic, and textural ideas.

- **Iterative variation cycles on your own material**
  - Load the artifact produced by the analyzer in the song engine and do short optimize → listen → tweak rounds.
  - Keep changes small per round to maintain "family resemblance" while discovering related variations. See also [Advanced usage and notes](#advanced-usage-and-notes) for model/cost tips.

- **Generate track variations with AI**
  - Use the [Melody Variation Generator](#melody-variation-generator-optional) to create multiple creative variations of any track from your MIDI or Final JSON.
  - Select specific parts to vary, let AI suggest variation types (syncopation, octaves, fills, etc.), and export original + variations in one MIDI file.
  - Perfect for exploring different takes on a melody, bassline, or drum pattern while maintaining musical coherence.

- **AI reinterpretation from analyzer summaries**
  - Analyze your multi‑track MIDI to produce compact section/track descriptions.
  - In the analyzer’s integrated actions, choose “Generate a NEW MIDI from the analyzed descriptions” to let the model re‑interpret your song from the textual plan.
  - Tips: adjust bars‑per‑section, roles, or the `inspiration` to steer fidelity vs. novelty; save multiple artifacts to compare different “takes”.

- **Targeted part‑to‑part transition building**
  - In your DAW, export only the tracks from the current section that should carry over into the next section as a multi‑track MIDI.
  - Analyze that MIDI to get a concise plan, then use the analyzer’s integrated actions to add NEW, context‑aware tracks specifically for the upcoming section/change.
  - Steer the result via your `inspiration` prompt and motif repetition to keep continuity while introducing contrast.

- **Lyrics and vocals on top of your MIDI**
  - Once a final artifact exists (from analyzer or generator), open the lyrics menu in the song engine to: generate lyrics for an existing melody or create a NEW vocal track (notes + lyrics + UST) across all parts.
  - Exports: Synthesizer V Studio 2 UST and two Emvoice TXT formats. See [Lyrics and Vocal Melody – quick guide](#lyrics-and-vocal-melody--quick-guide) for details.

- **Style frames and creative constraints**
  - In `config.yaml`, keep the “inspiration” prompt focused on feel, phrasing, and energy arc (avoid plugin/FX terms).
  - Try constraints: fixed section lengths, call‑and‑response enabled, intentionally sparse intros, or “one silent instrument per section” to create musical negative space.

- **Quality and cost tips**
  - Prefer short generate→optimize loops; resume long runs instead of restarting. Use dynamic context for cohesion.
  - Reliability: start with `gemini-2.5-pro`. Economy: `gemini-2.5-flash` + an optimization pass; auto‑escalate helps on tricky tracks (see [Advanced usage and notes](#advanced-usage-and-notes)).

- **On publishing and monetization (friendly note)**
  - CMC is most fun as a creativity tool—sketching, learning, and exploring musical ideas. If you choose to release or sell what you make, that’s your call. Please be mindful of originality, platform policies, and any third‑party material.

<a id="advanced-usage-and-notes"></a>
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

<a id="lyrics-and-vocal-melody--quick-guide"></a>
### 🎤 Lyrics and Vocal Melody – quick guide

- Entry point
  - In the `song_generator.py` menu, select “Generate Lyrics for a Track (Artifact/Progress)”. This option appears once a final artifact or a resumable progress file exists.
  - No artifact yet? Create one by generating a song, or analyze an existing multi‑track MIDI with `music_analyzer.py` and then run a generation/optimization so that an artifact/progress file is produced.

- Branches
  - Existing track (lyrics for existing melody): Pick a listed track; only lyrics are generated for its existing melody.
  - Generate NEW Vocal Track (notes + lyrics + UST): Choose the extra option shown after the track list to create a new vocal line across all parts (lyrics‑first, then fitting notes), incl. UST & Emvoice exports.

- Generate a NEW vocal track
  - Check the console “[Plan Summary]”: Hook Canonical, Chorus Lines, Imagery/Verbs, etc. If the hook isn’t right, adjust the quoted hook and retry.
  - The system will then:
    - Stage‑1 (lyrics): favor whole words, use `-` sustains on tiny slots, keep hook words unbroken where possible.
    - Stage‑2 (notes): soft hook contiguity (compact segments if melody is chopped), avoid new onsets on very short notes.

- Generate lyrics for an EXISTING vocal track (word‑first)
  - Pick the existing vocal track.
  - For heavily chopped melodies, expect compact hook segments instead of a forced contiguous chain.

- Read results efficiently
  - “[Plan Summary]” = quick sanity check (hook(s), chorus lines, repetition, palettes).
  - “[Vocal Plan]” = per‑part role + short hint to anticipate density and function.

- Troubleshooting
  - Meta words leaked: keep hook quoted; or set `pass_raw_prompt_to_stages: 0`.
  - Too many short syllables: shorten instruction lines.
  - Very sparse parts can be intentional (intro/breakdown).

- Exports
  - Outputs UST plus two Emvoice TXT files; optional single‑track vocal `.mid`.

<a id="general-notes-resume--exports"></a>
### General notes (resume & exports)

- **Resuming long runs**: The song generator saves progress and supports resuming via its interactive menu. You can also resume directly with:
  ```bash
  python song_generator.py --resume-file path/to/progress_run_*.json
  ```

- **Rebuilding from artifacts**: If you only need to export a `.mid` from an existing artifact or progress JSON without re-running generation, see the section "Artifact Builder (optional)" below.

- **MIDI channel policy**: Drums/percussion use MIDI Channel 10 (index 9). Melodic instruments are assigned sequentially across remaining channels, skipping 10.

- **Output filenames**:
  - Generated songs: dynamic names based on genre, key, tempo and timestamp (e.g., `Progressive_Psytrance_F#minor_140bpm_20250101-120000.mid`).
  - Lyric exports: `lyrics_<TrackName>_<Timestamp>.ust`, `..._emvoice.txt`, `..._emvoice_by_part.txt`. You can optionally export a single‑track MIDI for the new vocal.

- **Dependencies**: See `requirements.txt` (google-generativeai, midiutil, PyYAML, colorama, ruamel.yaml, mido). Standard library modules are used for everything else.

<a id="music-analyzer-optional"></a>
## 🧠 Music Analyzer (optional)

`music_analyzer.py` helps analyze existing MIDI files and derive a clean, consistent plan that you can apply to CMC.

- What it does:
  - Extracts tracks, note timing in beats, initial BPM and time signature from a MIDI file.
  - Builds compact, LLM‑friendly summaries of tracks to avoid oversized prompts.
  - Can propose updates to `config.yaml`/`song_settings.json` and save them for use with the generators.
  - Saves an analysis artifact (`analysis_run_*.json`) that you can use as a starting point for generation/optimization.
  - Integrated actions after analysis:
    - Optimize selected tracks only (keeps other tracks unchanged).
    - Add a new track to the whole song (context‑aware). Modes:
      1) Manual (your detailed description), 2) Minimal guided (short auto‑prompt),
      3) Guided full spec (you provide role + minimal idea; AI expands),
      4) Auto full spec (you choose role; AI proposes name/program/description).
    - Generate a NEW MIDI from the analyzed descriptions.
    - Full optimization of the ORIGINAL imported MIDI.
- When to use:
  - Use it when you want to use an existing multi‑track MIDI as a starting point, and produce artifacts that the generator or artifact builder can pick up.

<a id="melody-variation-generator-optional"></a>
## 🎵 Melody Variation Generator (optional)

`melody_variation_generator.py` generates AI‑powered variations of a selected track from MIDI files or Final JSON artifacts, creating multiple creative takes while maintaining musical coherence.

- What it does:
  - Accepts input from MIDI files (`.mid`, `.midi`) or Final JSON artifacts (`final_run_*.json`).
  - Analyzes which parts contain notes for the selected track, showing detailed metrics (note count, density, pitch range, velocity, duration).
  - Lets you select specific parts to vary (or use all parts with notes).
  - Uses AI to suggest creative variation types tailored to the track's role (e.g., "Syncopation", "Octaves", "Triplets", "Ghost Notes", "Fills").
  - Generates multiple variations (1-10) using different techniques while respecting:
    - The original musical intent and motifs
    - Key/scale constraints
    - Context from other tracks in the same parts
    - Smooth transitions between parts
    - Role-specific polyphony rules (monophonic, expressive monophonic, or polyphonic)
  - Exports a MIDI file containing the original track (for selected parts) plus all generated variations as separate tracks.
  - Supports MIDI automations (pitch bend, CC curves) if present in the original track.

- How it works:
  1. Select input: Choose a MIDI file or Final JSON artifact.
  2. Select track: Pick the track you want to vary from the available tracks.
  3. Analyze parts: Review which parts contain notes (with detailed metrics) and select which parts to vary.
  4. Generate variation types: AI suggests creative variation techniques based on the track's role and musical context.
  5. Generate variations: AI creates multiple variations, processing parts in chunks to manage token limits.
  6. Export: Saves a MIDI file with the original track and all variations.

- When to use:
  - Explore different takes on a melody, bassline, or drum pattern.
  - Generate alternative versions for A/B testing or layering.
  - Create variations that maintain musical coherence while adding creative interest.
  - Experiment with different techniques (syncopation, octaves, fills, etc.) on existing material.

- Usage:
  ```bash
  python melody_variation_generator.py
  ```

- Output:
  - MIDI file named `Variations_<TrackName>_<Timestamp>.mid` containing the original track (selected parts only) and all generated variations as separate tracks.

<a id="artifact-builder-optional"></a>
## 🧱 Artifact Builder (optional)

`artifact_builder.py` can rebuild `.mid` files from existing final artifacts or progress files without re‑running a full generation.

- What it does:
  - Lists final artifacts and resumable progress JSONs and lets you pick one.
  - Extracts `config`, `themes`, and `length` and performs a pure MIDI export.
- When to use:
  - Quickly render a `.mid` from a saved artifact (e.g., after manual JSON tweaks).
  - Rebuild a partial song from a progress snapshot.

<a id="further-advanced-notes-addendum"></a>
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

<a id="roadmap-ideas"></a>
## 🗺️ Roadmap (Ideas)

These are potential features being considered for future development. They are ideas, not commitments:

- **MIDITool**: Unified interface to simplify workflow across all tools (analysis, generation, optimization in one place).
- **Variable Section Lengths**: Music Crafter support for different bar lengths per section instead of fixed-length parts (e.g., 4-bar intro, 16-bar verse, 2-bar riser, 8-bar drop, 12-bar breakdown—each tailored to the musical flow).
- **Template System**: Save and reuse your favorite instrument/genre setups for faster project starts.
- **SynthV Format Support**: Support for SynthV's native file format to enable automation of voice expression parameters (instead of only UTAU format), allowing more advanced vocal synthesis control.

**Ideas and suggestions welcome!** If you have feature requests or workflow improvements in mind, feel free to open an issue or discussion on GitHub.
