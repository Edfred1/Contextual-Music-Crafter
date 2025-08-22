# Contextual Music Crafter (CMC)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb)

> **Note: Try it in your Browser!**
> The included [Google Colab Notebook](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb) has been updated to run the full suite of scripts directly in your browser with no local installation required. Please be aware that this Colab integration is considered experimental and has not been as extensively tested as running the scripts on a local machine. For the most stable experience, we recommend a local installation.

Contextual Music Crafter (CMC) is an intelligent, context-aware MIDI music generation tool that leverages the power of Google's Gemini to compose multi-track musical pieces. Unlike simple random note generators, CMC builds songs iteratively, instrument by instrument. Each new part is intelligently composed in response to the parts that have already been written, creating cohesive and musically interesting results.

The entire creative direction of the music is guided through an interactive setup process, making it accessible to both developers and musicians.

## ✨ Features

-   **Go Beyond Loops:** Generate complete songs with multiple, distinct sections (intro, verse, chorus) to tell a musical story.
-   **Guided Song Setup:** Use an interactive wizard to go from a simple idea to a full song structure, complete with AI-suggested instrumentation and creative direction for each part.
-   **Contextual Composition:** The AI listens to the existing music before adding a new part, creating cohesive arrangements that sound like they were composed with intent.
-   **Polish Your Tracks:** After generation, an AI "producer" can optimize each part to enhance groove, dynamics, and overall musicality for a more professional sound.
-   **Resumable Workflow:** Never lose your work. If a long generation job is interrupted, you can resume it right where you left off.
-   **Fine-Grained Control:** While the interactive assistant is powerful, you can always dive into the `config.yaml` and `song_settings.json` files to manually tweak every detail.

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
    -   `model_name`: The specific generative model to use (e.g., "gemini-2.5-flash-preview-05-20").
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

### 1. Optional Legacy Tools (single parts & quick edits)

If you prefer low-cost experiments or quick edits, you can use the original scripts controlled by `config.yaml`:

- `part_generator.py`: generate a single, complete musical part from scratch
- `part_extender.py`: add new parts to an existing MIDI file
- `part_variator.py`: create variations of selected tracks in a MIDI file

Each script reads your `config.yaml` and writes a new `.mid`. For most users, we recommend the full song workflow below.

### 2. Full Song Generation: The Creative Duo

For creating complete, multi-part songs, CMC uses two powerful scripts that work hand-in-hand: `music_crafter.py` (the planner) and `song_generator.py` (the builder).

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

- **Resuming long runs**: The song generator saves progress and supports resuming via its interactive menu. You can also resume directly with:
  ```bash
  python song_generator.py --resume-file path/to/progress_run_*.json
  ```

- **MIDI channel policy**: Drums/percussion use MIDI Channel 10 (index 9). Melodic instruments are assigned sequentially across remaining channels, skipping 10.

- **Output filenames**:
  - `part_generator.py`: dynamic names like `<Genre>_<Key>_<Bars>bars_<BPM>bpm_<Timestamp>.mid`.
  - `part_extender.py`: `<OriginalName>_ext_<N>.mid`.
  - `part_variator.py`: `<OriginalName>_var_<N>.mid`.

- **Dependencies**: See `requirements.txt` (google-generativeai, midiutil, PyYAML, colorama, ruamel.yaml, mido). Standard library modules are used for everything else.
