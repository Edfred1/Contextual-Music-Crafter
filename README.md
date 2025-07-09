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
    **Security Note:** It's highly recommended to use an environment variable or a secrets management tool rather than hardcoding your API key directly in the configuration file for public repositories.
3.  **Customize Your Music:**
    Adjust the parameters in `config.yaml` to define your desired musical output.

    -   `api_key`: Your Google AI API Key.
    -   `model_name`: The specific generative model to use (e.g., "gemini-2.5-flash-preview-05-20").
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

### 1. The Core Generators: The Recommended Starting Point

These are the original, direct scripts. They are perfect for developing individual musical ideas, creating loops, or editing existing MIDI files with minimal API usage. Their entire creative direction is controlled via the `config.yaml` file.

-   **`part_generator.py` - Create from Scratch**

    This is the simplest and most token-efficient entry point. The script generates a single, complete musical piece from scratch based on the settings in your `config.yaml`.

    1.  **Configure your desired sound** in `config.yaml`.
    2.  **Run the script** from your terminal:
        ```bash
        python part_generator.py
        ```
    3.  Enter the desired length for the composition when prompted. The script will create a brand new MIDI file in the project directory.

-   **`part_extender.py` - Extend an Existing MIDI File**

    This script loads an existing MIDI file and adds new instrument parts to it, using the original tracks as musical context. It's perfect for building a full arrangement around a simple loop.

    1.  Run the script and select the MIDI file you want to extend.
        ```bash
        python part_extender.py
        ```
    2.  The script will use the instruments defined in your `config.yaml` to add new layers. A new file with the `_ext` suffix will be saved.

-   **`part_variator.py` - Create Variations of Existing Parts**

    This script loads a MIDI file and lets you create new variations of specific tracks within it. It's ideal for creating alternative melodies, basslines, or drum patterns without changing the rest of the song.

    1.  Run the script and select the MIDI file you want to vary.
        ```bash
        python part_variator.py
        ```
    2.  Choose which track(s) you want to create a variation for. A new file with the `_var` suffix will be saved.

### 2. Advanced Tool: The Song Architect (`song_generator.py`)

For advanced users who are ready to compose a full song, `song_generator.py` is the next step. This powerful script generates complete, multi-part songs (e.g., intro, verse, chorus) with different musical directions for each section.

It uses not only `config.yaml` but also `song_settings.json` to define the structure of the entire song.

> **⚠️ Important Note on Token Usage:**
> Generating a full song with multiple complex parts is a powerful feature that can consume a significant amount of API tokens. **If you are using a free API key, it is possible to exhaust your daily limit during the generation of a single song.**
>
> **Don't worry, your progress will be saved!** The script is designed for this. If generation stops, you can simply resume it the next day by running the script with the `--resume` flag and the saved progress file. It will pick up exactly where it left off.
>
> **💡 A Note on Quality:** While the token usage is high, the results from generating a complete, multi-part song can be strikingly good, especially for well-defined musical concepts and genres that the AI model is familiar with. The contextual generation process truly shines when it has a full song structure to work with, often producing surprisingly cohesive and creative pieces.

### 3. Advanced Tool: The Creative Assistant (`music_crafter.py`)

This script is an interactive wizard that guides you through the entire creative process, from defining your idea to generating a complete song. While it uses the token-intensive `song_generator.py` for full song creation, it's the most powerful way to unlock CMC's full potential and produce impressive, structured musical pieces.

**To start, simply run the script:**
```
```