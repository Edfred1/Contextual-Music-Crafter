# Contextual Music Crafter (CMC)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb)

> **Note:** You can also use Contextual Music Crafter directly in your browser with [Google Colab](https://colab.research.google.com/github/Edfred1/Contextual-Music-Crafter/blob/main/CMC.ipynb). No local installation required!

Contextual Music Crafter (CMC) is an intelligent, context-aware MIDI music generation tool that leverages the power of Google's Generative AI to compose multi-track musical pieces. Unlike simple random note generators, CMC builds songs iteratively, instrument by instrument. Each new part is intelligently composed in response to the parts that have already been written, creating cohesive and musically interesting results.

The entire creative direction of the music—from genre and tempo to instrumentation and musical complexity—is controlled through a single, easy-to-edit `config.yaml` file, making it accessible to both developers and musicians.

## ✨ Features

-   **Context-Aware Composition:** The core of CMC. The AI analyzes the existing tracks before generating a new one, leading to a more coherent and "human-like" composition.
-   **Highly Configurable:** Control genre, BPM, key, scale, and instrumentation through the `config.yaml` file.
-   **Advanced Musical Intelligence:** The AI is guided by sophisticated prompts that understand musical roles (bass, harmony, melody), arrangement position, and even techniques like "call and response."
-   **Iterative Generation:** The script can be configured to generate multiple complete song variations in a single run, perfect for exploring different ideas quickly.
-   **Simple & Robust:** The script is self-contained and easy to run, with clear feedback during the generation process.

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

CMC comes with three main scripts, each for a different creative purpose:

### 1. `part_generator.py` - Create from Scratch

This is the original script. It generates a complete musical piece from zero based entirely on the settings in your `config.yaml`.

1.  **Configure your desired sound** in `config.yaml`.
2.  **Run the script** from your terminal:
    ```bash
    python part_generator.py
    ```
3.  **Enter the desired length** for the composition when prompted.
4.  The script will generate a brand new MIDI file in the project directory.

### 2. `part_extender.py` - Extend an Existing MIDI File

This script loads an existing MIDI file and adds new instrument parts to it, using the original tracks as musical context. It's perfect for taking a simple loop and building a full arrangement around it.

1.  **Run the script** from your terminal:
    ```bash
    python part_extender.py
    ```
2.  **Select the MIDI file** you want to extend from the list of found files.
3.  The script will use the instruments defined in your `config.yaml` to add new layers to your selected MIDI file.
4.  A new file with the `_ext` suffix will be saved (e.g., `MyOriginalFile_ext_1.mid`).

### 3. `part_variator.py` - Create Variations of Existing Parts

This script loads a MIDI file and lets you create new variations of specific tracks within it. It's ideal for creating alternative melodies, basslines, or drum patterns without changing the rest of the song.

1.  **Run the script** from your terminal:
    ```bash
    python part_variator.py
    ```
2.  **Select the MIDI file** you want to vary.
3.  **Choose which track(s)** you want to create a variation for.
4.  The script will use the `inspiration` prompt from your `config.yaml` as creative direction to generate a new version of the selected parts.
5.  A new file with the `_var` suffix will be saved (e.g., `MyOriginalFile_var_1.mid`).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Edfred1/Contextual-Music-Crafter/issues) of the repository.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.

---

### Optional: For Advanced Users (Git)

If you are familiar with Git, you can also clone the repository instead of downloading the ZIP:
```bash
git clone https://github.com/Edfred1/Contextual-Music-Crafter.git
cd Contextual-Music-Crafter
```
