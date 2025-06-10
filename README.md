# Contextual Music Crafter (CMC)

Contextual Music Crafter (CMC) is an intelligent, context-aware MIDI music generation tool that leverages the power of Google's Generative AI to compose multi-track musical pieces. Unlike simple random note generators, CMC builds songs iteratively, instrument by instrument. Each new part is intelligently composed in response to the parts that have already been written, creating cohesive and musically interesting results.

The entire creative direction of the music—from genre and tempo to instrumentation and musical complexity—is controlled through a single, easy-to-edit `config.yaml` file, making it accessible to both developers and musicians.

## ✨ Features

-   **Context-Aware Composition:** The core of CMC. The AI analyzes the existing tracks before generating a new one, leading to a more coherent and "human-like" composition.
-   **Highly Configurable:** Control genre, BPM, key, scale, and instrumentation through the `config.yaml` file.
-   **Advanced Musical Intelligence:** The AI is guided by sophisticated prompts that understand musical roles (bass, harmony, melody), arrangement position, and even techniques like "call and response."
-   **Iterative Generation:** The script can be configured to generate multiple complete song variations in a single run, perfect for exploring different ideas quickly.
-   **Simple & Robust:** The script is self-contained and easy to run, with clear feedback during the generation process.

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.7+
-   An active Google AI API Key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Edfred1/Contextual-Music-Crafter.git
    cd Contextual-Music-Crafter
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

1.  **Open `config.yaml`** in a text editor.

2.  **Set your API Key:**
    Find the `api_key` field and replace `"YOUR_GOOGLE_AI_API_KEY"` with your actual key.
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

1.  **Configure your desired sound** in `config.yaml`.
2.  **Run the script** from your terminal:
    ```bash
    python part_generator.py
    ```
3.  **Enter the desired length** for the composition when prompted.
4.  The script will generate the track, instrument by instrument, showing its progress in the console.
5.  The final MIDI file(s) will be saved in the project directory with a descriptive name based on your configuration (e.g., `House_Music_Funky_Bassline_Cminor_125bpm_20231027-123456.mid`).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Edfred1/Contextual-Music-Crafter/issues) of the repository.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE.md` file for details. 