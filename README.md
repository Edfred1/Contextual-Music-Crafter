# Contextual Music Crafter (CMC)

Contextual Music Crafter (CMC) is an intelligent, context-aware MIDI music generation tool that leverages the power of Google's Gemini AI to compose multi-track musical pieces. Unlike simple random note generators, CMC builds songs iteratively, instrument by instrument, where each new part is intelligently composed in response to the parts that have already been written. This creates cohesive, musically interesting results.

The entire creative direction of the music—from genre and tempo to instrumentation and musical complexity—is controlled through a single, easy-to-edit `config.yaml` file, making it accessible to both developers and musicians.

## ✨ Features

-   **Context-Aware Composition:** The core of CMC. Drums know about the bass, melodies know about the harmony, creating a true "virtual jam session."
-   **Highly Configurable:** Control genre, BPM, key, scale, instrumentation, and even the compositional order of instruments through `config.yaml`.
-   **Advanced Musical Prompts:** The AI is guided by sophisticated musical principles like "call-and-response," the use of silence (rests), and creating complementary rhythms, leading to more dynamic and professional-sounding results.
-   **Iterative Variation:** Generate endless variations of your creations. Each variation builds upon the previous one, allowing for evolutionary development of musical ideas.
-   **Flexible and Robust:** Automatically finds its configuration file and saves MIDI files in its own directory, ensuring predictable behavior.

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
    Open your new `config.yaml` in a text editor. Replace `"YOUR_API_KEY_HERE"` with your actual Google AI API key.
    ```yaml
    # API Configuration
    api_key: "YOUR_API_KEY_HERE"
    ```

3.  **Customize your music:**
    Modify the fields under `Musical Base Settings`, `Instrument Configuration`, and `Advanced Generation Control` to your liking. This is where the magic happens!

    -   `inspiration`: Artists that will influence the style.
    -   `genre`: The musical genre.
    -   `bpm`: Tempo in beats per minute.
    -   `key_scale`: The key and scale (e.g., "C# minor", "E phrygian").
    -   `instruments`: Define your band. Assign a MIDI program number to each instrument name.
    -   `generation_order`: **Crucially**, define the order in which instruments are composed. A good starting point is the rhythm section first (`drums`, `bass`).

## 🎹 How to Use

Simply run the main script from your terminal:

```bash
python part_generator.py
```

-   The script will prompt you to choose a length (in bars) for your composition.
-   It will then generate the track, instrument by instrument, showing its progress.
-   The final MIDI file will be saved in the project directory with a descriptive name.
-   After a track is generated, you will be asked if you want to create a variation of it.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Edfred1/Contextual-Music-Crafter/issues).

## 📄 License

This project is licensed under the MIT License - see the `LICENSE.md` file for details. 