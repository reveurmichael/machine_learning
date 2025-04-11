
# Video Transcription and Translation with AI

## Introduction

Welcome to this tutorial on using AI for video transcription and translation! In this session, you'll learn how to leverage powerful AI models to automatically:

1. Transcribe speech from videos to create English subtitles
2. Translate those subtitles from English to Chinese
3. Create bilingual SRT subtitle files ready for use with video players

This tutorial introduces two key AI technologies:
- **faster-whisper**: An optimized implementation of OpenAI's Whisper model for speech recognition
- **Helsinki-NLP/opus-mt-en-zh**: A machine translation model specifically trained for English to Chinese translation

By the end of this tutorial, you'll have created your own Python script capable of processing videos and generating bilingual subtitles automatically!


## Installation

### 1. Install Required Libraries

Install the necessary Python packages:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install faster-whisper
pip install ffmpeg-python
```

### 2. Install FFmpeg

FFmpeg is essential for audio processing. Here's how to install it:

**For Windows:**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (Windows builds)
2. Extract the files to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to your system PATH:
   - Search for "Edit environment variables" in Windows
   - Select "Environment Variables"
   - Edit the "Path" variable
   - Add the path to the FFmpeg bin folder (e.g., `C:\ffmpeg\bin`)
   - Click OK and restart any open command prompts

**For macOS:**
Using Homebrew:
```bash
brew install ffmpeg
```

### 3. Verify Installation

Test your setup:

```python
import torch
from transformers import pipeline
from faster_whisper import WhisperModel

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
```

## Understanding Hugging Face

Before we dive into the code, let's talk about Hugging Face - often called the "GitHub of AI."

### What is Hugging Face?

Hugging Face is a company and community platform focused on natural language processing (NLP) and machine learning. It has become the central hub for sharing, discovering, and collaborating on machine learning models and datasets.

Key aspects of Hugging Face:
- **Model Hub**: Repository of thousands of pre-trained models
- **Transformers Library**: Easy-to-use API for state-of-the-art models
- **Datasets**: Collection of public datasets for training and testing
- **Spaces**: Interactive demos of machine learning applications
- **Community**: Collaboration between researchers, developers, and organizations

Fun fact: Hugging Face is a **French** company founded in 2016!

For our tutorial, we'll use two models from Hugging Face:
- The Whisper model (via faster-whisper, an optimized implementation)
- Helsinki-NLP's English-to-Chinese translation model

## Building Our Video Transcription and Translation Script

Now, let's create our script step by step. We'll build a Python program that:
1. Finds video files in a directory
2. Transcribes speech to English text using Whisper
3. Translates the text to Chinese
4. Formats and saves as SRT subtitle files

### Step 1: Import Required Libraries

Create a new Python file named `video_transcription.py` and add these imports:

```python
import os
import glob
import subprocess
from pathlib import Path
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from faster_whisper import WhisperModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"  # Force use of PyTorch
os.environ["USE_TF"] = "0"  # Disable TensorFlow

```

### Step 2: Create a Function to Find Video Files

```python
def get_video_files(directory="./"):
    """Get all video files in the specified directory."""
    video_extensions = ["*.mp4", "*.mkv", "*.webm", "*.flv"]
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext)))

    return video_files
```

This function searches for video files with common extensions in the specified directory.

### Step 3: Create a Function to Transcribe Videos

```python
def transcribe_video(video_path):
    """Transcribe video using Whisper model."""
    print(f"Transcribing {video_path}...")

    # Using faster-whisper for better performance
    model_size = "medium"
    # Run on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # Load the Whisper model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe the audio
    segments, _ = model.transcribe(
        video_path, language="en", task="transcribe", vad_filter=True
    )

    # Format as SRT
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()

        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")

    return srt_content
```

This function:
- Initializes the Whisper model with the "medium" size
- Uses GPU if available
- Transcribes the video's audio
- Formats the transcription into SRT format

### Step 4: Create a Helper Function for SRT Timestamps

```python
def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
```

This function converts time in seconds to the SRT format (HH:MM:SS,mmm).

### Step 5: Create a Function to Translate Text

```python
def translate_text(text, translator):
    """Translate text from English to Chinese."""
    if not text.strip():
        return ""

    translated = translator(text, max_length=512)
    return translated[0]["translation_text"]
```

This function uses the Helsinki-NLP model to translate English text to Chinese.

### Step 6: Create the Main Function

```python
def main():
    # Load translation model
    print("Loading translation model...")
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)

    video_files = get_video_files()

    if not video_files:
        print("No video files found in the current directory.")
        return

    print(f"Found {len(video_files)} video file(s).")

    for video_path in video_files:
        video_filename = Path(video_path).stem
        srt_path = f"{video_filename}.srt"

        # Transcribe video
        srt_content = transcribe_video(video_path)

        # Translate each subtitle line
        print(f"Translating subtitles for {video_path}...")
        translated_srt = []
        for line in srt_content:
            parts = line.strip().split("\n")
            if len(parts) >= 3:  # Valid subtitle entry
                subtitle_index = parts[0]
                timestamp = parts[1]
                english_text = parts[2]

                # Translate to Chinese
                chinese_translation = translate_text(english_text, translator)

                # Combine English and Chinese
                combined_text = (
                    f"{english_text}\n{chinese_translation}"
                    if chinese_translation
                    else english_text
                )

                translated_srt.append(
                    f"{subtitle_index}\n{timestamp}\n{combined_text}\n\n"
                )

        # Write to SRT file
        with open(srt_path, "w", encoding="utf-8") as f:
            f.writelines(translated_srt)

        print(f"Subtitles with translation saved to {srt_path}")


if __name__ == "__main__":
    main()
```

This main function:
- Loads the Helsinki-NLP translation model
- Finds all video files in the current directory
- For each video:
  - Transcribes it using Whisper
  - Translates each subtitle line
  - Creates a bilingual subtitle file
  - Saves it as an SRT file

### Step 7: Run Your Script

Save your complete script and run it:

```bash
python video_transcription.py
```

## Testing with Sample Videos

You'll be provided with three test videos in your WeChat group:
1. "But what is a neural network？ ｜ Deep learning chapter 1 [aircAruvnKk].mp4"
2. "Be Careful of AI Face-Swapping ｜ Stranger Copycat ｜ Safety Cartoons for Kids ｜ Sheriff Labrador [xjDEb6V_iFs].mp4"
3. "I Let a Robot Dog Take Over My House [UiaXh5Q5WQg].mp4"

Place these videos in the same directory as your script and run it to process them.

## What's Happening Behind the Scenes?

### Whisper Model

The Whisper model:
- Was developed by OpenAI
- Is trained on 680,000 hours of multilingual data
- Performs speech recognition, language identification, and translation
- Has different sizes (tiny, base, small, medium, large)
- In our script, we use "medium" for a balance of accuracy and speed

### Faster-Whisper

Faster-Whisper:
- Is an optimized implementation of Whisper
- Uses CTranslate2 for faster inference
- Reduces memory usage and increases speed
- Maintains the same accuracy as the original model

### Helsinki-NLP Translation Model

The "Helsinki-NLP/opus-mt-en-zh" model:
- Was developed at the University of Helsinki
- Is trained on millions of sentence pairs
- Specializes in English to Chinese translation
- Uses the transformer architecture for high-quality translations

## Troubleshooting

### Common Issues and Solutions

1. **FFmpeg not found**
   - Ensure FFmpeg is correctly installed and in your PATH
   - Try reinstalling following the instructions above

2. **Out of memory errors**
   - Try using a smaller Whisper model: change `model_size = "medium"` to `model_size = "small"` or `model_size = "base"`
   - Close other applications to free up memory

3. **Slow processing**
   - Whisper is resource-intensive; processing can take time
   - If available, ensure you're using a GPU for faster processing

4. **Model download issues**
   - Ensure you have a stable internet connection
   - The first run will download models (which can be large)
   - Subsequent runs will use cached models

5. **Character encoding issues in SRT files**
   - Ensure you're using `encoding="utf-8"` when writing files
   - This is especially important for Chinese characters

## Conclusion

Congratulations! You've built a powerful AI tool that can transcribe videos and create bilingual subtitles automatically. This combines multiple cutting-edge AI technologies:

1. Speech recognition with Whisper
2. Machine translation with the Helsinki-NLP model
3. Python automation to tie it all together
