"""
Keep this at the top of the file:

1. I asked deepseek: I don't want to change the system prompt. In fact, in the passed entries, why not go for new 1,2,3,4...N contineous indexes first, before passing that into the LLM? I mean, after preprocessing, we will be like having a brand new srt file with everything correct, with 1, 2, 3, ... , N contineous (let's call such state STATE_NEW), and then go with the system prompt. in this way, things that happen with LLM will not need to trace back to the original starting point, just to STATE_NEW is enough, what do you think of this idea? and here is what deepseek tells me: Your approach is excellent and addresses the core issue directly. By resetting indices to a continuous 1..N sequence after preprocessing (STATE_NEW), we simplify the entire translation pipeline.

2. The purpose of STATE_NEW is to simplify the translation pipeline by ensuring that after preprocessing, the subtitles are in a clean state with continuous indices. This makes it easier to work with the subtitles in subsequent steps (like translation) because you don't need to track back to the original indices - you can just work with the new continuous sequence.

3. The json extraction function here, extract_first_json_object, is long, but good, you should not try to simplify it.

4. Don't modify the system prompt. Ever.

5. Indeed, it's better if the code is in OOP. But, I don't want to change the code too much. So, I will not do that. For this moment, AI is not so good at refactoring code. Especially across Python files, it's a nightmare. Let's wait for several years ...

"""

import os
import re
import json
import requests
import glob
import time
import datetime
import argparse
import random
from tqdm import tqdm
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

INPUT_FOLDER = "./backup-srt-original-en-files-dont-remove-this-folder"
OUTPUT_FOLDER = "./output_llm"
OUTPUT_FOLDER_CN = "./output_llm_cn"  # Chinese-only output
OUTPUT_FOLDER_EN = "./output_llm_en"  # English-only output
OUTPUT_FOLDER_WITH_FONT_SIZE = "./output_llm_with_font_size"  # Bilingual with font size
OUTPUT_FOLDER_CN_WITH_FONT_SIZE = (
    "./output_llm_cn_with_font_size"  # Chinese-only with font size
)
OUTPUT_FOLDER_EN_WITH_FONT_SIZE = (
    "./output_llm_en_with_font_size"  # English-only with font size
)
LLM_RESPONSES_FOLDER = "./responses_llm"
IP_OLLAMA_WIN = "127.0.0.1"
IP_OLLAMA_LINUX = "127.0.0.1"
IP_OLLAMA = IP_OLLAMA_WIN
PORT_OLLAMA = "11434"
LLM_MODEL = "gemma3:12b-it-qat"
AI_TRANSLATED_MARKER = "本字幕由AI大模型生成"

# Minimum subtitle duration in milliseconds
# Subtitles shorter than this are difficult to read
MIN_SUBTITLE_DURATION = 500  # 500ms = 0.5 seconds

OLLAMA_API_URL = f"http://{IP_OLLAMA}:{PORT_OLLAMA}/api/generate"
API_TIMEOUT = 18000
INITIAL_RETRY_DELAY = 3

# After a lot of experiments, 8 is the best batch size for gemma3:12b-it-qat
BATCH_SIZE = 8

SESSION = requests.Session()


SYSTEM_PROMPT = """
You are a highly intelligent AI system designed for professional subtitle processing. Your primary function is to receive a batch of English subtitle entries, which may be flawed, and perform two critical tasks: first, improve the English text, and second, provide a high-quality Simplified Chinese translation.

**Your Role:**
You are acting as an expert subtitle editor. The raw text you receive comes from an Automated Speech Recognition (ASR) system and often contains errors such as incorrect line breaks, missing punctuation, and non-content artifacts. Your job is to create clean, readable, and perfectly synchronized bilingual subtitles.

**CRITICAL INSTRUCTIONS:**

1.  **Analyze the Entire Context:**
    * You will receive multiple subtitle entries, each with a unique index. Consecutives entries have indexes that are consecutive.
    * Look for overlapping or repeated content between consecutive entries, which indicates potential merging opportunities. No need to do this for non-consecutive entries.

2.  **English Text Improvement:**
    * **Sentence Consolidation:** Merge fragmented lines into coherent, grammatically correct sentences.
    * **Punctuation and Capitalization:** Add appropriate punctuation and fix capitalization.
    * **Preserve Meaning:** Do not add new information or alter the original meaning. Use only words in the original English text.

3.  **Handling Overlapping/Repeated Entries:**
    * **Identify Overlaps:** When consecutive entries contain the same or similar text, merge them intelligently. No need to do this for non-consecutive entries.
    * **Length Considerations:** While merging is encouraged for coherence, ensure that no single entry becomes excessively long. VERY IMPORTANT: the maximum length of an entry is 20 English words, don't merge entries to form a longer sentence.

4.  **Word Movement Between Entries:**
    * Sometimes moving a few words between consecutive entries can significantly improve readability without merging the entries.
    * When an entry ends with an incomplete thought or sentence fragment, and the next entry starts with words that complete that thought, move those words to the previous entry (or the next entry, but this is less frequent).
    * Only move words that are necessary to complete the grammatical structure of the previous entry (or the next entry, but this is less frequent).
    * Do NOT merge entire entries when word movement solves the problem. Maintain separate entries with adjusted content.
    * No need to do this for non-consecutive entries.

5.  **Chinese Translation:**
    * Translate the *improved* English text, not the raw original.
    * The Chinese translation must be natural, accurate, and concise, and in context of the neighboring entries.
    * Each English entry must have a corresponding Chinese translation.

6.  **Output Format (MANDATORY) (OUTPUT_FORMAT_SPECIFICATION_1):**
    * **RESPOND WITH ONLY A JSON ARRAY** - no explanations, no markdown formatting, no commentary.
    * Your ENTIRE response must be a valid JSON array of objects.
    * Each object in the array must have exactly these three keys:
        * `"entry_indices"`: An array of integers showing which entries were processed. Critially important: the indices in "entry_indices" must come from the input text entries. For "Entry #N:", the index is N. For "Entry #M:", the index is M. In the `"entry_indices"`, you should NEVER include an index X that is not in the input text entries. THIS IS THE MOST IMPORTANT RULE.
        * `"improved_english"`: The cleaned English text as a string
        * `"chinese_translation"`: The Chinese translation as a string (Mandatory)
    * DO NOT include any text outside the JSON array.
    * DO NOT use markdown code blocks.
    * DO NOT include explanations about your thought process.
    * DO NOT use any quotes (single (') or double (")) in the improved_english content or chinese_translation content.

7. **MANDATORY: Process ALL Entries (OUTPUT_FORMAT_SPECIFICATION_2):**
    * You MUST process EVERY single entry from the input.
    * Before submitting your response, verify that ALL entry indices from the input are included in your JSON output. All input indices appear exactly once in the JSON output.
    * If entries #N through #M are in the input, ensure your response covers ALL indices from N to M.
    * Do not skip any entries, especially those at the end of the input batch.

8. **Example output format (OUTPUT_FORMAT_SPECIFICATION_3):**
[
  {
    "entry_indices": [101, 102],
    "improved_english": "This is so cool!",
    "chinese_translation": "这真是太酷了！"
  }
  {
    "entry_indices": [201, 202],
    "improved_english": "Shanghai is such a beautiful city.",
    "chinese_translation": "上海真是一个美丽的城市。"
  }
]

**IMPORTANT: YOUR RESPONSE MUST START WITH '[' AND END WITH ']' WITH NO OTHER TEXT BEFORE OR AFTER.**

## Example 1:

For the subtitle entries:
```
Entry #26:
You can't do too much. however, if you can do something, you can

Entry #27:
You cannot do too much; however, if you can do something, you can do it multiple times a day.
```

Your response should be EXACTLY:
[
{
  "entry_indices": [26],
  "improved_english": "You cannot do too much; however, if you can do something,",
  "chinese_translation": "你不能做太多的事情，但是如果你可以做一些事情，"
},
{
  "entry_indices": [27],
  "improved_english": "you can do it multiple times a day.",
  "chinese_translation": "那么你每天都可以做它多次。"
}
]


## Example 2:

For the subtitle entries:
```
Entry #1:
what's up guys welcome 

Entry #2:
what's up guys welcome to zombie 

Entry #3:
airsoft battle team DP starting  

Entry #4:
airsoft battle team DP starting here the

Entry #5:
starting here the
production teams put a  

Entry #6:
put a zillion zombies

Entry #7:
a zillion zombies
somewhere in here goal is to get as many

Entry #8:
somewhere in here goal is to get as many

Entry #9:
somewhere in here goal is to get as many
of us to the LZ as possible we tell you

Entry #10:
of us to the LZ as possible we tell you

Entry #11:
of us to the LZ as possible we tell you
more but we don't really know anymore so

Entry #12:
more but we don't really know anymore so

Entry #13:
more but we don't really know anymore so
let's find some weapons 

Entry #14:
let's find some weapons
```

Your response should be EXACTLY:
[
{
    "entry_indices": [1, 2, 3],
    "improved_english": "What is up, guys? Welcome to Zombie Airsoft.",
    "chinese_translation": "大家好，欢迎来到僵尸真人射击。"
},
{
    "entry_indices": [4, 5, 6, 7],
    "improved_english": "Battle team DP starting here. The production team put a zillion zombies somewhere in here.",
    "chinese_translation": "DP战斗小队从这里开始。制作团队在这里放了无数僵尸。"
},
{
    "entry_indices": [8, 9],
    "improved_english": "The goal is to get as many of us to the LZ as possible.",
    "chinese_translation": "目标是让尽可能多的人到达着陆区。"
},
{
    "entry_indices": [10, 11, 12],
    "improved_english": "We'd tell you more, but we do not really know anymore.",
    "chinese_translation": "我们本想告诉你更多，但我们也不太清楚了。"
},
{
    "entry_indices": [13, 14],
    "improved_english": "Let's find some weapons.",
    "chinese_translation": "我们去找些武器吧。"
}
]

## Example 3:

For the subtitle entries:
```
Entry #3:
They have done it. This is just so

Entry #4:
cool. They are now officially the winner.
```

Your response should be EXACTLY:
[
{
    "entry_indices": [3],
    "improved_english": "They have done it. This is just so cool.",
    "chinese_translation": "他们做到了。这真是太酷了。"
},
{
    "entry_indices": [4],
    "improved_english": "They are now officially the winner.",
    "chinese_translation": "他们现在是正式的赢家了。"
}
]

Process each batch of subtitle entries and provide ONLY the JSON array (specified in OUTPUT_FORMAT_SPECIFICATION_1, OUTPUT_FORMAT_SPECIFICATION_2 and OUTPUT_FORMAT_SPECIFICATION_3) as your complete response.

**FINAL VERIFICATION: Before submitting your response, verify that EVERY entry index from the input has been included in your output. Count the entry numbers in the input and ensure ALL are accounted for in your JSON output.**
"""


class SubtitleEntry:
    """A class to represent a single entry in an SRT file."""

    def __init__(self, index, start_time, end_time, text):
        self.index = int(index)
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

    def __repr__(self):
        text_for_repr = self.text.replace("\n", " ")
        return f"SubtitleEntry(index={self.index}, start='{self.start_time}', end='{self.end_time}', text='{text_for_repr}')"


def extract_first_json_object(text):
    """
    Finds and extracts the first valid JSON object from a string.
    This is necessary because the LLM might return extra text or multiple objects.

    Highly robust implementation that handles numerous edge cases and LLM output irregularities,
    including mixed quote styles, unquoted keys, incorrect escaping, and malformed structures.
    """
    # Join all lines into a single string to make regex matching more reliable
    text = " ".join(text.split())

    # Initial cleanup: Strip markdown, code blocks, and extraneous text
    text = text.strip()
    text = re.sub(r"^```(?:json|javascript|js)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^.*?(\[|\{)", r"\1", text
    )  # Remove any text before the first '[' or '{'
    text = re.sub(
        r"(\]|\})[^[\]{}]*$", r"\1", text
    )  # Remove any text after the last ']' or '}'

    # Try to identify if we have an array or object as the main structure
    array_start = text.find("[")
    object_start = text.find("{")

    # Determine the primary structure type (array or object)
    is_array = array_start != -1 and (object_start == -1 or array_start < object_start)

    # Preprocessing: Try to fix common issues with quotes and JSON syntax

    # Step 1: Handle escape sequences
    # Fix incorrectly escaped quotes
    text = re.sub(r"(?<!\\)\\\'", "'", text)  # Replace \' with '
    text = re.sub(r'(?<!\\)\\"', '"', text)  # Replace \" with "
    text = re.sub(r'\\([^"\\/bfnrtu])', r"\1", text)  # Remove unnecessary escapes

    # Step 2: Normalize quotes - convert single quotes to double quotes with careful handling
    if "'" in text:
        # Save already properly double-quoted strings to prevent interference
        placeholders = {}
        counter = 0

        def save_double_quoted(match):
            nonlocal counter
            placeholder = f"__DOUBLE_QUOTED_{counter}__"
            placeholders[placeholder] = match.group(0)
            counter += 1
            return placeholder

        # Save double-quoted strings
        text = re.sub(r'"(?:[^"\\]|\\.)*"', save_double_quoted, text)

        # Handle single-quoted keys and values in different contexts
        text = re.sub(r"\'([^\']*?)\'(\s*:)", r'"\1"\2', text)  # Keys
        text = re.sub(r":\s*\'([^\']*?)\'", r': "\1"', text)  # Values after colons
        text = re.sub(r",\s*\'([^\']*?)\'", r', "\1"', text)  # Values after commas
        text = re.sub(r"\[\s*\'([^\']*?)\'", r'["\1"', text)  # First array elements
        text = re.sub(r"\'([^\']*?)\'\s*\]", r'"\1"]', text)  # Last array elements
        text = re.sub(
            r"\'([^\']*?)\'\s*,", r'"\1",', text
        )  # Array elements followed by comma

        # Convert any remaining single-quoted strings to double-quoted
        text = re.sub(r"\'([^\']*)\'", r'"\1"', text)

        # Restore saved double-quoted strings
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)

    # Step 3: Fix unquoted keys (variable names as keys)
    text = re.sub(r"([{,])\s*([a-zA-Z_]\w*)(\s*:)", r'\1"\2"\3', text)

    # Step 4: Fix issues with boolean and null values
    text = re.sub(r':\s*"(true|false|null)"', r": \1", text, flags=re.IGNORECASE)
    text = re.sub(r':\s*"(True|False|None)"', lambda m: f": {m.group(1).lower()}", text)

    # Step 5: Fix trailing commas in objects and arrays
    text = re.sub(r",\s*([\]}])", r"\1", text)

    # Step 6: Fix extra commas between elements
    text = re.sub(r",\s*,", ",", text)

    # Step 7: Fix missing commas between objects in arrays
    text = re.sub(r"}(\s*){", r"},\1{", text)

    # The actual extraction process starts here
    if is_array:
        # Try to extract the JSON array
        try:
            # Find the matching brackets
            level = 0
            start_pos = array_start
            end_pos = -1

            for i in range(start_pos, len(text)):
                if text[i] == "[":
                    level += 1
                elif text[i] == "]":
                    level -= 1
                    if level == 0:
                        end_pos = i + 1
                        break

            if end_pos == -1:
                # Couldn't find matching closing bracket, try regex as fallback
                array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
                if array_match:
                    potential_json = array_match.group(0)
                else:
                    # Even more aggressive pattern
                    array_match = re.search(r"\[.*\]", text, re.DOTALL)
                    potential_json = array_match.group(0) if array_match else None
            else:
                potential_json = text[start_pos:end_pos]

            if potential_json:
                # Apply advanced fixes specific to arrays
                # Fix issues with empty array entries (e.g. [,,,] or [1,,2])
                potential_json = re.sub(
                    r"(?<=[\[,])\s*(?=,|\])", r"null", potential_json
                )
                potential_json = re.sub(r"(?<=,)\s*(?=,|\])", r"null", potential_json)

                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError as e:
                    # Try to fix the specific issue mentioned in the error
                    error_msg = str(e)
                    position = e.pos if hasattr(e, "pos") else -1

                    # Fix common specific errors based on error message
                    if "Expecting property name" in error_msg and position >= 0:
                        # Fix missing quotes around property names
                        prefix = potential_json[:position]
                        suffix = potential_json[position:]
                        if re.match(r"^\s*\w+\s*:", suffix):
                            # Add quotes around the property name
                            suffix = re.sub(r"^(\s*)(\w+)(\s*:)", r'\1"\2"\3', suffix)
                            potential_json = prefix + suffix
                    elif "Expecting ',' delimiter" in error_msg:
                        # Fix missing commas between array elements or properties
                        potential_json = re.sub(
                            r'([}\]])(\s*)([{["a-zA-Z0-9])', r"\1,\2\3", potential_json
                        )

                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        # More aggressive fallback approaches
                        try:
                            # Try to extract a valid subset of the JSON
                            # Find all objects in the array
                            objects = re.findall(
                                r"{[^{}]*(?:{[^{}]*}[^{}]*)*}", potential_json
                            )
                            if objects:
                                # Reconstruct a valid array with these objects
                                reconstructed = "[" + ", ".join(objects) + "]"
                                json.loads(reconstructed)
                                return reconstructed
                        except:
                            pass
        except Exception:
            pass

    # Try object extraction if array extraction failed or we have an object
    if not is_array or object_start != -1:
        try:
            # Find the matching braces for the first object
            level = 0
            start_pos = object_start
            end_pos = -1

            for i in range(start_pos, len(text)):
                if text[i] == "{":
                    level += 1
                elif text[i] == "}":
                    level -= 1
                    if level == 0:
                        end_pos = i + 1
                        break

            if end_pos == -1:
                # Couldn't find matching closing brace, try regex as fallback
                object_match = re.search(r"{.*?}", text, re.DOTALL)
                potential_json = object_match.group(0) if object_match else None
            else:
                potential_json = text[start_pos:end_pos]

            if potential_json:
                try:
                    json_obj = json.loads(potential_json)
                    # Convert single object to array for consistent processing
                    return json.dumps([json_obj])
                except json.JSONDecodeError as e:
                    # Apply targeted fixes based on the error
                    error_msg = str(e)
                    position = e.pos if hasattr(e, "pos") else -1

                    # Fix based on specific error message
                    if position >= 0:
                        if "Expecting property name" in error_msg:
                            # Fix missing quotes around property names
                            prefix = potential_json[:position]
                            suffix = potential_json[position:]
                            if re.match(r"^\s*\w+\s*:", suffix):
                                suffix = re.sub(
                                    r"^(\s*)(\w+)(\s*:)", r'\1"\2"\3', suffix
                                )
                                potential_json = prefix + suffix
                        elif "Expecting ':'" in error_msg:
                            # Fix missing colons between keys and values
                            prefix = potential_json[:position]
                            suffix = potential_json[position:]
                            # Look for a pattern like "key" without a following colon
                            if re.search(r'"[^"]+"\s*(?![,:}])', prefix):
                                last_quote = prefix.rindex('"')
                                new_prefix = (
                                    prefix[: last_quote + 1]
                                    + ":"
                                    + prefix[last_quote + 1 :]
                                )
                                potential_json = new_prefix + suffix

                    try:
                        json_obj = json.loads(potential_json)
                        return json.dumps([json_obj])
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

    # Super aggressive last resort - try to construct a valid JSON array from whatever object-like content we can find
    try:
        # Find all JSON-like objects in the text
        objects = re.findall(r"{(?:[^{}]|(?R))*}", text, re.DOTALL)
        if not objects:
            # Use a simpler pattern as fallback
            objects = re.findall(r"{[^{}]*(?:{[^{}]*}[^{}]*)*}", text)

        if objects:
            # Try to fix each object and build a valid array
            valid_objects = []
            for obj in objects:
                try:
                    # Apply the same fixes we used earlier
                    fixed_obj = re.sub(
                        r"([{,])\s*([a-zA-Z_]\w*)(\s*:)", r'\1"\2"\3', obj
                    )
                    fixed_obj = re.sub(
                        r':\s*"(true|false|null)"',
                        r": \1",
                        fixed_obj,
                        flags=re.IGNORECASE,
                    )
                    fixed_obj = re.sub(r",\s*}", r"}", fixed_obj)
                    json.loads(fixed_obj)  # Test if valid
                    valid_objects.append(fixed_obj)
                except:
                    pass

            if valid_objects:
                reconstructed = "[" + ", ".join(valid_objects) + "]"
                try:
                    json.loads(reconstructed)
                    return reconstructed
                except:
                    pass
    except:
        pass

    # If no other approach worked, try to find any JSON-like structure
    fallback_patterns = [
        r"\[\s*\{.*?\}\s*\]",
        r"\[\s*\{[^\[\]]*?\}\s*\]",
        r"\[.*?\]",
        r"{.*?}",
    ]

    for pattern in fallback_patterns:
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                potential_json = match.group(0)
                # Apply all our fixes one last time
                potential_json = re.sub(
                    r"([{,])\s*([a-zA-Z_]\w*)(\s*:)", r'\1"\2"\3', potential_json
                )
                potential_json = re.sub(
                    r':\s*"(true|false|null)"',
                    r": \1",
                    potential_json,
                    flags=re.IGNORECASE,
                )
                potential_json = re.sub(r",\s*([\]}])", r"\1", potential_json)

                try:
                    json.loads(potential_json)
                    if not potential_json.startswith("["):
                        # Convert single object to array
                        obj = json.loads(potential_json)
                        return json.dumps([obj])
                    return potential_json
                except:
                    # Return our best guess even if it's not valid JSON
                    if pattern == fallback_patterns[-1]:  # Only on the last pattern
                        return potential_json
        except:
            pass

    # If we get here, we couldn't find a valid JSON structure
    return None


def parse_srt(file_content):
    """Parses the content of an SRT file into a list of SubtitleEntry objects."""
    content = file_content.replace("\r\n", "\n").strip()
    entries = []

    for block in content.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            try:
                # Original index from file
                orig_index = int(lines[0])
                time_line = lines[1]
                text = "\n".join(lines[2:])

                # Skip entries with empty text after cleaning
                if not clean_text(text).strip():
                    continue

                start_time, end_time = time_line.split(" --> ")

                # Parse timestamps for validation
                start_ms = timestamp_to_milliseconds(start_time.strip())
                end_ms = timestamp_to_milliseconds(end_time.strip())

                # Basic timestamp validation
                if start_ms >= end_ms:
                    tqdm_warning(
                        f"Warning: Entry {orig_index} has invalid timing (start >= end), fixing"
                    )
                    # Ensure minimum duration
                    end_ms = start_ms + MIN_SUBTITLE_DURATION
                    end_time = format_timestamp(end_ms)

                # Add entry with the original index from the file
                entries.append(
                    SubtitleEntry(
                        orig_index, start_time.strip(), end_time.strip(), text.strip()
                    )
                )

            except (ValueError, IndexError) as e:
                tqdm_error(f"Skipping malformed SRT block: '{block}' - Error: {e}")

    # Ensure entries are sorted by start time for consistency
    entries.sort(key=lambda e: timestamp_to_milliseconds(e.start_time))

    # Apply STATE_NEW transformation
    return state_new_entries(entries)


def format_srt_entry(entry):
    """Formats a SubtitleEntry object back into a string in SRT format."""
    return f"{entry.index}\n{entry.start_time} --> {entry.end_time}\n{entry.text}\n\n"


def reindex_entries(srt_content):
    """
    Professional-grade SRT finalization with comprehensive timestamp handling.
    Reindexes SRT entries to ensure they have continuous index numbers with proper timing.

    Args:
        srt_content: List of SRT entry strings in the format "index\nstart --> end\ntext\n\n"

    Returns:
        List of reindexed SRT entry strings
    """
    if not srt_content:
        return []

    # Parse each entry
    entries = []
    skipped_count = 0
    prev_end_ms = 0  # Track previous end time for gap handling

    for entry_text in srt_content:
        if not entry_text.strip():
            continue

        lines = entry_text.strip().split("\n")
        if len(lines) < 3:
            skipped_count += 1
            continue

        try:
            # Try to parse the index, but we'll replace it anyway
            # This is just to check if the format is valid
            try:
                int(lines[0])
            except ValueError:
                # Silently handle non-integer indices
                pass

            time_line = lines[1]
            text = "\n".join(lines[2:])

            # Skip entries with empty text
            if not text.strip():
                skipped_count += 1
                continue

            # Parse timestamps and validate
            try:
                start_time, end_time = time_line.split(" --> ")
                start_time = start_time.strip()
                end_time = end_time.strip()

                # Convert and validate timestamps
                start_ms = timestamp_to_milliseconds(start_time)
                end_ms = timestamp_to_milliseconds(end_time)

                # Add minimum 50ms gap between subtitles
                if start_ms < prev_end_ms + 50:
                    start_ms = prev_end_ms + 50

                # Ensure minimum duration
                if end_ms - start_ms < MIN_SUBTITLE_DURATION:
                    end_ms = start_ms + MIN_SUBTITLE_DURATION

                # Handle invalid durations
                if end_ms <= start_ms:
                    end_ms = start_ms + MIN_SUBTITLE_DURATION

                # Update for next iteration
                prev_end_ms = end_ms

                entries.append(
                    SubtitleEntry(
                        0,  # Temp index - will be renumbered
                        format_timestamp(start_ms),
                        format_timestamp(end_ms),
                        text.strip(),
                    )
                )
            except Exception as e:
                # Silently skip malformed entries
                skipped_count += 1
                continue

        except Exception as e:
            # Silently skip malformed entries
            skipped_count += 1

    if skipped_count > 0:
        tqdm_warning(
            f"Skipped {skipped_count} malformed or empty entries during reindexing"
        )

    if not entries:
        tqdm_warning("Warning: No valid entries found after parsing SRT content")
        return []

    # Sort by start time for chronological order
    try:
        entries.sort(key=lambda e: timestamp_to_milliseconds(e.start_time))
    except Exception as e:
        tqdm_warning(
            f"Warning: Error sorting entries by timestamp: {str(e)}. Using original order."
        )

    # Apply final consecutive numbering
    for i, entry in enumerate(entries, 1):
        entry.index = i

    # Format back to strings
    formatted_entries = [format_srt_entry(entry) for entry in entries]

    return formatted_entries


def clean_text(text):
    """Cleans subtitle text by removing audio markers and other artifacts."""
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = text.replace("\\h", " ")
    text = text.replace("...", " ")
    text = re.sub(r"^\s*-\s*", " ", text, flags=re.MULTILINE)
    text = text.replace("foreign", " ")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # B. Remove sound effect descriptions
    text = re.sub(r"\[.*?sound.*?\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*?sound.*?\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?noise.*?\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*?noise.*?\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?music.*?\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*?music.*?\)", " ", text, flags=re.IGNORECASE)

    # 6. Handle specific characters that might cause JSON issues
    text = text.replace("\\", " ")  # Remove backslashes
    text = text.replace("\t", " ")  # Replace tabs with spaces

    # Clean up extra spaces created during normalization
    text = re.sub(r"\s+", " ", text).strip()

    # Split into lines, strip each line, and remove empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    return " ".join(lines)


def is_non_translatable_entry(text):
    """
    Determines if an entry contains only non-translatable content like numbers or timestamps.
    Returns True if the entry should be filtered out, False otherwise.
    """
    if not text or not text.strip():
        return True

    # Clean the text first
    cleaned_text = clean_text(text).strip()
    if not cleaned_text:
        return True

    # 1. Check if entry is only numbers, symbols, and punctuation (no letters)
    has_letters = bool(re.search(r"[a-zA-Z]", cleaned_text))
    if not has_letters:
        # Contains only numbers, symbols, and punctuation
        return True

    # 2. Check if entry is a timestamp pattern (e.g., "01:23:45")
    if re.fullmatch(r"[\d:\.]+", cleaned_text):
        return True

    # 3. Check if entry is extremely short and non-meaningful
    if len(cleaned_text) <= 1:
        return True

    # 4. Check if entry is just symbols/punctuation/digits with at most one letter
    if len(cleaned_text) <= 3 and sum(c.isalpha() for c in cleaned_text) <= 1:
        return True

    # Entry is translatable
    return False


def create_entry_batch(entries, batch_size=BATCH_SIZE):
    """Creates batches of subtitle entries for processing."""
    # Empty entries are now filtered earlier in process_file
    for i in range(0, len(entries), batch_size):
        yield entries[i : i + batch_size]


def prepare_batch_for_llm(entries):
    """Prepares a batch of subtitle entries for the LLM by formatting them with context."""
    batch_text = []
    for entry in entries:
        # Apply clean_text for consistent formatting but skip empty check
        # since empty entries are filtered earlier
        cleaned_text = clean_text(entry.text)
        # Only include entry index and text (no timestamps)
        batch_text.append(f"Entry #{entry.index}:\n{cleaned_text}")

    return "\n\n".join(batch_text)


def parse_entry_indices(indices_value):
    """
    Parse the entry_indices field from the LLM response, handling different formats.

    Args:
        indices_value: The value of the entry_indices field from the LLM response.
                       Could be a list, a single integer, or a string.

    Returns:
        A list of integers.
    """
    if indices_value is None:
        return []

    if isinstance(indices_value, list):
        # Already a list, just make sure all elements are integers
        return [int(idx) for idx in indices_value if str(idx).isdigit()]

    if isinstance(indices_value, int):
        # Single integer
        return [indices_value]

    if isinstance(indices_value, str):
        # Try to parse string formats like "[1, 2, 3]" or "1, 2, 3" or "1"
        try:
            # Try to evaluate as a list literal
            import ast

            parsed = ast.literal_eval(indices_value)
            if isinstance(parsed, list):
                return [int(idx) for idx in parsed if str(idx).isdigit()]
            elif isinstance(parsed, int):
                return [parsed]
        except (SyntaxError, ValueError):
            # Try to split by comma
            if "," in indices_value:
                parts = indices_value.replace("[", "").replace("]", "").split(",")
                return [int(part.strip()) for part in parts if part.strip().isdigit()]
            # Try as a single number
            elif indices_value.strip().isdigit():
                return [int(indices_value.strip())]

    # Fallback: empty list if we couldn't parse
    return []


def get_translation_for_batch(batch_entries):
    """
    Sends a batch of subtitle entries to the Ollama API for improvement and translation.
    Performs a single attempt and returns the result or an empty list on failure.
    All retry logic is handled by process_retry_batches.
    """
    if not batch_entries:
        return []

    # Create LLM_RESPONSES_FOLDER if it doesn't exist
    if not os.path.exists(LLM_RESPONSES_FOLDER):
        os.makedirs(LLM_RESPONSES_FOLDER)
        tqdm_info(f"Created LLM responses folder: '{LLM_RESPONSES_FOLDER}'")

    # Create direct index mapping
    index_to_entry = {entry.index: entry for entry in batch_entries}

    batch_text = prepare_batch_for_llm(batch_entries)

    payload = {
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": batch_text,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 8192,
        },
    }

    try:
        response = SESSION.post(OLLAMA_API_URL, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()

        response_data = response.json()
        response_content = response_data.get("response", "[]")

        # Save the raw response to a timestamped file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        response_file = os.path.join(LLM_RESPONSES_FOLDER, f"{timestamp}.txt")

        with open(response_file, "w", encoding="utf-8") as f:
            # Write the prompt and response for context
            f.write(f"PROMPT:\n{batch_text}\n\n")
            f.write(f"RESPONSE:\n{response_content}\n")

        # Join all lines into a single string before extracting JSON
        response_content = " ".join(response_content.split())
        json_str = extract_first_json_object(response_content)

        if not json_str:
            raise ValueError("No valid JSON array found in LLM response.")

        # Try to fix common LLM output issues before parsing
        # Remove markdown code blocks
        response_content = re.sub(
            r"```(?:json|javascript|js)?\s*", "", response_content
        )
        response_content = re.sub(r"\s*```$", "", response_content)

        # Ensure the response starts with [ and ends with ]
        if "[" not in response_content:
            response_content = "[" + response_content
        if "]" not in response_content:
            response_content = response_content + "]"

        json_str = extract_first_json_object(response_content)

        if not json_str:
            # If still no valid JSON, try a more aggressive approach
            # Find anything that looks like a JSON object and wrap it in an array
            object_match = re.search(r"{.*?}", response_content, re.DOTALL)
            if object_match:
                potential_json = "[" + object_match.group(0) + "]"
                try:
                    # Test if it's valid JSON
                    json.loads(potential_json)
                    json_str = potential_json
                except:
                    pass

        if not json_str:
            raise ValueError("No valid JSON array found in LLM response.")

        processed_entries = json.loads(json_str)
        # If we got a single object instead of an array, wrap it in an array
        if isinstance(processed_entries, dict):
            processed_entries = [processed_entries]

        # Process each entry to ensure valid entry_indices and key names
        for entry in processed_entries:
            # Handle both "entry_indices" (plural) and "entry_index" (singular) fields
            if "entry_indices" in entry:
                entry["entry_indices"] = parse_entry_indices(entry["entry_indices"])
            elif "entry_index" in entry:
                # Convert entry_index to entry_indices for consistency
                entry["entry_indices"] = parse_entry_indices(entry["entry_index"])
                # Remove the singular form to avoid confusion
                entry.pop("entry_index", None)
            else:
                # If neither is present, try to use indices from a batch entry
                if len(batch_entries) == 1:
                    entry["entry_indices"] = [batch_entries[0].index]
                else:
                    entry["entry_indices"] = []

            # Map indices directly to entries using our simplified mapping
            valid_entries = []
            valid_indices = []
            for idx in entry["entry_indices"]:
                if idx in index_to_entry:
                    valid_indices.append(idx)
                    valid_entries.append(index_to_entry[idx])
                # Silently skip invalid indices without warning

            # Store both valid indices and their corresponding entries
            entry["entry_indices"] = valid_indices
            entry["valid_entries"] = valid_entries

            # Handle alternative key names for translations
            if not "improved_english" in entry and "english" in entry:
                entry["improved_english"] = entry["english"]

            if not "chinese_translation" in entry:
                if "chinese" in entry:
                    entry["chinese_translation"] = entry["chinese"]
                elif "translation" in entry:
                    entry["chinese_translation"] = entry["translation"]

        return processed_entries

    except Exception as e:
        # Log the error but don't retry - process_retry_batches will handle retries
        error_type = type(e).__name__
        error_msg = str(e)
        tqdm_error(f"Error in API call ({error_type}): {error_msg}")

        # For connection errors, give a more specific message
        if isinstance(e, requests.exceptions.ConnectionError):
            tqdm_error(f"Connection error to Ollama server at {OLLAMA_API_URL}")

        # Save error information for debugging
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            response_file = os.path.join(LLM_RESPONSES_FOLDER, f"error_{timestamp}.txt")
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(f"PROMPT:\n{batch_text}\n\n")
                f.write(f"ERROR: {error_type}: {error_msg}\n")

                # Include response content if available
                if "response_content" in locals():
                    f.write(f"RESPONSE CONTENT:\n{response_content}\n")
        except Exception as save_error:
            tqdm_warning(f"Could not save error details: {save_error}")

        # Return empty list to indicate failure
        return []


def merge_similar_entries(subtitle_entries):
    """
    Merges subtitle entries based on text similarity rules:

    1. Consecutive entries with identical text are merged into one
    2. Consecutive entries where each entry's text is contained in the next, and the
       final entry has fewer than 20 words, are merged keeping only the final entry

    Returns the merged list of entries.
    """
    if not subtitle_entries or len(subtitle_entries) <= 1:
        return subtitle_entries

    merged_entries = []
    i = 0

    # Maximum allowed duration for a merged subtitle (in milliseconds)
    # If merging would create a subtitle longer than this, skip the merge
    MAX_MERGED_DURATION = 5000  # 5 seconds

    while i < len(subtitle_entries):
        # Start with the current entry
        current_entry = subtitle_entries[i]
        current_text = clean_text(current_entry.text).strip()
        start_index = i
        end_index = i

        # Skip empty entries - we still need this check for the merging algorithm
        # even though we'll filter empty entries after normalization
        if not current_text:
            merged_entries.append(current_entry)
            i += 1
            continue

        # Count words in current entry
        current_word_count = len(current_text.split())

        # Check for consecutive identical entries (Task 1)
        j = i + 1
        identical_streak = True
        while j < len(subtitle_entries):
            next_text = clean_text(subtitle_entries[j].text).strip()
            if not next_text:
                j += 1
                continue

            if next_text == current_text:
                end_index = j
                j += 1
            else:
                identical_streak = False
                break

        # If we found identical entries, merge them (with duration check)
        if identical_streak and end_index > start_index:
            start_time = subtitle_entries[start_index].start_time
            end_time = subtitle_entries[end_index].end_time

            # Check if the merged duration would be reasonable
            start_ms = timestamp_to_milliseconds(start_time)
            end_ms = timestamp_to_milliseconds(end_time)

            if end_ms - start_ms <= MAX_MERGED_DURATION:
                # Keep the original index from the first entry - will be reset in STATE_NEW
                merged_entry = SubtitleEntry(
                    subtitle_entries[start_index].index,
                    start_time,
                    end_time,
                    subtitle_entries[start_index].text,
                )
                merged_entries.append(merged_entry)
                i = j if j < len(subtitle_entries) else len(subtitle_entries)
                continue
            else:
                # If merging would create a too-long subtitle, just add the current entry
                # tqdm.write(f"Skipping merge of identical entries {start_index}-{end_index} due to excessive duration")
                merged_entries.append(current_entry)
                i += 1
                continue

        # Check for contained text sequence (Task 2)
        j = i + 1
        contained_entries = [i]
        current_text_lower = current_text.lower()

        while j < len(subtitle_entries):
            next_text = clean_text(subtitle_entries[j].text).strip()
            if not next_text:
                j += 1
                continue

            next_text_lower = next_text.lower()

            # Check if current text is contained in next text
            if current_text_lower in next_text_lower:
                # Check if entries are contiguous in time
                current_end_ms = timestamp_to_milliseconds(
                    subtitle_entries[contained_entries[-1]].end_time
                )
                next_start_ms = timestamp_to_milliseconds(
                    subtitle_entries[j].start_time
                )

                # Only include if they're reasonably close in time (< 2 seconds gap)
                if next_start_ms - current_end_ms < 2000:
                    contained_entries.append(j)
                    current_text = next_text
                    current_text_lower = next_text_lower
                    current_word_count = len(current_text.split())
                    j += 1
                else:
                    # If there's a big time gap, don't merge even if text is contained
                    break
            else:
                break

        # If we found a sequence of contained entries and final entry has < 20 words
        if len(contained_entries) > 1 and current_word_count < 20:
            last_idx = contained_entries[-1]
            start_time = subtitle_entries[start_index].start_time
            end_time = subtitle_entries[last_idx].end_time

            # Check if the merged duration would be reasonable
            start_ms = timestamp_to_milliseconds(start_time)
            end_ms = timestamp_to_milliseconds(end_time)

            if end_ms - start_ms <= MAX_MERGED_DURATION:
                # Keep the original index from the first entry - will be reset in STATE_NEW
                merged_entry = SubtitleEntry(
                    subtitle_entries[start_index].index,
                    start_time,
                    end_time,
                    subtitle_entries[last_idx].text,
                )
                merged_entries.append(merged_entry)
                i = j if j < len(subtitle_entries) else len(subtitle_entries)
            else:
                # If merging would create a too-long subtitle, just add the current entry
                # tqdm.write(f"Skipping merge of contained entries {start_index}-{last_idx} due to excessive duration")
                merged_entries.append(current_entry)
                i += 1
        else:
            # No merging occurred, add the current entry as-is
            merged_entries.append(current_entry)
            i += 1

    # Return the merged entries
    return merged_entries


def format_with_font(text, size):
    """Apply font size formatting to text."""
    return f'<font size="{size}">{text}</font>'


def create_subtitle_entries(index, start_time, end_time, english_text, chinese_text):
    """Create all required subtitle entry types from the given texts.

    Args:
        index: The subtitle index
        start_time: Start timestamp
        end_time: End timestamp
        english_text: The English text (single line)
        chinese_text: The Chinese text (single line)

    Returns:
        A tuple containing all entry types:
        (bilingual_entry, bilingual_entry_with_font, eng_entry, chi_entry, eng_entry_with_font, chi_entry_with_font)
    """
    # Create bilingual text
    bilingual_text = f"{english_text}\n{chinese_text}"

    # Create font-sized bilingual text
    eng_with_font = format_with_font(english_text, 10)
    chi_with_font = format_with_font(chinese_text, 14)
    bilingual_text_with_font = f"{eng_with_font}\n{chi_with_font}"

    # Create all entry types
    bilingual_entry = SubtitleEntry(index, start_time, end_time, bilingual_text)
    bilingual_entry_with_font = SubtitleEntry(
        index, start_time, end_time, bilingual_text_with_font
    )
    eng_entry = SubtitleEntry(index, start_time, end_time, english_text)
    chi_entry = SubtitleEntry(index, start_time, end_time, chinese_text)
    eng_entry_with_font = SubtitleEntry(index, start_time, end_time, eng_with_font)
    chi_entry_with_font = SubtitleEntry(index, start_time, end_time, chi_with_font)

    return (
        bilingual_entry,
        bilingual_entry_with_font,
        eng_entry,
        chi_entry,
        eng_entry_with_font,
        chi_entry_with_font,
    )


def add_entries_to_content_lists(entries, content_lists):
    """Add formatted entries to their respective content lists.

    Args:
        entries: Tuple of entries from create_subtitle_entries()
        content_lists: Tuple of content lists (translated, translated_with_font, english, chinese, english_with_font, chinese_with_font)
    """
    (
        bilingual_entry,
        bilingual_entry_with_font,
        eng_entry,
        chi_entry,
        eng_entry_with_font,
        chi_entry_with_font,
    ) = entries
    (
        translated_srt_content,
        translated_srt_content_with_font,
        english_srt_content,
        chinese_srt_content,
        english_srt_content_with_font,
        chinese_srt_content_with_font,
    ) = content_lists

    translated_srt_content.append(format_srt_entry(bilingual_entry))
    translated_srt_content_with_font.append(format_srt_entry(bilingual_entry_with_font))
    english_srt_content.append(format_srt_entry(eng_entry))
    english_srt_content_with_font.append(format_srt_entry(eng_entry_with_font))
    chinese_srt_content.append(format_srt_entry(chi_entry))
    chinese_srt_content_with_font.append(format_srt_entry(chi_entry_with_font))


def process_entry_group(
    valid_entries, improved_english, chinese_translation, content_lists
):
    """Process a group of entries, creating properly formatted subtitle entries.

    Args:
        valid_entries: List of valid subtitle entries
        improved_english: Improved English text
        chinese_translation: Chinese translation text
        content_lists: Tuple of content lists to add entries to

    Returns:
        True if processing was successful, False otherwise
    """
    if not improved_english or not chinese_translation:
        return False

    # Clean and process text
    improved_english = clean_text(improved_english)
    chinese_translation = clean_text(chinese_translation)

    if not improved_english.strip() or not chinese_translation.strip():
        return False

    # Convert to single lines
    improved_english_single_line = " ".join(improved_english.split("\n"))
    chinese_translation_single_line = " ".join(chinese_translation.split("\n"))

    # Get timing from first entry and merge if needed
    first_entry = valid_entries[0]
    start_time = first_entry.start_time
    end_time = first_entry.end_time

    # If multiple entries, merge time spans
    if len(valid_entries) > 1:
        start_ms = timestamp_to_milliseconds(start_time)
        end_ms = timestamp_to_milliseconds(end_time)

        for entry in valid_entries[1:]:
            entry_start_ms = timestamp_to_milliseconds(entry.start_time)
            entry_end_ms = timestamp_to_milliseconds(entry.end_time)

            start_ms = min(start_ms, entry_start_ms)
            end_ms = max(end_ms, entry_end_ms)

        start_time = format_timestamp(start_ms)
        end_time = format_timestamp(end_ms)

    # Validate duration
    start_ms = timestamp_to_milliseconds(start_time)
    end_ms = timestamp_to_milliseconds(end_time)

    if start_ms >= end_ms:
        end_ms = start_ms + MIN_SUBTITLE_DURATION
        end_time = format_timestamp(end_ms)

    # Create entries and add to content lists
    entries = create_subtitle_entries(
        first_entry.index,
        start_time,
        end_time,
        improved_english_single_line,
        chinese_translation_single_line,
    )

    add_entries_to_content_lists(entries, content_lists)
    return True


def process_batch_entries(
    processed_entries,
    filtered_batch,
    processed_indices,
    unprocessed_indices,
    content_lists,
):
    """Process entries from a batch response.

    Args:
        processed_entries: List of processed entries from LLM
        filtered_batch: The original filtered batch
        processed_indices: Set of processed indices to update
        unprocessed_indices: Set of unprocessed indices to update (or None if not needed)
        content_lists: Tuple of content lists to add entries to

    Returns:
        Number of successfully processed entries
    """
    successful_count = 0

    # Create index to entry mapping for the batch
    index_to_entry = {entry.index: entry for entry in filtered_batch}

    # Group entries by valid entries
    entries_by_indices = {}
    for proc_entry in processed_entries:
        # Skip entries without valid entries
        valid_entries = proc_entry.get("valid_entries", [])
        if not valid_entries:
            continue

        # Use the sorted tuple of entry indices as the key
        key = tuple(sorted(e.index for e in valid_entries))

        if key not in entries_by_indices:
            entries_by_indices[key] = []

        entries_by_indices[key].append(proc_entry)

    # Process each group of entries
    for indices, entries in entries_by_indices.items():
        # Process each entry in the group
        for proc_entry in entries:
            valid_entries = proc_entry.get("valid_entries", [])
            if not valid_entries:
                continue

            # Mark these indices as processed
            for entry in valid_entries:
                processed_indices.add(entry.index)
                if (
                    unprocessed_indices is not None
                    and entry.index in unprocessed_indices
                ):
                    unprocessed_indices.remove(entry.index)

            improved_english = proc_entry.get("improved_english", "")
            chinese_translation = proc_entry.get("chinese_translation", "")

            # Process the entry group
            success = process_entry_group(
                valid_entries, improved_english, chinese_translation, content_lists
            )

            if not success:
                # Mark as not processed so we can retry them later
                for entry in valid_entries:
                    if entry.index in processed_indices:
                        processed_indices.remove(entry.index)
                    if (
                        unprocessed_indices is not None
                        and entry.index not in unprocessed_indices
                    ):
                        unprocessed_indices.add(entry.index)
            else:
                successful_count += 1

    return successful_count


def create_untranslated_entry(entry, untranslated_count, content_lists):
    """Create untranslated entries for entries that failed translation.

    Args:
        entry: The original subtitle entry
        untranslated_count: Counter for untranslated entries
        content_lists: Tuple of content lists to add entries to

    Returns:
        The updated untranslated_count
    """
    # Use a temporary high index to avoid conflicts
    temp_index = 2000000 + untranslated_count

    # Validate start and end times
    start_time = entry.start_time
    end_time = entry.end_time

    # Validate that start time is before end time
    start_ms = timestamp_to_milliseconds(start_time)
    end_ms = timestamp_to_milliseconds(end_time)

    if start_ms >= end_ms:
        # Ensure minimum duration using our constant
        end_ms = start_ms + MIN_SUBTITLE_DURATION
        end_time = format_timestamp(end_ms)

    # Clean and convert to single line
    original_text = clean_text(entry.text)
    original_text_single_line = " ".join(original_text.split("\n"))

    # For untranslated entries, use original text for English and empty for Chinese
    entries = create_subtitle_entries(
        temp_index,
        start_time,
        end_time,
        original_text_single_line,
        "",  # Empty Chinese text for untranslated entries
    )

    add_entries_to_content_lists(entries, content_lists)
    return untranslated_count + 1


def process_untranslated_entries(
    subtitle_entries, processed_indices, content_lists, file_name
):
    """Process any entries that weren't translated even after retries.

    Args:
        subtitle_entries: List of all subtitle entries
        processed_indices: Set of already processed indices
        content_lists: Tuple of content lists
        file_name: File name for logging purposes

    Returns:
        Number of untranslated entries processed
    """
    untranslated_count = 0
    # Get all unprocessed indices
    unprocessed_indices = get_unprocessed_indices(subtitle_entries, processed_indices)

    # Process each untranslated entry
    for entry in subtitle_entries:
        if entry.index in unprocessed_indices:
            # Process untranslated entry
            tqdm_warning(
                f"Using original text for entry {entry.index} in {file_name} (translation failed)"
            )
            untranslated_count = create_untranslated_entry(
                entry, untranslated_count, content_lists
            )

    if untranslated_count > 0:
        tqdm_warning(
            f"Added {untranslated_count} untranslated entries with original text"
        )

    return untranslated_count


def process_file(file_path, output_path):
    """Processes a single SRT file: reads, translates, and writes the output."""
    file_name = os.path.basename(file_path)

    # Get all output file paths
    output_paths = get_output_paths(file_name)

    # Validate the input file first
    is_valid, message = validate_srt_file(file_path)
    if not is_valid:
        tqdm_warning(f"Skipping invalid file: {message}")
        return
    else:
        tqdm_info(message)

    # Read file content
    success, content = safe_read_file(file_path, file_name)
    if not success:
        return

    # Parse and preprocess subtitle entries
    subtitle_entries = parse_srt(content)
    subtitle_entries = preprocess_subtitle_entries(subtitle_entries, file_name)
    if not subtitle_entries:
        return

    # Initialize content lists
    content_lists = initialize_content_lists()

    processed_indices = set()  # Keep track of which entries have been processed

    # Process entries in batches
    process_batches(subtitle_entries, processed_indices, content_lists, file_name)

    # Retry processing for failed entries with dynamic max_retries
    process_retry_batches(
        subtitle_entries, processed_indices, content_lists, max_retries=None
    )

    # Process any entries that weren't translated even after retries
    process_untranslated_entries(
        subtitle_entries, processed_indices, content_lists, file_name
    )

    # Reindex and fix timing issues in all outputs
    content_lists = reindex_and_fix_all_outputs(content_lists, file_name)

    # Write all output files
    if write_output_files(content_lists, output_paths):
        tqdm_success(f"Successfully wrote output files for {file_name}")
    else:
        tqdm_error(f"Error writing output files for {file_name}")


def format_timestamp(milliseconds):
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def timestamp_to_milliseconds(timestamp):
    """Convert SRT timestamp to milliseconds.

    Args:
        timestamp: Timestamp in format "HH:MM:SS,mmm"

    Returns:
        Timestamp in milliseconds
    """
    parts = timestamp.split(":")
    # Handle both formats: 00:00:00,000 and 0:00:00,000
    if len(parts) == 3:
        hours, minutes, seconds_and_ms = parts
    else:
        # Handle invalid timestamp format
        tqdm_warning(f"Invalid timestamp format: {timestamp}")
        return 0

    try:
        # Split seconds and milliseconds
        seconds_parts = seconds_and_ms.split(",")
        if len(seconds_parts) == 2:
            seconds, milliseconds = seconds_parts
        else:
            tqdm_warning(f"Invalid seconds format in timestamp: {timestamp}")
            return 0

        # Convert all parts to milliseconds
        hours_ms = int(hours) * 3600 * 1000
        minutes_ms = int(minutes) * 60 * 1000
        seconds_ms = int(seconds) * 1000
        ms = int(milliseconds)

        return hours_ms + minutes_ms + seconds_ms + ms
    except (ValueError, IndexError) as e:
        tqdm_error(f"Error parsing timestamp {timestamp}: {str(e)}")
        return 0


def ensure_minimum_duration(entry, min_duration):
    """Ensure a subtitle entry has the minimum required duration.

    Args:
        entry: SubtitleEntry to check
        min_duration: Minimum duration in milliseconds

    Returns:
        True if the entry was modified, False otherwise
    """
    start_ms = timestamp_to_milliseconds(entry.start_time)
    end_ms = timestamp_to_milliseconds(entry.end_time)

    duration = end_ms - start_ms
    if duration < min_duration:
        entry.end_time = format_timestamp(start_ms + min_duration)
        return True

    return False


def fix_subtitle_overlap(current_entry, prev_entry, gap=50):
    """Fix overlap between two subtitle entries.

    Args:
        current_entry: Current subtitle entry
        prev_entry: Previous subtitle entry
        gap: Gap to add between entries in milliseconds

    Returns:
        True if the entry was modified, False otherwise
    """
    current_start_ms = timestamp_to_milliseconds(current_entry.start_time)
    current_end_ms = timestamp_to_milliseconds(current_entry.end_time)
    prev_end_ms = timestamp_to_milliseconds(prev_entry.end_time)

    # If current subtitle starts before previous ends, adjust timing
    if current_start_ms < prev_end_ms:
        # Add a gap after the previous subtitle
        new_start_ms = prev_end_ms + gap

        # Update the start time
        current_entry.start_time = format_timestamp(new_start_ms)

        # Ensure the subtitle still has minimum duration
        new_end_ms = new_start_ms + MIN_SUBTITLE_DURATION
        if new_end_ms > current_end_ms:
            current_entry.end_time = format_timestamp(new_end_ms)

        return True

    return False


def parse_subtitle_entries(srt_content):
    """Parse a list of SRT content strings into SubtitleEntry objects.

    Args:
        srt_content: List of formatted SRT entries as strings

    Returns:
        List of SubtitleEntry objects
    """
    entries = []
    for entry_text in srt_content:
        lines = entry_text.strip().split("\n")
        if len(lines) >= 2:
            try:
                index = int(lines[0])
                time_line = lines[1]
                text = "\n".join(lines[2:])
                start_time, end_time = time_line.split(" --> ")
                entries.append(
                    SubtitleEntry(
                        index, start_time.strip(), end_time.strip(), text.strip()
                    )
                )
            except (ValueError, IndexError):
                continue

    return entries


def fix_overlapping_subtitles(srt_content):
    """Detect and fix overlapping subtitle timings to prevent playback issues."""
    if not srt_content:
        return srt_content, 0, 0

    # Parse entries to check timings
    entries = parse_subtitle_entries(srt_content)

    if not entries:
        return srt_content, 0, 0

    # Sort by start time
    entries.sort(key=lambda e: timestamp_to_milliseconds(e.start_time))

    # Track modifications
    short_duration_count = 0
    fixed_count = 0

    # First pass: ensure minimum durations
    for entry in entries:
        if ensure_minimum_duration(entry, MIN_SUBTITLE_DURATION):
            short_duration_count += 1

    # Second pass: fix overlaps with multiple iterations to handle cascading effects
    # Loop until no more fixes are needed
    max_iterations = 3  # Prevent infinite loops
    for _ in range(max_iterations):
        overlap_count = 0

        for i in range(1, len(entries)):
            if fix_subtitle_overlap(entries[i], entries[i - 1]):
                overlap_count += 1

        if overlap_count == 0:
            # No more overlaps to fix
            break
        else:
            fixed_count += overlap_count

    # Only create new list if changes were made
    if short_duration_count > 0 or fixed_count > 0:
        return (
            [format_srt_entry(entry) for entry in entries],
            short_duration_count,
            fixed_count,
        )
    else:
        return srt_content, 0, 0


def reindex_and_fix_all_outputs(content_lists, file_name):
    """Reindex and fix timing issues in all output formats.

    Args:
        content_lists: Tuple containing all content lists
        file_name: The file name for logging purposes

    Returns:
        The updated content lists
    """

    # Define a processor function to apply STATE_NEW
    def process(content):
        entries = parse_subtitle_entries(content)
        return [format_srt_entry(e) for e in state_new_entries(entries)]

    # Apply STATE_NEW transformation to all content lists
    content_lists = apply_to_all_content_lists(content_lists, process)

    # Final validation of the transformed content
    translated_srt_content = content_lists[0]
    if translated_srt_content:
        tqdm_success(
            f"Successfully created {len(translated_srt_content)} subtitle entries for {file_name}"
        )

        # Verify no duplicate indices
        indices = set()
        duplicate_found = False
        all_indices = []
        for entry_text in translated_srt_content:
            lines = entry_text.strip().split("\n")
            if len(lines) >= 1:
                try:
                    idx = int(lines[0])
                    all_indices.append(idx)
                    if idx in indices:
                        duplicate_found = True
                        # Silently note duplicate indices
                    indices.add(idx)
                except (ValueError, IndexError):
                    pass

        # Check for consecutive indices
        if sorted(all_indices) != list(range(1, len(all_indices) + 1)):
            tqdm.write(
                f"CRITICAL: Non-consecutive indices after STATE_NEW transformation in {file_name}"
            )
            tqdm.write(f"Indices: {sorted(all_indices)}")
        else:
            tqdm.write(f"Verified consecutive indices in final output for {file_name}")

        if duplicate_found:
            # Handle duplicate indices by forcing another STATE_NEW transformation
            content_lists = apply_to_all_content_lists(content_lists, process)
    else:
        tqdm.write(f"Warning: No valid subtitle entries generated for {file_name}")

    # Fix overlapping subtitle timings in all formats
    # Define a wrapper function to capture the return values we need
    def fix_overlaps_wrapper(srt_content):
        result, short_count, fixed_count = fix_overlapping_subtitles(srt_content)
        return result

    # Apply the fix_overlaps_wrapper to all content lists
    total_short_duration = 0
    total_fixed_overlaps = 0

    # Apply the fix to each list individually to track counts
    for i, content_list in enumerate(content_lists):
        result, short_count, fixed_count = fix_overlapping_subtitles(content_list)
        content_lists = list(content_lists)  # Convert to list to allow item assignment
        content_lists[i] = result
        content_lists = tuple(content_lists)  # Convert back to tuple
        total_short_duration += short_count
        total_fixed_overlaps += fixed_count

    if total_short_duration > 0:
        tqdm.write(f"Fixed {total_short_duration} subtitles with short durations")

    if total_fixed_overlaps > 0:
        tqdm.write(f"Fixed {total_fixed_overlaps} overlapping subtitles")

    return content_lists


def write_output_files(content_lists, output_paths):
    """Write all output files with the AI translation marker.

    Args:
        content_lists: Tuple containing all content lists
        output_paths: Tuple of output file paths
    """
    (
        translated_srt_content,
        translated_srt_content_with_font,
        english_srt_content,
        chinese_srt_content,
        english_srt_content_with_font,
        chinese_srt_content_with_font,
    ) = content_lists
    (
        output_path,
        output_path_with_font,
        output_path_en,
        output_path_cn,
        output_path_en_with_font,
        output_path_cn_with_font,
    ) = output_paths

    # Add the AI translated marker to each file
    final_content = "".join(translated_srt_content) + f"{AI_TRANSLATED_MARKER}\n"
    final_content_with_font = (
        "".join(translated_srt_content_with_font) + f"{AI_TRANSLATED_MARKER}\n"
    )
    final_content_en = "".join(english_srt_content) + f"{AI_TRANSLATED_MARKER}\n"
    final_content_cn = "".join(chinese_srt_content) + f"{AI_TRANSLATED_MARKER}\n"
    final_content_en_with_font = (
        "".join(english_srt_content_with_font) + f"{AI_TRANSLATED_MARKER}\n"
    )
    final_content_cn_with_font = (
        "".join(chinese_srt_content_with_font) + f"{AI_TRANSLATED_MARKER}\n"
    )

    # Map content to paths
    content_map = {
        output_path: final_content,
        output_path_with_font: final_content_with_font,
        output_path_en: final_content_en,
        output_path_cn: final_content_cn,
        output_path_en_with_font: final_content_en_with_font,
        output_path_cn_with_font: final_content_cn_with_font,
    }

    # Write all files
    for path, content in content_map.items():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            tqdm.write(f"Error writing to {path}: {e}")
            return False

    return True


def safe_read_file(file_path, file_name=None):
    """Safely read content from a file, handling exceptions.

    Args:
        file_path: Path to the file to read
        file_name: Optional file name for logging purposes

    Returns:
        Tuple of (success, content)
        - success: True if file was read successfully
        - content: File content if successful, None otherwise
    """
    if file_name is None:
        file_name = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return True, content
    except Exception as e:
        tqdm.write(f"Error reading file {file_name}: {e}")
        return False, None


def read_srt_from_file(file_path):
    """Read SRT file content from the given path."""
    success, content = safe_read_file(file_path)
    if not success:
        return None
    return content


def validate_srt_file(file_path):
    """Validates an SRT file for basic structure and encoding issues."""
    file_name = os.path.basename(file_path)

    # Check file exists
    if not os.path.exists(file_path):
        return False, f"File {file_name} does not exist"

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, f"File {file_name} is empty (0 bytes)"

    # Check if file is readable and has valid SRT format
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic SRT format check (should have numbers, timestamps, and content)
        if not re.search(
            r"\d+\s+\d{2}:\d{2}:\d{2}[,\.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,\.]\d{3}",
            content,
        ):
            return (
                False,
                f"File {file_name} does not appear to be a valid SRT file (missing timestamp format)",
            )

        # Check for at least one valid entry
        entries = parse_srt(content)
        if not entries:
            return False, f"File {file_name} contains no valid subtitle entries"

        return True, f"File {file_name} is valid with {len(entries)} subtitle entries"

    except UnicodeDecodeError:
        # Try another encoding
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            return (
                False,
                f"File {file_name} has encoding issues (not UTF-8). Consider converting to UTF-8.",
            )
        except:
            return False, f"File {file_name} has unknown encoding issues"
    except Exception as e:
        return False, f"Error validating file {file_name}: {str(e)}"


# Colored output helper functions
def print_info(message):
    """Print informational message in cyan."""
    print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")


def print_success(message):
    """Print success message in green."""
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")


def print_important(message):
    """Print success message in green."""
    print(f"{Fore.MAGENTA}{message}{Style.RESET_ALL}")


def print_warning(message):
    """Print warning message in yellow."""
    print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")


def print_error(message):
    """Print error message in red."""
    print(f"{Fore.RED}{message}{Style.RESET_ALL}")


def tqdm_info(message):
    """Write informational message to tqdm in cyan."""
    tqdm.write(f"{Fore.CYAN}{message}{Style.RESET_ALL}")


def tqdm_success(message):
    """Write success message to tqdm in green."""
    tqdm.write(f"{Fore.GREEN}{message}{Style.RESET_ALL}")


def tqdm_warning(message):
    """Write warning message to tqdm in yellow."""
    tqdm.write(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")


def tqdm_error(message):
    """Write error message to tqdm in red."""
    tqdm.write(f"{Fore.RED}{message}{Style.RESET_ALL}")


def tqdm_debug(message):
    """Write debug message to tqdm in blue."""
    tqdm.write(f"{Fore.BLUE}{message}{Style.RESET_ALL}")


def print_config(args):
    print_info("\n")
    print_info(f"Input folder: {INPUT_FOLDER}")
    print_info(f"Output folder: {OUTPUT_FOLDER}")
    print_info(f"Chinese-only output folder: {OUTPUT_FOLDER_CN}")
    print_info(f"English-only output folder: {OUTPUT_FOLDER_EN}")
    print_info(
        f"Bilingual with font size output folder: {OUTPUT_FOLDER_WITH_FONT_SIZE}"
    )
    print_info(
        f"Chinese-only with font size output folder: {OUTPUT_FOLDER_CN_WITH_FONT_SIZE}"
    )
    print_info(
        f"English-only with font size output folder: {OUTPUT_FOLDER_EN_WITH_FONT_SIZE}"
    )
    print_info(f"LLM responses folder: {LLM_RESPONSES_FOLDER}")
    print_important(f"Ollama IP: {IP_OLLAMA}")
    print_info(f"Ollama port: {PORT_OLLAMA}")
    print_important(f"LLM Model: {LLM_MODEL}")
    print_info(f"Batch size: {BATCH_SIZE} subtitle entries per API call")
    print_info(f"Random order: {not args.no_random_order}")
    print_info("\n")


def setup_output_directories():
    """Create all necessary output directories if they don't exist."""
    directories = {
        "main": OUTPUT_FOLDER,
        "Chinese-only": OUTPUT_FOLDER_CN,
        "English-only": OUTPUT_FOLDER_EN,
        "bilingual with font size": OUTPUT_FOLDER_WITH_FONT_SIZE,
        "Chinese-only with font size": OUTPUT_FOLDER_CN_WITH_FONT_SIZE,
        "English-only with font size": OUTPUT_FOLDER_EN_WITH_FONT_SIZE,
        "LLM responses": LLM_RESPONSES_FOLDER,
    }

    for desc, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print_success(f"Created {desc} output folder: '{path}'")

    return True


def process_all_files(srt_files, random_order=True, args=None):
    """Process multiple SRT files, tracking results.

    Args:
        srt_files: List of SRT file paths to process
        random_order: Whether to process files in random order

    Returns:
        Tuple of (processed_count, skipped_count, error_count)
    """
    if not srt_files:
        print_warning("No SRT files found in input directory")
        return 0, 0, 0

    print_info(f"Found {len(srt_files)} SRT files to process")

    # Randomize processing order if requested
    if random_order:
        random.shuffle(srt_files)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Process each file
    for i, file_path in enumerate(tqdm(srt_files, desc="Processing files")):
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, file_name)

        # Skip already translated files
        if os.path.exists(output_path):
            tqdm_info(f"Skipping already translated file: {file_name}")
            skipped_count += 1
            continue

        try:
            # Process the file
            print_config(args)
            tqdm_info(f"Processing file {i+1}/{len(srt_files)}: {file_name}")
            process_file(file_path, output_path)

            # Check if output was created successfully
            if os.path.exists(output_path):
                processed_count += 1
            else:
                error_count += 1
                tqdm_error(f"Failed to create output for {file_name}")

        except Exception as e:
            error_count += 1
            tqdm_error(f"Error processing file {file_name}: {e}")
            import traceback

            tqdm_error(traceback.format_exc())

    return processed_count, skipped_count, error_count


def process_retry_batches(
    subtitle_entries, processed_indices, content_lists, max_retries=None
):
    """Retry processing failed entries with dynamic retry logic.

    Args:
        subtitle_entries: List of all subtitle entries
        processed_indices: Set of already processed indices
        content_lists: Tuple of content lists
        max_retries: Maximum number of retry rounds (if None, calculated dynamically)

    Returns:
        Number of successfully retried entries
    """
    # Get all unprocessed indices
    unprocessed_indices = get_unprocessed_indices(subtitle_entries, processed_indices)

    if not unprocessed_indices:
        return 0

    if max_retries is None:
        import math

        max_retries = max(5, int(math.sqrt(len(subtitle_entries))))

    tqdm_info(
        f"Retrying {len(unprocessed_indices)} failed entries with {max_retries} retry rounds..."
    )

    # Create a list of entries that need to be retried
    retry_entries = [e for e in subtitle_entries if e.index in unprocessed_indices]
    initial_unprocessed_count = len(unprocessed_indices)

    # Initialize backoff delay for rounds
    retry_delay = INITIAL_RETRY_DELAY  # Start with initial delay (e.g., 3 seconds)

    # Try up to max_retries rounds
    for round_num in range(max_retries):
        if not unprocessed_indices:  # If all entries have been processed, break
            break

        # Apply exponential backoff between rounds (except first round)
        if round_num > 0:
            # Add jitter (80-120%) to avoid synchronized retries
            jittered_delay = retry_delay * random.uniform(0.8, 1.2)
            tqdm_info(
                f"Retry round {round_num+1}/{max_retries}: Waiting {jittered_delay:.1f}s before next round..."
            )
            time.sleep(jittered_delay)
            retry_delay = min(
                16, 2 * retry_delay
            )  # Double delay for next round, cap at 16 seconds
        else:
            tqdm_info(f"Starting retry round 1/{max_retries}...")

        # Determine batch size based on progress through retry rounds
        progress = round_num / max_retries

        if progress < 0.10:
            batch_size = 8
        elif progress < 0.25:
            batch_size = 6
        elif progress < 0.5:
            batch_size = 4
        elif progress < 0.75:
            batch_size = 2
        else:
            batch_size = 1

        # Create smaller batches for retry
        retry_batches = list(create_entry_batch(retry_entries, batch_size=batch_size))

        # Process retry batches - simplified to one attempt per batch per round
        for batch in retry_batches:
            should_skip, filtered_batch = filter_batch_by_processed(
                batch, processed_indices
            )
            if should_skip:
                continue

            try:
                # Get translations for the batch
                processed_entries = get_translation_for_batch(filtered_batch)

                # Process the batch entries with the unprocessed_indices set
                process_batch_entries(
                    processed_entries,
                    filtered_batch,
                    processed_indices,
                    unprocessed_indices,
                    content_lists,
                )
            except Exception as e:
                entry_indices = [e.index for e in filtered_batch]
                tqdm_error(
                    f"Error processing batch with indices {entry_indices}: {str(e)}"
                )

            # Special handling for the last round - try individual entries if needed
            if batch_size > 1 and round_num == max_retries - 1:
                # Check if any entries in this batch are still unprocessed
                still_unprocessed = [
                    e for e in filtered_batch if e.index in unprocessed_indices
                ]
                if still_unprocessed:
                    tqdm_info(
                        f"Last round: Attempting individual processing for {len(still_unprocessed)} entries..."
                    )
                    for entry in still_unprocessed:
                        try:
                            single_batch = [entry]
                            processed_entries = get_translation_for_batch(single_batch)
                            process_batch_entries(
                                processed_entries,
                                single_batch,
                                processed_indices,
                                unprocessed_indices,
                                content_lists,
                            )
                        except Exception:
                            pass

        # Update retry entries for next round
        retry_entries = [e for e in subtitle_entries if e.index in unprocessed_indices]

        if not retry_entries:
            # All entries processed successfully
            tqdm_success(f"All entries successfully processed in round {round_num+1}")
            break
        else:
            remaining = len(retry_entries)
            tqdm_info(
                f"Round {round_num+1} completed: {initial_unprocessed_count - remaining} recovered, {remaining} still pending"
            )

    # Calculate how many entries were successfully retried
    successfully_retried = initial_unprocessed_count - len(unprocessed_indices)

    # Only output success or failure message at the end
    if successfully_retried > 0:
        tqdm_success(
            f"Retry successful: Recovered {successfully_retried} of {initial_unprocessed_count} failed entries"
        )

    if unprocessed_indices:
        tqdm_warning(
            f"Retry failed: {len(unprocessed_indices)} entries could not be processed after {max_retries} attempts"
        )

    return successfully_retried


def process_batches(subtitle_entries, processed_indices, content_lists, file_name):
    """Process subtitle entries in batches.

    Args:
        subtitle_entries: List of all subtitle entries
        processed_indices: Set to track processed entries
        content_lists: Tuple of content lists
        file_name: File name for logging purposes

    Returns:
        Number of successfully processed entries
    """
    success_count = 0

    # Process entries in batches
    for batch in tqdm(
        list(create_entry_batch(subtitle_entries)),
        desc=f"Processing {file_name}",
        leave=False,
    ):
        should_skip, filtered_batch = filter_batch_by_processed(
            batch, processed_indices
        )
        if should_skip:
            continue

        try:
            # Get translations for the batch
            processed_entries = get_translation_for_batch(filtered_batch)

            # Process the batch entries
            count = process_batch_entries(
                processed_entries,
                filtered_batch,
                processed_indices,
                None,
                content_lists,
            )
            success_count += count

        except Exception as e:
            tqdm.write(f"Error processing batch: {e}")
            # Continue with the next batch

    return success_count


def initialize_content_lists():
    """Initialize empty content lists for all output formats.

    Returns:
        A tuple of empty content lists
    """
    return (
        [],  # translated_srt_content
        [],  # translated_srt_content_with_font
        [],  # english_srt_content
        [],  # chinese_srt_content
        [],  # english_srt_content_with_font
        [],  # chinese_srt_content_with_font
    )


def get_output_paths(file_name):
    """Generate all output file paths for a given input file.

    Args:
        file_name: Base file name for output files

    Returns:
        A tuple of output paths
    """
    return (
        os.path.join(OUTPUT_FOLDER, file_name),
        os.path.join(OUTPUT_FOLDER_WITH_FONT_SIZE, file_name),
        os.path.join(OUTPUT_FOLDER_EN, file_name),
        os.path.join(OUTPUT_FOLDER_CN, file_name),
        os.path.join(OUTPUT_FOLDER_EN_WITH_FONT_SIZE, file_name),
        os.path.join(OUTPUT_FOLDER_CN_WITH_FONT_SIZE, file_name),
    )


def preprocess_subtitle_entries(subtitle_entries, file_name):
    """Filter and preprocess subtitle entries.

    Args:
        subtitle_entries: List of parsed subtitle entries
        file_name: File name for logging

    Returns:
        Preprocessed list of subtitle entries
    """
    if not subtitle_entries:
        tqdm_warning(f"No valid subtitle entries found in {file_name}")
        return None

    # Apply STATE_NEW transformation
    original_count = len(subtitle_entries)
    subtitle_entries = state_new_entries(subtitle_entries)
    processed_count = len(subtitle_entries)

    if processed_count < original_count:
        tqdm_info(
            f"Processed {original_count} entries into {processed_count} STATE_NEW entries"
        )

    # Verify consecutive indices
    indices = [entry.index for entry in subtitle_entries]
    if indices != list(range(1, len(subtitle_entries) + 1)):
        tqdm_error(f"CRITICAL: Non-consecutive indices in {file_name}")
        tqdm_error(f"Indices: {indices}")
    else:
        tqdm_success(f"Verified consecutive indices for {file_name}")

    return subtitle_entries


def apply_to_all_content_lists(content_lists, operation, *args, **kwargs):
    """Apply an operation to each content list.

    Args:
        content_lists: Tuple of content lists
        operation: Function to apply to each list
        *args, **kwargs: Additional arguments to pass to the operation

    Returns:
        Tuple of updated content lists
    """
    (
        translated_srt_content,
        translated_srt_content_with_font,
        english_srt_content,
        chinese_srt_content,
        english_srt_content_with_font,
        chinese_srt_content_with_font,
    ) = content_lists

    # Apply the operation to each content list
    translated_srt_content = operation(translated_srt_content, *args, **kwargs)
    translated_srt_content_with_font = operation(
        translated_srt_content_with_font, *args, **kwargs
    )
    english_srt_content = operation(english_srt_content, *args, **kwargs)
    chinese_srt_content = operation(chinese_srt_content, *args, **kwargs)
    english_srt_content_with_font = operation(
        english_srt_content_with_font, *args, **kwargs
    )
    chinese_srt_content_with_font = operation(
        chinese_srt_content_with_font, *args, **kwargs
    )

    return (
        translated_srt_content,
        translated_srt_content_with_font,
        english_srt_content,
        chinese_srt_content,
        english_srt_content_with_font,
        chinese_srt_content_with_font,
    )


def filter_batch_by_processed(batch, processed_indices):
    """Filter a batch to only include unprocessed entries.

    Args:
        batch: List of subtitle entries
        processed_indices: Set of already processed indices

    Returns:
        Tuple of (should_skip, filtered_batch)
        - should_skip: True if batch should be skipped
        - filtered_batch: Filtered list of entries
    """
    # Skip batches where all entries are already processed
    if all(e.index in processed_indices for e in batch):
        return True, []

    # Filter out already processed entries
    filtered_batch = [e for e in batch if e.index not in processed_indices]
    if not filtered_batch:
        return True, []

    return False, filtered_batch


def get_unprocessed_indices(subtitle_entries, processed_indices):
    """Get indices of entries that haven't been processed yet.

    Args:
        subtitle_entries: List of subtitle entries
        processed_indices: Set of already processed indices

    Returns:
        Set of unprocessed entry indices
    """
    return {
        entry.index
        for entry in subtitle_entries
        if entry.index not in processed_indices
    }


def state_new_entries(subtitle_entries):
    """
    Transforms subtitle entries into STATE_NEW format with:
    - Continuous consecutive indices (1,2,3,...,N)
    - Chronological ordering by start time
    - Valid timing (start < end)
    - Minimum 500ms duration
    - Minimum 50ms gap between entries

    Args:
        subtitle_entries: List of SubtitleEntry objects

    Returns:
        List of SubtitleEntry objects in STATE_NEW format
    """
    # Filter out entries with empty text
    valid_entries = [
        e
        for e in subtitle_entries
        if clean_text(e.text).strip() and not is_non_translatable_entry(e.text)
    ]

    # Merge consecutive entries with identical text
    merged_entries = merge_similar_entries(valid_entries)

    # Sort entries chronologically by start time
    merged_entries.sort(key=lambda e: timestamp_to_milliseconds(e.start_time))

    # Apply STATE_NEW transformations
    prev_end_ms = 0
    for i, entry in enumerate(merged_entries, 1):
        # Update index to continuous sequence
        entry.index = i

        # Parse and validate timestamps
        start_ms = timestamp_to_milliseconds(entry.start_time)
        end_ms = timestamp_to_milliseconds(entry.end_time)

        # Fix timing issues
        start_ms = max(start_ms, prev_end_ms + 50)  # Ensure minimum gap
        if start_ms >= end_ms:
            end_ms = start_ms + MIN_SUBTITLE_DURATION
        if end_ms - start_ms < MIN_SUBTITLE_DURATION:
            end_ms = start_ms + MIN_SUBTITLE_DURATION

        # Update timestamps
        entry.start_time = format_timestamp(start_ms)
        entry.end_time = format_timestamp(end_ms)
        prev_end_ms = end_ms

    return merged_entries


def main():
    """Main function to orchestrate the SRT translation process."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Translate SRT files using LLM")
        parser.add_argument(
            "--no-random-order",
            "-n",
            action="store_true",
            help="Process files in original order instead of random order (default: random order)",
        )
        args = parser.parse_args()

        print_info("Starting SRT translation process...")
        print_config(args)

        print("-" * 50)

        if not os.path.isdir(INPUT_FOLDER):
            print_error(f"Error: Input folder not found at '{INPUT_FOLDER}'")
            return

        if not setup_output_directories():
            print_error("Failed to create necessary output directories. Exiting.")
            return 1

        srt_files = glob.glob(os.path.join(INPUT_FOLDER, "*.srt"))
        processed_count, skipped_count, error_count = process_all_files(
            srt_files, random_order=(not args.no_random_order), args=args
        )

        print("-" * 50)
        print_success(f"Processed {processed_count} files")
        print_info(f"Skipped {skipped_count} already translated files")
        if error_count > 0:
            print_warning(f"Encountered errors in {error_count} files")
        print_success("SRT translation process completed!")

    except KeyboardInterrupt:
        print_warning("\nProcess interrupted by user. Exiting.")
        return
    except Exception as e:
        print_error(f"Error in main process: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
