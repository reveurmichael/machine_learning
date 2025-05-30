The continue mode is missing. Bring it back. Here is what continue mode is about:


1. **Purpose of Continue Mode**:
   - Continue mode allows you to add more games to an existing experiment
   - It's designed to help gather more data points to achieve statistical significance according to the central limit theorem
   - The mode preserves all previous game data and statistics while adding new games

2. **How to Use Continue Mode**:
   - Use the `--continue-with-game-in-dir` argument (in mai.py) followed by the directory path of the previous experiment
   - Example: `python main.py --continue-with-game-in-dir ./results/my_experiment_directory`

3. **Restrictions in Continue Mode**:
   - Only the following arguments are allowed when using continue mode:
     - `--max-games`
     - `--no-gui`
     - `--sleep-before-launching`
   - Other arguments like `--provider`, `--model`, `--parser-provider`, etc. are not allowed as they must match the original experiment settings

4. **How it Works**:
   - When continue mode is activated, the system:
     1. Validates the continuation directory exists and contains required files (summary.json and prompts directory)
     2. Reads existing game data and statistics from previous games
     3. Determines the next game number to start from
     4. Cleans any existing prompt files for games that will be replayed
     5. Sets up the game manager with continuation settings
     6. Records continuation metadata including timestamps and session information

5. **Data Management**:
   - Statistics from all games (old and new) are merged into the summary.json file
   - The system tracks:
     - Total games played
     - Total score
     - Total steps
     - Time statistics
     - Token usage statistics
     - Step statistics
     - JSON parsing statistics

6. **Continuation Tracking**:
   - Each continuation is recorded with:
     - A timestamp
     - Continuation count
     - Previous session data
     - Game statistics from the previous session

7. **File Structure**:
   - Games are saved as `game_N.json` files
   - A `summary.json` file contains aggregated statistics
   - A `prompts` directory stores the prompts used
   - A `responses` directory stores the responses

This mode is particularly useful when you want to:
- Add more games to an existing experiment
- Ensure statistical significance of results



On my side, I have already bring utils/continuation_utils.py back. 



Here are the key Python code components that implement continue mode:

1. **Main Entry Point** (`main.py`):
```python
def main():
    try:
        args = parse_arguments()
        if args.continue_with_game_in_dir:
            # Create and run the game manager with continuation
            GameManager.continue_from_directory(args)
        else:
            # Create and run the game manager normally
            game_manager = GameManager(args)
            game_manager.run()
```

2. **GameManager Class** (`game_manager.py`):
```python
class GameManager:
    def continue_from_session(self, log_dir, start_game_number):
        """Continue from a previous game session."""
        # Set up continuation session
        setup_continuation_session(self, log_dir, start_game_number)
        
        # Set up LLM clients
        setup_llm_clients(self)
        
        # Handle game state for continuation
        handle_continuation_game_state(self)
        
        # Run the game loop
        self.run_game_loop()
        
        # Report final statistics
        self.report_final_statistics()

    @classmethod
    def continue_from_directory(cls, args):
        """Factory method to create a GameManager instance for continuation."""
        return continue_from_directory(cls, args)
```

3. **Continuation Utilities** (`utils/continuation_utils.py`):
```python
def continue_from_directory(game_manager_class, args):
    """Factory method to create a GameManager instance for continuation."""
    log_dir = args.continue_with_game_in_dir
    
    # Validate directory and files
    if not os.path.isdir(log_dir):
        print(Fore.RED + f"❌ Continuation directory does not exist: '{log_dir}'")
        sys.exit(1)
        
    # Check for summary.json
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Get next game number
    next_game = get_next_game_number(log_dir)
    
    # Clean existing prompt files
    clean_prompt_files(log_dir, next_game)
    
    # Create game manager with continuation settings
    game_manager = game_manager_class(args)
    args.is_continuation = True
    
    # Continue from session
    try:
        game_manager.continue_from_session(log_dir, next_game)
    except Exception as e:
        print(Fore.RED + f"❌ Error continuing from session: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    return game_manager
```

4. **Game Data Management** (`core/game_data.py`):
```python
def record_continuation(self, previous_session_data=None):
    """Record that this game is a continuation of a previous session."""
    self.is_continuation = True
    self.continuation_count += 1
    continuation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize continuation data structures
    if not hasattr(self, 'continuation_timestamps'):
        self.continuation_timestamps = []
    if not hasattr(self, 'continuation_metadata'):
        self.continuation_metadata = []
    
    # Add current timestamp
    self.continuation_timestamps.append(continuation_timestamp)
    
    # Handle previous session data if provided
    if previous_session_data:
        # Add previous game statistics
        if 'game_statistics' in previous_session_data:
            game_stats = previous_session_data['game_statistics']
            continuation_meta['previous_session'] = {
                'total_games': game_stats.get('total_games', 0),
                'total_score': game_stats.get('total_score', 0),
                'total_steps': game_stats.get('total_steps', 0)
            }
```

5. **Statistics Merging** (`utils/json_utils.py`):
```python
def merge_game_stats_for_continuation(log_dir):
    """Merge statistics from all game files for continuation mode."""
    # Get all game files
    game_files = glob.glob(os.path.join(log_dir, "game_*.json")) + glob.glob(os.path.join(log_dir, "game*.json"))
    
    # Initialize aggregated stats
    aggregated_stats = {
        "game_statistics": {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        },
        "time_statistics": {...},
        "token_usage_stats": {...},
        "step_stats": {...},
        "json_parsing_stats": {...}
    }
    
    # Process each game file
    for game_file in game_files:
        with open(game_file, "r", encoding='utf-8') as f:
            game_data = json.load(f)
            # Update all statistics from the game file
            # ...
```

These components work together to:
1. Validate the continuation directory and required files
2. Read and merge existing game statistics
3. Set up the game manager for continuation
4. Track continuation metadata and timestamps
5. Ensure proper statistics aggregation across all games
6. Maintain the same experimental conditions from the original session

The code is designed to be robust, with proper error handling and validation at each step, ensuring that continuation mode maintains data integrity and statistical accuracy across multiple sessions.










Here is how you can test continue mode:

let's say we have ./logs/experimentA which has had N complete games. 

then, 

python main.py --continue-with-game-in-dir ./logs/experimentA --max-games M 

will let the experiment run additional M - N games. 



1. **Basic Usage** :
```bash
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games M
```
This will run M-N additional games, where N is the number of existing games in the experiment directory.

2. **Additional Allowed Arguments**:
When using continue mode, you can also use these optional arguments:
- `--no-gui`: Run without the graphical interface (text-only mode)
- `--sleep-before-launching`: Time to sleep (in minutes) before launching the program

3. **Restricted Arguments**:
You cannot use these arguments with continue mode as they must match the original experiment:
- `--provider`
- `--model`
- `--parser-provider`
- `--parser-model`
- `--move-pause`
- `--max-steps`
- `--max-empty-moves`

4. **Directory Requirements**:
The experiment directory must contain:
- A `summary.json` file
- A `prompts` directory
- Game files in the format `game_N.json`.

5. **Statistics Handling**:
- All statistics from previous games are preserved
- New game statistics are appended to the existing data
- The `summary.json` file is automatically updated with merged statistics
- Time and token usage statistics are properly tracked across continuations

6. **Multiple Continuations**:
You can continue an experiment multiple times:
```bash
# First continuation
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games M

# Second continuation
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games P

# Third continuation
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games Q
```
Each continuation will:
- Start from the last game number
- Add the specified number of new games (M-N games, P-M games, Q-P games, etc.)
- Update all statistics accordingly

7. **Error Handling**:
- The system will validate the directory and required files before starting
- If any required files are missing, it will show an error message
- If invalid arguments are used, it will show an error message
- The system maintains data integrity even if a continuation is interrupted

8. **Best Practices**:
- Always ensure the previous experiment is completely finished before continuing
- Use meaningful directory names for your experiments
- Keep track of how many games you've run in each experiment
- Consider using the `--no-gui` option for faster execution when running many games
- Use `--sleep-before-launching` if you need to delay the start of a continuation

This mode is particularly useful for:
- Gathering more data points for statistical significance
- Continuing interrupted experiments
- Running long-term experiments in multiple sessions
- Maintaining consistent experimental conditions across multiple runs





# Comprehensive Code Analysis of Continue Mode

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Code Implementation Details](#code-implementation-details)
4. [Data Flow](#data-flow)
5. [Error Handling](#error-handling)
6. [Statistics Management](#statistics-management)
7. [Potential Improvements](#potential-improvements)
8. [Code Examples](#code-examples)

## Overview

Continue mode is a sophisticated feature that allows extending existing experiments by adding more games while maintaining experimental consistency. This analysis provides a detailed examination of its implementation, focusing on code structure, data management, and potential areas for improvement.

## Core Components

### 1. Main Entry Point (`main.py`)

The continue mode is primarily controlled through the main entry point, which handles argument parsing and initialization:

```python
def parse_arguments():
    parser = argparse.ArgumentParser(description='LLM-guided Snake game')
    # ... other argument definitions ...
    parser.add_argument('--continue-with-game-in-dir', type=str, default=None,
                      help='Continue from a previous game session in the specified directory')

    args = parser.parse_args()

    if args.continue_with_game_in_dir:
        # Validate directory and files
        if not os.path.isdir(args.continue_with_game_in_dir):
            raise ValueError(f"Directory '{args.continue_with_game_in_dir}' does not exist")

        # Check for required files
        summary_path = os.path.join(args.continue_with_game_in_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise ValueError(f"Missing summary.json in '{args.continue_with_game_in_dir}'")

        # Validate allowed arguments
        raw_args = ' '.join(sys.argv[1:])
        disallowed_args = [
            "--provider", "--model", "--parser-provider", "--parser-model",
            "--move-pause", "--max-steps", "--max-empty-moves",
        ]  

        for arg in disallowed_args:
            if arg in raw_args and not raw_args.startswith(f"--continue-with-game-in-dir {args.continue_with_game_in_dir}"):
                raise ValueError(f"Cannot use {arg} with --continue-with-game-in-dir")
```

### 2. Game Manager (`game_manager.py`)

The GameManager class handles the core continuation logic:

```python
class GameManager:
    def continue_from_session(self, log_dir, start_game_number):
        """Continue from a previous game session."""
        setup_continuation_session(self, log_dir, start_game_number)
        setup_llm_clients(self)
        handle_continuation_game_state(self)
        self.run_game_loop()
        self.report_final_statistics()

    @classmethod
    def continue_from_directory(cls, args):
        """Factory method to create a GameManager instance for continuation."""
        return continue_from_directory(cls, args)
```

### 3. Continuation Utilities (`utils/continuation_utils.py`)

The continuation utilities provide essential functions for managing continuation sessions:

```python
def continue_from_directory(game_manager_class, args):
    """Factory method to create a GameManager instance for continuation."""
    log_dir = args.continue_with_game_in_dir
    
    # Validate directory and files
    if not os.path.isdir(log_dir):
        print(Fore.RED + f"❌ Continuation directory does not exist: '{log_dir}'")
        sys.exit(1)
        
    # Check for summary.json
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Get next game number
    next_game = get_next_game_number(log_dir)
    
    # Clean existing prompt files
    clean_prompt_files(log_dir, next_game)
    
    # Create and run game manager
    game_manager = game_manager_class(args)
    args.is_continuation = True
    
    try:
        game_manager.continue_from_session(log_dir, next_game)
    except Exception as e:
        print(Fore.RED + f"❌ Error continuing from session: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    return game_manager
```

### 4. Game Data Management (`core/game_data.py`)

The GameData class handles tracking and managing statistics:

```python
def record_continuation(self, previous_session_data=None):
    """Record that this game is a continuation of a previous session."""
    self.is_continuation = True
    self.continuation_count += 1
    continuation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize continuation data structures
    if not hasattr(self, 'continuation_timestamps'):
        self.continuation_timestamps = []
    if not hasattr(self, 'continuation_metadata'):
        self.continuation_metadata = []
    
    # Add current timestamp
    self.continuation_timestamps.append(continuation_timestamp)
    
    # Handle previous session data
    if previous_session_data:
        if 'game_statistics' in previous_session_data:
            game_stats = previous_session_data['game_statistics']
            continuation_meta['previous_session'] = {
                'total_games': game_stats.get('total_games', 0),
                'total_score': game_stats.get('total_score', 0),
                'total_steps': game_stats.get('total_steps', 0)
            }
```

### 5. Statistics Management (`utils/json_utils.py`)

The statistics management system handles merging and updating game statistics:

```python
def merge_game_stats_for_continuation(log_dir):
    """Merge statistics from all game files for continuation mode."""
    game_files = glob.glob(os.path.join(log_dir, "game_*.json")) + glob.glob(os.path.join(log_dir, "game*.json"))
    
    aggregated_stats = {
        "game_statistics": {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        },
        "time_statistics": {
            "total_llm_communication_time": 0,
            "total_game_movement_time": 0,
            "total_waiting_time": 0
        },
        "token_usage_stats": {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        },
        "step_stats": {
            "empty_steps": 0,
            "error_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0
        },
        "json_parsing_stats": {
            "total_extraction_attempts": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "json_decode_errors": 0,
            "text_extraction_errors": 0,
            "pattern_extraction_success": 0
        }
    }
```

## Data Flow

1. **Initialization Flow**:
   - Command line arguments are parsed
   - Directory and file validation occurs
   - Game manager is initialized with continuation settings
   - Previous game data is loaded

2. **Game Execution Flow**:
   - Game state is synchronized with previous session
   - New games are executed
   - Statistics are collected and merged
   - Results are saved

3. **Statistics Flow**:
   - Previous statistics are loaded
   - New game statistics are collected
   - Statistics are merged
   - Updated statistics are saved

## Error Handling

The system implements comprehensive error handling:

1. **Directory Validation**:
```python
if not os.path.isdir(log_dir):
    print(Fore.RED + f"❌ Continuation directory does not exist: '{log_dir}'")
    sys.exit(1)
```

2. **File Validation**:
```python
if not os.path.exists(summary_path):
    print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
    sys.exit(1)
```

3. **Argument Validation**:
```python
for arg in disallowed_args:
    if arg in raw_args and not raw_args.startswith(f"--continue-with-game-in-dir {args.continue_with_game_in_dir}"):
        raise ValueError(f"Cannot use {arg} with --continue-with-game-in-dir")
```

4. **Exception Handling**:
```python
try:
    game_manager.continue_from_session(log_dir, next_game)
except Exception as e:
    print(Fore.RED + f"❌ Error continuing from session: {e}")
    traceback.print_exc()
    sys.exit(1)
```

## Statistics Management

The statistics management system is comprehensive and handles:

1. **Game Statistics**:
   - Total games played
   - Total score
   - Total steps
   - Individual game scores

2. **Time Statistics**:
   - LLM communication time
   - Game movement time
   - Waiting time

3. **Token Usage**:
   - Primary LLM tokens
   - Secondary LLM tokens
   - Prompt and completion tokens

4. **Step Statistics**:
   - Empty steps
   - Error steps
   - Valid steps
   - Invalid reversals

5. **JSON Parsing Statistics**:
   - Extraction attempts
   - Successful extractions
   - Failed extractions
   - Various error types

## Potential Improvements

1. **Code Organization**:
   - Consider splitting the continuation logic into a separate module
   - Implement a dedicated ContinuationManager class
   - Create interfaces for different types of continuations

2. **Error Handling**:
   - Implement more specific exception types
   - Add recovery mechanisms for partial failures
   - Improve error messages and logging

3. **Statistics Management**:
   - Implement real-time statistics updates
   - Add statistical analysis tools
   - Improve data visualization capabilities

4. **Performance**:
   - Implement parallel processing for statistics merging
   - Add caching for frequently accessed data
   - Optimize file I/O operations

5. **User Interface**:
   - Add progress indicators for long operations
   - Implement better feedback for continuation status
   - Add visualization of continuation progress

6. **Testing**:
   - Add comprehensive unit tests
   - Implement integration tests
   - Add performance benchmarks

7. **Documentation**:
   - Add more detailed inline documentation
   - Create usage examples
   - Document error scenarios and solutions

8. **Configuration**:
   - Make more parameters configurable
   - Add configuration validation
   - Implement configuration versioning

9. **Data Management**:
   - Implement data compression
   - Add data backup mechanisms
   - Improve data integrity checks

10. **Security**:
    - Add input validation
    - Implement access control
    - Add data encryption options

## Code Examples

### Example 1: Basic Continuation Usage
```python
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games 10
```

### Example 2: Continuation with GUI Disabled
```python
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games 10 --no-gui
```

### Example 3: Continuation with Delayed Start
```python
python main.py --continue-with-game-in-dir ./logs/experimentA --max-games 10 --sleep-before-launching 5
```

## Conclusion

The continue mode implementation is robust and well-structured, providing a solid foundation for extending experiments. While there are areas for improvement, the current implementation successfully handles the core requirements of maintaining experimental consistency while adding new games to existing experiments.

The system's modular design allows for easy extension and modification, and the comprehensive error handling ensures reliable operation. The statistics management system provides detailed insights into the experiment's progress and results.

Future improvements should focus on enhancing user experience, optimizing performance, and adding more advanced features for data analysis and visualization.
