// Constants - colors will be overridden by values from the server
let COLORS = {
    SNAKE_HEAD: '#3498db',    // Blue for snake head
    SNAKE_BODY: '#2980b9',    // Darker blue for snake body
    APPLE: '#e74c3c',         // Red for apple
    BACKGROUND: '#2c3e50',    // Dark background
    GRID: '#34495e',          // Grid lines
};

// End reason mapping
const END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS": "Max Steps",
    "EMPTY_MOVES": "Empty Moves",
    "ERROR": "LLM Error"
};

// DOM elements
const loadingMessage = document.getElementById('loading-message');
const gameContainer = document.getElementById('game-container');
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const gameNumber = document.getElementById('game-number');
const scoreElement = document.getElementById('score');
const progressElement = document.getElementById('progress');
const endReasonElement = document.getElementById('end-reason');
const endReasonContainer = document.getElementById('end-reason-container');
const primaryLlmElement = document.getElementById('primary-llm');
const secondaryLlmElement = document.getElementById('secondary-llm');
const playPauseButton = document.getElementById('play-pause');
const prevGameButton = document.getElementById('prev-game');
const nextGameButton = document.getElementById('next-game');
const restartButton = document.getElementById('restart');
// Note: These elements use 'speed' in their IDs but control move pause time
// speed-up button actually decreases pause time, speed-down button increases pause time
const movePauseDecreaseButton = document.getElementById('speed-up');      // Decreases pause time
const movePauseIncreaseButton = document.getElementById('speed-down');    // Increases pause time
const movePauseValueElement = document.getElementById('speed-value');     // Displays pause multiplier
const progressBar = document.getElementById('progress-bar');
const pausedIndicator = document.getElementById('paused-indicator');

// Game state
let gameState = null;
let pixelSize = 0;
let updateInterval = null;
let retryCount = 0;
const MAX_RETRIES = 10;

// Initialize
function init() {
    setupEventListeners();
    startPolling();
    document.addEventListener('keydown', handleKeyDown);
}

function setupEventListeners() {
    playPauseButton.addEventListener('click', togglePlayPause);
    prevGameButton.addEventListener('click', () => sendCommand('prev_game'));
    nextGameButton.addEventListener('click', () => sendCommand('next_game'));
    restartButton.addEventListener('click', () => sendCommand('restart_game'));
    movePauseDecreaseButton.addEventListener('click', () => sendCommand('speed_up'));
    movePauseIncreaseButton.addEventListener('click', () => sendCommand('speed_down'));
}

function startPolling() {
    updateInterval = setInterval(fetchGameState, 100);
}

async function fetchGameState() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
            
            if (!gameState && retryCount < MAX_RETRIES) {
                retryCount++;
                return;
            }
            
            if (!gameState && retryCount >= MAX_RETRIES) {
                loadingMessage.innerHTML = `<p class="error-message">Error loading game data: ${data.error}</p>`;
                clearInterval(updateInterval);
                return;
            }
            
            return;
        }
        
        gameState = data;
        
        if (loadingMessage.style.display !== 'none') {
            loadingMessage.style.display = 'none';
            gameContainer.style.display = 'flex';
        }
        
        if (gameState.colors) {
            COLORS.SNAKE_HEAD = rgbArrayToHex(gameState.colors.snake_head) || COLORS.SNAKE_HEAD;
            COLORS.SNAKE_BODY = rgbArrayToHex(gameState.colors.snake_body) || COLORS.SNAKE_BODY;
            COLORS.APPLE = rgbArrayToHex(gameState.colors.apple) || COLORS.APPLE;
            COLORS.BACKGROUND = rgbArrayToHex(gameState.colors.background) || COLORS.BACKGROUND;
            COLORS.GRID = rgbArrayToHex(gameState.colors.grid) || COLORS.GRID;
        }
        
        updateUI();
        drawGame();
    } catch (error) {
        console.error('Error fetching game state:', error);
        
        if (!gameState && retryCount < MAX_RETRIES) {
            retryCount++;
            return;
        }
        
        if (!gameState && retryCount >= MAX_RETRIES) {
            loadingMessage.innerHTML = `<p class="error-message">Error connecting to server: ${error.message}</p>`;
            clearInterval(updateInterval);
            return;
        }
    }
}

function rgbArrayToHex(rgbArray) {
    if (!Array.isArray(rgbArray) || rgbArray.length < 3) {
        return null;
    }
    return `#${rgbArray[0].toString(16).padStart(2, '0')}${rgbArray[1].toString(16).padStart(2, '0')}${rgbArray[2].toString(16).padStart(2, '0')}`;
}

function updateUI() {
    if (!gameState) return;
    
    // Update game info
    gameNumber.textContent = gameState.game_number;
    scoreElement.textContent = gameState.score;
    
    // Update document title
    document.title = `Snake Game ${gameState.game_number} - Score: ${gameState.score}`;
    
    // Update progress
    progressElement.textContent = `${gameState.move_index}/${gameState.total_moves}`;
    
    // Update progress bar
    if (gameState.total_moves > 0) {
        const progressPercent = (gameState.move_index / gameState.total_moves) * 100;
        progressBar.style.width = `${progressPercent}%`;
    } else {
        progressBar.style.width = '0%';
    }
    
    // Update end reason if available
    if (gameState.game_end_reason) {
        const reason = END_REASON_MAP[gameState.game_end_reason] || gameState.game_end_reason;
        endReasonElement.textContent = reason;
        endReasonContainer.style.display = 'block';
    } else {
        endReasonContainer.style.display = 'none';
    }
    
    // Update paused indicator
    pausedIndicator.style.display = gameState.paused ? 'block' : 'none';
    
    // Update LLM info
    primaryLlmElement.textContent = gameState.primary_llm || 'Unknown';
    secondaryLlmElement.textContent = gameState.secondary_llm || 'None';
    
    // Update play/pause button
    playPauseButton.textContent = gameState.paused ? 'Play' : 'Pause';
    
    // Update move pause display in seconds
    if (gameState.move_pause) {
        movePauseValueElement.textContent = `${gameState.move_pause.toFixed(1)}s`;
    } else {
        // Fallback to calculating from speed if move_pause is not provided
        const pauseTime = gameState.speed > 0 ? 1.0 / gameState.speed : 1.0;
        movePauseValueElement.textContent = `${pauseTime.toFixed(1)}s`;
    }
}

function drawGame() {
    if (!gameState) return;
    
    const gridSize = gameState.grid_size || 10;
    const maxSize = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.7);
    
    // Calculate pixel size to ensure a perfect fit
    pixelSize = Math.floor(maxSize / gridSize);
    
    // Set canvas dimensions to exactly fit the grid (no extra margins)
    canvas.width = pixelSize * gridSize;
    canvas.height = pixelSize * gridSize;
    
    // Clear canvas
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid lines
    drawGrid(gridSize, pixelSize);
    
    // Draw snake
    if (gameState.snake_positions && gameState.snake_positions.length > 0) {
        // Draw body first
        for (let i = 0; i < gameState.snake_positions.length - 1; i++) {
            const [x, y] = gameState.snake_positions[i];
            drawRect(x, y, COLORS.SNAKE_BODY);
        }
        
        // Draw head
        const head = gameState.snake_positions[gameState.snake_positions.length - 1];
        drawRect(head[0], head[1], COLORS.SNAKE_HEAD);
    }
    
    // Draw apple
    if (gameState.apple_position && gameState.apple_position.length === 2) {
        const [x, y] = gameState.apple_position;
        drawRect(x, y, COLORS.APPLE);
    }
}

function drawGrid(gridSize, pixelSize) {
    ctx.strokeStyle = COLORS.GRID;
    ctx.lineWidth = 0.5;
    
    // Draw vertical lines
    for (let i = 0; i <= gridSize; i++) {
        ctx.beginPath();
        ctx.moveTo(i * pixelSize, 0);
        ctx.lineTo(i * pixelSize, gridSize * pixelSize);
        ctx.stroke();
    }
    
    // Draw horizontal lines
    for (let i = 0; i <= gridSize; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * pixelSize);
        ctx.lineTo(gridSize * pixelSize, i * pixelSize);
        ctx.stroke();
    }
}

function drawRect(x, y, color) {
    ctx.fillStyle = color;
    
    ctx.fillRect(
        x * pixelSize + 1,
        (gameState.grid_size - 1 - y) * pixelSize + 1,
        pixelSize - 2,
        pixelSize - 2
    );
}

function togglePlayPause() {
    if (!gameState) return;
    const command = gameState.paused ? 'play' : 'pause';
    sendCommand(command);
}

async function sendCommand(command) {
    try {
        const response = await fetch('/api/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ command })
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
        }
        
        // Update move pause display based on response
        if (data.move_pause) {
            movePauseValueElement.textContent = `${data.move_pause.toFixed(1)}s`;
        } else if (data.speed) {
            // Fallback to calculating from speed if move_pause is not provided
            const pauseTime = data.speed > 0 ? 1.0 / data.speed : 1.0;
            movePauseValueElement.textContent = `${pauseTime.toFixed(1)}s`;
        }
    } catch (error) {
        console.error('Error sending command:', error);
    }
}

function handleKeyDown(event) {
    switch (event.key) {
        case ' ': // Space
            togglePlayPause();
            event.preventDefault();
            break;
        case 'ArrowLeft':
            sendCommand('prev_game');
            event.preventDefault();
            break;
        case 'ArrowRight':
            sendCommand('next_game');
            event.preventDefault();
            break;
        case 'r':
        case 'R':
            sendCommand('restart_game');
            event.preventDefault();
            break;
        case 'ArrowUp':
            sendCommand('speed_down'); // Up key - increase pause time
            event.preventDefault(); // Prevent page scrolling
            break;
        case 'ArrowDown':
            sendCommand('speed_up'); // Down key - decrease pause time
            event.preventDefault(); // Prevent page scrolling
            break;
    }
}

// Handle window resize
window.addEventListener('resize', drawGame);

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);