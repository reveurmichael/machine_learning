// COLORS is defined in common.js; no need to redeclare here.

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
// Note: These elements control the *pause* between moves (lower pause = faster playback)
const movePauseDecreaseButton = document.getElementById('speed-up');      // Decreases pause time (faster)
const movePauseIncreaseButton = document.getElementById('speed-down');    // Increases pause time (slower)
const movePauseValueElement = document.getElementById('speed-value');     // Displays pause in seconds
const progressBar = document.getElementById('progress-bar');
const pausedIndicator = document.getElementById('paused-indicator');

// Game state
let gameState = null;
let pixelSize = 0;
let updateInterval = null;
let retryCount = 0;
const MAX_RETRIES = 10;
let isFetching = false;

// Initialize
function init() {
    // Initialize sidebar for replay mode
    initializeSidebar('replay');
    setupEventListeners();
    startPolling();
    document.addEventListener('keydown', handleKeyDown);
}

function setupEventListeners() {
    const safeAdd = (el, evt, fn) => { if (el) el.addEventListener(evt, fn); };

    safeAdd(playPauseButton, 'click', togglePlayPause);
    safeAdd(prevGameButton, 'click', () => sendCommand('prev_game'));
    safeAdd(nextGameButton, 'click', () => sendCommand('next_game'));
    safeAdd(restartButton, 'click', () => sendCommand('restart_game'));
    safeAdd(movePauseDecreaseButton, 'click', () => sendCommand('speed_up'));
    safeAdd(movePauseIncreaseButton, 'click', () => sendCommand('speed_down'));
}

function startPolling() {
    // Poll every 100 ms; avoid overlapping requests.
    updateInterval = setInterval(fetchGameState, 100);
}

async function fetchGameState() {
    if (isFetching) return;
    isFetching = true;
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
    } finally {
        isFetching = false;
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
    
    // Use the sidebar manager to update all UI elements
    if (sidebarManager) {
        sidebarManager.updateWithGameState(gameState);
    }
    
    const safeSet = (el, val) => { if (el) el.textContent = val; };
    const safeShow = (el, show) => { if (el) el.style.display = show ? 'block' : 'none'; };

    // Update document title
    document.title = `Snake Game ${gameState.game_number || 1} - Score: ${gameState.score || 0}`;
    
    // Update progress
    safeSet(progressElement, `${gameState.move_index || 0}/${gameState.total_moves || 0}`);
    
    // Update progress bar
    if (progressBar && gameState.total_moves > 0) {
        const progressPercent = ((gameState.move_index || 0) / gameState.total_moves) * 100;
        progressBar.style.width = `${progressPercent}%`;
    } else {
        if (progressBar) progressBar.style.width = '0%';
    }
    
    // Update paused indicator
    safeShow(pausedIndicator, gameState.paused);
    
    // Update play/pause button
    safeSet(playPauseButton, gameState.paused ? 'Play' : 'Pause');
    
    // Update move pause display in seconds
    if (movePauseValueElement && gameState.pause_between_moves) {
        movePauseValueElement.textContent = `${gameState.pause_between_moves.toFixed(1)}s`;
    } else {
        // Fallback to calculating from speed if pause_between_moves is not provided
        if (movePauseValueElement) {
            const pauseTime = gameState.speed > 0 ? 1.0 / gameState.speed : 1.0;
            movePauseValueElement.textContent = `${pauseTime.toFixed(1)}s`;
        }
    }
}

function drawGame() {
    if (!gameState) return;
    
    const gridSize = gameState.grid_size || 10;
    const maxSize = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.7);
    
    // Calculate pixel size to ensure a perfect fit
    pixelSize = Math.floor(maxSize / gridSize);
    
    // Canvas sizing
    canvas.width  = pixelSize * gridSize;
    canvas.height = pixelSize * gridSize;

    // Clear background
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    drawGrid(ctx, gridSize, pixelSize);

    // Snake
    if (Array.isArray(gameState.snake_positions) && gameState.snake_positions.length) {
        // Body
        for (let i = 0; i < gameState.snake_positions.length - 1; i++) {
            const [x, yGame] = gameState.snake_positions[i];
            drawRect(ctx, x, (gridSize - 1) - yGame, COLORS.SNAKE_BODY, pixelSize);
        }
        // Head
        const [hx, hyGame] = gameState.snake_positions[gameState.snake_positions.length - 1];
        drawRect(ctx, hx, (gridSize - 1) - hyGame, COLORS.SNAKE_HEAD, pixelSize);
    }

    // Apple
    if (Array.isArray(gameState.apple_position) && gameState.apple_position.length === 2) {
        const [ax, ayGame] = gameState.apple_position;
        drawRect(ctx, ax, (gridSize - 1) - ayGame, COLORS.APPLE, pixelSize);
    }
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
        if (data.pause_between_moves) {
            movePauseValueElement.textContent = `${data.pause_between_moves.toFixed(1)}s`;
        } else if (data.speed) {
            // Fallback to calculating from speed if pause_between_moves is not provided
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
            sendCommand('speed_up'); // Up key - speed up (decrease pause time)
            event.preventDefault(); // Prevent page scrolling
            break;
        case 'ArrowDown':
            sendCommand('speed_down'); // Down key - slow down (increase pause time)
            event.preventDefault(); // Prevent page scrolling
            break;
    }
}

// Handle window resize
window.addEventListener('resize', drawGame);

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);