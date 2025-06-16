/**
 * Snake Game Web Human Play
 * JavaScript for controlling the snake game in human play mode
 */

// DOM elements
const loadingMessage = document.getElementById('loading-message');
const gameContainer = document.getElementById('game-container');
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');
const stepsElement = document.getElementById('steps');
const endReasonElement = document.getElementById('end-reason');
const endReasonContainer = document.getElementById('end-reason-container');
const gameOverIndicator = document.getElementById('game-over-indicator');
const resetButton = document.getElementById('reset-button');

// Game state
let gameState = null;
let pixelSize = 0;
let updateInterval = null;
let retryCount = 0;
const MAX_RETRIES = 10;

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
    setupEventListeners();
    startPolling();
}

function setupEventListeners() {
    // Keyboard controls
    document.addEventListener('keydown', handleKeyDown);
    
    // Reset button
    resetButton.addEventListener('click', resetGame);
    
    // Window resize event to redraw the game board
    window.addEventListener('resize', () => {
        if (gameState) {
            drawGame();
        }
    });
}

function startPolling() {
    // Poll for game state every 20ms
    updateInterval = setInterval(fetchGameState, 20);
}

async function fetchGameState() {
    try {
        const data = await sendApiRequest('/api/state');
        
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
        
        // Update colors if provided
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

function updateUI() {
    if (!gameState) return;
    
    // Update score and steps
    scoreElement.textContent = gameState.score;
    stepsElement.textContent = gameState.steps;
    
    // Update document title
    document.title = `Snake Game - Score: ${gameState.score}`;
    
    // Update game over state
    gameOverIndicator.style.display = gameState.game_over ? 'block' : 'none';
    
    // Update end reason if available
    if (gameState.game_end_reason) {
        endReasonElement.textContent = gameState.game_end_reason;
        endReasonContainer.style.display = 'block';
    } else {
        endReasonContainer.style.display = 'none';
    }
}

function drawGame() {
    if (!gameState) return;
    
    const gridSize = gameState.grid_size || 10;
    // Adjust the size calculation to maximize available space
    const maxWidth = window.innerWidth * 0.6;
    const maxHeight = window.innerHeight * 0.8;
    const maxSize = Math.min(maxWidth, maxHeight);
    
    // Calculate pixel size to ensure a perfect fit
    pixelSize = Math.floor(maxSize / gridSize);
    
    // Set canvas dimensions to exactly fit the grid (no extra margins)
    canvas.width = pixelSize * gridSize;
    canvas.height = pixelSize * gridSize;
    
    // Clear canvas
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid lines
    drawGrid(ctx, gridSize, pixelSize);
    
    // Draw snake
    if (gameState.snake_positions && gameState.snake_positions.length > 0) {
        // Draw body first
        for (let i = 0; i < gameState.snake_positions.length - 1; i++) {
            const [x, y] = gameState.snake_positions[i];
            // Convert y-coordinate: y in game logic to y in canvas
            // y=0 should be at bottom, but in canvas y=0 is at top
            const canvasY = (gridSize - 1) - y;
            drawRect(ctx, x, canvasY, COLORS.SNAKE_BODY, pixelSize);
        }
        
        // Draw head
        const head = gameState.snake_positions[gameState.snake_positions.length - 1];
        // Convert y-coordinate for head
        const headCanvasY = (gridSize - 1) - head[1];
        drawRect(ctx, head[0], headCanvasY, COLORS.SNAKE_HEAD, pixelSize);
    }
    
    // Draw apple
    if (gameState.apple_position && gameState.apple_position.length === 2) {
        const [x, y] = gameState.apple_position;
        // Convert y-coordinate for apple
        const appleCanvasY = (gridSize - 1) - y;
        drawRect(ctx, x, appleCanvasY, COLORS.APPLE, pixelSize);
    }
}

function handleKeyDown(event) {
    if (!gameState) return;
    
    // If game over, only respond to R key for reset
    if (gameState.game_over) {
        if (event.key === 'r' || event.key === 'R') {
            resetGame();
        }
        return;
    }
    
    let direction = null;
    
    // Map keys to directions - ensure all keys are captured properly
    switch (event.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
            direction = 'UP';
            break;
        case 'ArrowDown':
        case 's':
        case 'S':
            direction = 'DOWN';
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            direction = 'LEFT';
            break;
        case 'ArrowRight':
        case 'd':
        case 'D':
            direction = 'RIGHT';
            break;
        case 'r':
        case 'R':
            resetGame();
            return;
        default:
            return; // Ignore other keys
    }
    
    if (direction) {
        event.preventDefault(); // Prevent default behavior like scrolling
        makeMove(direction);
    }
}

async function makeMove(direction) {
    try {
        await sendApiRequest('/api/move', 'POST', { direction });
    } catch (error) {
        console.error('Error making move:', error);
    }
}

async function resetGame() {
    try {
        await sendApiRequest('/api/reset', 'POST');
    } catch (error) {
        console.error('Error resetting game:', error);
    }
} 