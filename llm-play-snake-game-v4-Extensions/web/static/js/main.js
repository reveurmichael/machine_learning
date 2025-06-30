/**
 * Snake Game – Live LLM Mode
 * Polls /api/state to render the game board and update sidebar information.
 * Reuses helper functions from common.js
 */

// DOM elements
const loadingMessage = document.getElementById('loading-message');
const gameContainer  = document.getElementById('game-container');
const canvas         = document.getElementById('game-canvas');
const ctx            = canvas.getContext('2d');
const scoreElement   = document.getElementById('score');
const stepsElement   = document.getElementById('steps');
const plannedMovesEl = document.getElementById('planned-moves');
const llmResponseEl  = document.getElementById('llm-response');
const finishedIndicator = document.getElementById('finished-indicator');

// Game state
let gameState = null;
let pixelSize = 0;
let updateInterval = null;
let retryCount = 0;
const MAX_RETRIES = 10;

// Prevent overlapping fetches
let isFetching = false;

// Track last rendered step to skip unnecessary draws
let lastRenderedStep = -1;

// On load
document.addEventListener('DOMContentLoaded', init);

function init() {
    // Initialize sidebar for LLM mode
    initializeSidebar('llm');
    startPolling();
    window.addEventListener('resize', () => {
        if (gameState) drawGame();
    });
}

function startPolling() {
    // Poll every 100 ms; overlap guard (`isFetching`) already present
    updateInterval = setInterval(fetchGameState, 100);
}

async function fetchGameState() {
    if (isFetching) return; // Skip if previous request still in flight
    isFetching = true;
    try {
        const data = await sendApiRequest('/api/state');

        if (data.error) {
            handleError(data.error);
            return;
        }

        gameState = data;

        if (loadingMessage.style.display !== 'none') {
            loadingMessage.style.display = 'none';
            gameContainer.style.display = 'flex';
        }

        // Update colour palette if the server provides overrides
        if (gameState.colors) {
            COLORS.SNAKE_HEAD = rgbArrayToHex(gameState.colors.snake_head) || COLORS.SNAKE_HEAD;
            COLORS.SNAKE_BODY = rgbArrayToHex(gameState.colors.snake_body) || COLORS.SNAKE_BODY;
            COLORS.APPLE      = rgbArrayToHex(gameState.colors.apple)      || COLORS.APPLE;
            COLORS.BACKGROUND = rgbArrayToHex(gameState.colors.background) || COLORS.BACKGROUND;
            COLORS.GRID       = rgbArrayToHex(gameState.colors.grid)       || COLORS.GRID;
        }

        updateUI();

        // Only redraw if new step arrived
        if (gameState.steps !== lastRenderedStep) {
            drawGame();
            lastRenderedStep = gameState.steps;
        }
    } catch (e) {
        handleError(e.message);
    } finally {
        isFetching = false;
    }
}

function handleError(msg) {
    console.error('Error fetching game state:', msg);
    if (!gameState && retryCount < MAX_RETRIES) {
        retryCount++; return;
    }
    if (!gameState && retryCount >= MAX_RETRIES) {
        loadingMessage.innerHTML = `<p class="error-message">${msg}</p>`;
        clearInterval(updateInterval);
    }
}

function updateUI() {
    if (!gameState) return;

    // Use the sidebar manager to update all UI elements
    if (sidebarManager) {
        sidebarManager.updateWithGameState(gameState);
    }

    // Update page title
    document.title = `Snake ${gameState.score} pts – ${gameState.steps} steps`;
}

function drawGame() {
    if (!gameState) return;

    const gridSize = gameState.grid_size || 10;
    const maxSize = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.8);
    pixelSize = Math.floor(maxSize / gridSize);

    canvas.width  = pixelSize * gridSize;
    canvas.height = pixelSize * gridSize;

    // Clear canvas
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid(ctx, gridSize, pixelSize);

    // Draw snake body first
    if (gameState.snake_positions && gameState.snake_positions.length > 0) {
        for (let i = 0; i < gameState.snake_positions.length - 1; i++) {
            const [x, yGame] = gameState.snake_positions[i];
            const yCanvas = (gridSize - 1) - yGame;
            drawRect(ctx, x, yCanvas, COLORS.SNAKE_BODY, pixelSize);
        }
        // Head
        const [hx, hyGame] = gameState.snake_positions[gameState.snake_positions.length - 1];
        drawRect(ctx, hx, (gridSize - 1) - hyGame, COLORS.SNAKE_HEAD, pixelSize);
    }

    // Draw apple
    if (gameState.apple_position && gameState.apple_position.length === 2) {
        const [ax, ayGame] = gameState.apple_position;
        drawRect(ctx, ax, (gridSize - 1) - ayGame, COLORS.APPLE, pixelSize);
    }
} 