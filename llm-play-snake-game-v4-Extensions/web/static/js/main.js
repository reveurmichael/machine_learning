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

    // Update basic game information
    if (scoreElement) {
        scoreElement.textContent = gameState.score || 0;
    }
    
    if (stepsElement) {
        stepsElement.textContent = gameState.steps || 0;
    }

    // Update planned moves
    if (plannedMovesEl) {
        if (gameState.planned_moves && gameState.planned_moves.length > 0) {
            plannedMovesEl.textContent = gameState.planned_moves.join(' → ');
        } else {
            plannedMovesEl.textContent = 'No moves planned yet';
        }
    }

    // Update LLM response
    if (llmResponseEl) {
        if (gameState.llm_response) {
            llmResponseEl.textContent = gameState.llm_response;
        } else {
            llmResponseEl.textContent = 'Waiting for LLM response...';
        }
    }

    // Update finished indicator
    if (finishedIndicator) {
        if (gameState.session_finished) {
            finishedIndicator.style.display = 'block';
        } else {
            finishedIndicator.style.display = 'none';
        }
    }

    // Update page title
    document.title = `Snake ${gameState.score} pts – ${gameState.steps} steps`;
}

function drawGame() {
    if (!gameState || !canvas) return;

    const gridSize = gameState.grid_size || 10;
    const maxSize = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.8);
    pixelSize = Math.floor(maxSize / gridSize);

    // Set canvas size
    const canvasSize = pixelSize * gridSize;
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // Clear canvas
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Draw grid
    drawGrid(ctx, gridSize, pixelSize);

    // Draw snake
    if (gameState.snake_positions && gameState.snake_positions.length > 0) {
        // Draw snake body first
        for (let i = 0; i < gameState.snake_positions.length - 1; i++) {
            const [x, yGame] = gameState.snake_positions[i];
            const yCanvas = (gridSize - 1) - yGame;  // Flip Y-axis: Python (0,0) at bottom-left → Canvas (0,0) at top-left
            drawRect(ctx, x, yCanvas, COLORS.SNAKE_BODY, pixelSize);
        }
        
        // Draw snake head
        const [hx, hyGame] = gameState.snake_positions[gameState.snake_positions.length - 1];
        const yCanvas = (gridSize - 1) - hyGame;  // Flip Y-axis for head too
        drawRect(ctx, hx, yCanvas, COLORS.SNAKE_HEAD, pixelSize);
    }

    // Draw apple
    if (gameState.apple_position) {
        const [ax, ayGame] = gameState.apple_position;
        const yCanvas = (gridSize - 1) - ayGame;  // Flip Y-axis for apple too
        drawRect(ctx, ax, yCanvas, COLORS.APPLE, pixelSize);
    }
} 