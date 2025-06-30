/**
 * Common JavaScript functions for Snake Game Web
 * Shared between human play and replay modes
 */

// Constants - colors will be overridden by values from the server
let COLORS = {
    SNAKE_HEAD: '#3498db',    // Blue for snake head
    SNAKE_BODY: '#2980b9',    // Darker blue for snake body
    APPLE: '#e74c3c',         // Red for apple
    BACKGROUND: '#2c3e50',    // Dark background
    GRID: '#34495e',          // Grid lines
};

/**
 * Sidebar Management System
 * Handles showing/hiding appropriate information panels based on game mode
 * Provides reusable functions for all game modes and future extensions
 */
class SidebarManager {
    constructor() {
        this.currentMode = null;
        this.elements = this.initializeElements();
    }
    
    /**
     * Initialize references to all sidebar elements
     */
    initializeElements() {
        return {
            // Basic game info (always present)
            scoreElement: document.getElementById('score'),
            stepsElement: document.getElementById('steps'),
            
            // Status indicators
            gameOverIndicator: document.getElementById('game-over-indicator'),
            finishedIndicator: document.getElementById('finished-indicator'),
            pausedIndicator: document.getElementById('paused-indicator'),
            
            // End reason
            endReasonContainer: document.getElementById('end-reason-container'),
            endReasonElement: document.getElementById('end-reason'),
            
            // LLM specific sections
            llmSection: document.getElementById('llm-section'),
            llmResponseElement: document.getElementById('llm-response'),
            movesSection: document.getElementById('moves-section'),
            plannedMovesElement: document.getElementById('planned-moves'),
            
            // Controls section
            controlsSection: document.getElementById('controls-section'),
            resetButton: document.getElementById('reset-button'),
            
            // Algorithm info (for heuristics extensions)
            algorithmSection: document.getElementById('algorithm-section'),
            algorithmNameElement: document.getElementById('algorithm-name'),
            pathLengthElement: document.getElementById('path-length'),
            nodesExploredElement: document.getElementById('nodes-explored'),
            
            // Training metrics (for ML/RL extensions)
            trainingSection: document.getElementById('training-section'),
            episodeElement: document.getElementById('episode'),
            rewardElement: document.getElementById('reward'),
            lossElement: document.getElementById('loss'),
            
            // Replay specific
            gameNumberElement: document.getElementById('game-number'),
            progressElement: document.getElementById('progress'),
            progressBar: document.getElementById('progress-bar'),
            primaryLlmElement: document.getElementById('primary-llm'),
            secondaryLlmElement: document.getElementById('secondary-llm'),
            
            // Replay controls
            playPauseButton: document.getElementById('play-pause'),
            prevGameButton: document.getElementById('prev-game'),
            nextGameButton: document.getElementById('next-game'),
            restartButton: document.getElementById('restart'),
            speedUpButton: document.getElementById('speed-up'),
            speedDownButton: document.getElementById('speed-down'),
            speedValueElement: document.getElementById('speed-value')
        };
    }
    
    /**
     * Configure sidebar for specific game mode
     * @param {string} mode - Game mode: 'human', 'llm', 'replay', 'heuristic', 'ml', 'rl'
     */
    setMode(mode) {
        this.currentMode = mode;
        this.hideAllSections();
        
        switch(mode) {
            case 'human':
                this.showHumanSections();
                break;
            case 'llm':
                this.showLLMSections();
                break;
            case 'replay':
                this.showReplaySections();
                break;
            case 'heuristic':
                this.showHeuristicSections();
                break;
            case 'ml':
            case 'supervised':
                this.showMLSections();
                break;
            case 'rl':
            case 'reinforcement':
                this.showRLSections();
                break;
            default:
                console.warn(`Unknown sidebar mode: ${mode}`);
        }
    }
    
    /**
     * Hide all sidebar sections
     */
    hideAllSections() {
        const sections = [
            'llm-section', 'moves-section', 'controls-section', 
            'algorithm-section', 'training-section', 'progress-section', 
            'llm-info-section', 'replay-controls-section'
        ];
        
        sections.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
        
        // Hide all indicators
        const indicators = [
            'game-over-indicator', 'finished-indicator', 'paused-indicator'
        ];
        
        indicators.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
    }
    
    /**
     * Show sections for human play mode
     */
    showHumanSections() {
        this.safeShow('controls-section');
        console.log('[Sidebar] Configured for human play mode');
    }
    
    /**
     * Show sections for LLM mode
     */
    showLLMSections() {
        this.safeShow('llm-section');
        this.safeShow('moves-section');
        console.log('[Sidebar] Configured for LLM mode');
    }
    
    /**
     * Show sections for replay mode
     */
    showReplaySections() {
        this.safeShow('progress-section');
        this.safeShow('llm-info-section');
        this.safeShow('replay-controls-section');
        console.log('[Sidebar] Configured for replay mode');
    }
    
    /**
     * Show sections for heuristic algorithms (Task 1)
     */
    showHeuristicSections() {
        this.safeShow('algorithm-section');
        console.log('[Sidebar] Configured for heuristic mode');
    }
    
    /**
     * Show sections for machine learning (Task 2, 4, 5)
     */
    showMLSections() {
        this.safeShow('training-section');
        console.log('[Sidebar] Configured for ML mode');
    }
    
    /**
     * Show sections for reinforcement learning (Task 3)
     */
    showRLSections() {
        this.safeShow('training-section');
        console.log('[Sidebar] Configured for RL mode');
    }
    
    /**
     * Update sidebar with game state data
     * @param {Object} gameState - Current game state
     */
    updateWithGameState(gameState) {
        if (!gameState) return;
        
        // Update basic game info (always present)
        this.safeSet(this.elements.scoreElement, gameState.score || 0);
        this.safeSet(this.elements.stepsElement, gameState.steps || 0);
        
        // Update status indicators
        this.updateStatusIndicators(gameState);
        
        // Update mode-specific sections
        switch(this.currentMode) {
            case 'human':
                this.updateHumanSections(gameState);
                break;
            case 'llm':
                this.updateLLMSections(gameState);
                break;
            case 'replay':
                this.updateReplaySections(gameState);
                break;
            case 'heuristic':
                this.updateHeuristicSections(gameState);
                break;
            case 'ml':
            case 'supervised':
                this.updateMLSections(gameState);
                break;
            case 'rl':
            case 'reinforcement':
                this.updateRLSections(gameState);
                break;
        }
    }
    
    /**
     * Update status indicators (game over, finished, paused)
     */
    updateStatusIndicators(gameState) {
        // Game over indicator
        if (gameState.game_over) {
            this.safeShow('game-over-indicator');
        } else {
            this.safeHide('game-over-indicator');
        }
        
        // Session finished indicator
        if (gameState.running === false && this.currentMode === 'llm') {
            this.safeShow('finished-indicator');
        } else {
            this.safeHide('finished-indicator');
        }
        
        // End reason
        if (gameState.game_end_reason || gameState.end_reason) {
            const reason = gameState.game_end_reason || gameState.end_reason;
            this.safeSet(this.elements.endReasonElement, reason);
            this.safeShow('end-reason-container');
        } else {
            this.safeHide('end-reason-container');
        }
    }
    
    /**
     * Update human play specific sections
     */
    updateHumanSections(gameState) {
        // Controls are static, no dynamic updates needed
    }
    
    /**
     * Update LLM specific sections
     */
    updateLLMSections(gameState) {
        // LLM response
        if (gameState.llm_response) {
            this.safeSet(this.elements.llmResponseElement, gameState.llm_response);
        }
        
        // Planned moves
        if (gameState.planned_moves && Array.isArray(gameState.planned_moves)) {
            const movesText = gameState.planned_moves.join(', ');
            this.safeSet(this.elements.plannedMovesElement, movesText);
        }
    }
    
    /**
     * Update replay specific sections
     */
    updateReplaySections(gameState) {
        // Game number
        if (gameState.game_number) {
            this.safeSet(this.elements.gameNumberElement, gameState.game_number);
        }
        
        // LLM info
        if (gameState.primary_llm) {
            this.safeSet(this.elements.primaryLlmElement, gameState.primary_llm);
        }
        if (gameState.parser_llm || gameState.secondary_llm) {
            this.safeSet(this.elements.secondaryLlmElement, gameState.parser_llm || gameState.secondary_llm || 'None');
        }
    }
    
    /**
     * Update heuristic algorithm sections
     */
    updateHeuristicSections(gameState) {
        if (gameState.algorithm) {
            this.safeSet(this.elements.algorithmNameElement, gameState.algorithm);
        }
        if (gameState.path_length !== undefined) {
            this.safeSet(this.elements.pathLengthElement, gameState.path_length);
        }
        if (gameState.nodes_explored !== undefined) {
            this.safeSet(this.elements.nodesExploredElement, gameState.nodes_explored);
        }
    }
    
    /**
     * Update machine learning sections
     */
    updateMLSections(gameState) {
        if (gameState.episode !== undefined) {
            this.safeSet(this.elements.episodeElement, gameState.episode);
        }
        if (gameState.reward !== undefined) {
            this.safeSet(this.elements.rewardElement, gameState.reward);
        }
        if (gameState.loss !== undefined) {
            this.safeSet(this.elements.lossElement, gameState.loss);
        }
    }
    
    /**
     * Update reinforcement learning sections
     */
    updateRLSections(gameState) {
        // RL uses same training metrics as ML
        this.updateMLSections(gameState);
    }
    
    /**
     * Safely set text content of an element
     */
    safeSet(element, value) {
        if (element) {
            element.textContent = value;
        }
    }
    
    /**
     * Safely show an element by ID
     */
    safeShow(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
        }
    }
    
    /**
     * Safely hide an element by ID
     */
    safeHide(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    }
    
    /**
     * Show error message
     */
    showErrorMessage(message) {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.innerHTML = `<p class="error-message">${message}</p>`;
        }
    }
    
    /**
     * Show game container and hide loading message
     */
    showGameContainer() {
        const loadingMessage = document.getElementById('loading-message');
        const gameContainer = document.getElementById('game-container');
        
        if (loadingMessage && loadingMessage.style.display !== 'none') {
            loadingMessage.style.display = 'none';
        }
        if (gameContainer) {
            gameContainer.style.display = 'flex';
        }
    }
}

// Global sidebar manager instance
let sidebarManager = null;

/**
 * Initialize sidebar manager (call this in each mode's init function)
 */
function initializeSidebar(mode) {
    if (!sidebarManager) {
        sidebarManager = new SidebarManager();
    }
    sidebarManager.setMode(mode);
    return sidebarManager;
}

/**
 * Converts RGB array to hex color string
 * @param {Array} rgbArray - Array of RGB values [r, g, b]
 * @returns {string} Hex color string or null if invalid input
 */
function rgbArrayToHex(rgbArray) {
    if (!Array.isArray(rgbArray) || rgbArray.length < 3) {
        return null;
    }
    return `#${rgbArray[0].toString(16).padStart(2, '0')}${rgbArray[1].toString(16).padStart(2, '0')}${rgbArray[2].toString(16).padStart(2, '0')}`;
}

/**
 * Draws a grid on a canvas context
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} gridSize - Grid size
 * @param {number} pixelSize - Size of each grid cell in pixels
 */
function drawGrid(ctx, gridSize, pixelSize) {
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

/**
 * Draws a rectangle on a canvas context
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate (already converted to canvas coordinates)
 * @param {string} color - Color of the rectangle
 * @param {number} pixelSize - Size of each grid cell in pixels
 */
function drawRect(ctx, x, y, color, pixelSize) {
    ctx.fillStyle = color;
    // Make the rectangles fill the cell with just a small gap for grid visibility
    ctx.fillRect(
        x * pixelSize + 1,
        y * pixelSize + 1,
        pixelSize - 2,
        pixelSize - 2
    );
}

/**
 * Sends an API request
 * @param {string} url - API endpoint URL
 * @param {string} method - HTTP method (GET or POST)
 * @param {Object} data - Data to send (for POST requests)
 * @returns {Promise} Promise resolving to the API response
 */
async function sendApiRequest(url, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {}
        };
        
        if (data) {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(url, options);
        return await response.json();
    } catch (error) {
        console.error(`API request error (${url}):`, error);
        throw error;
    }
} 