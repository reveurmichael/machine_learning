// COLORS is defined in common.js; no need to redeclare here.

// Fresh MVC Architecture for Replay Mode
// Completely self-contained system with no legacy dependencies

// ===== MODEL =====
class ReplayModel {
    constructor() {
        this.state = null;
        this.listeners = [];
        this.isPolling = false;
        this.pollInterval = null;
    }
    
    addListener(callback) {
        this.listeners.push(callback);
    }
    
    notifyListeners() {
        this.listeners.forEach(callback => callback(this.state));
    }
    
    async fetchState() {
        try {
            const response = await fetch('/api/state');
            const data = await response.json();
            
            if (data.error) {
                console.error('API Error:', data.error);
                return false;
            }
            
            this.state = data;
            this.notifyListeners();
            return true;
        } catch (error) {
            console.error('Network Error:', error);
            return false;
        }
    }
    
    async sendCommand(command) {
        try {
            const response = await fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: command })
            });
            
            const data = await response.json();
            
            // Handle API response format: {status: 'ok'|'error', message: '...'}
            if (data.status === 'error') {
                console.error('Command Error:', data.message || data.error || 'Unknown error');
                return false;
            }
            
            // Refresh state after command
            await this.fetchState();
            return true;
        } catch (error) {
            console.error('Command Network Error:', error);
            return false;
        }
    }
    
    startPolling() {
        if (this.isPolling) return;
        
        this.isPolling = true;
        this.pollInterval = setInterval(() => {
            this.fetchState();
        }, 100);
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        this.isPolling = false;
    }
}

// ===== VIEW =====
class ReplayView {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas?.getContext('2d');
        this.elements = this.initializeElements();
        this.showGameContainer();
        this.showReplaySections();
    }
    
    initializeElements() {
        return {
            // Game info
            gameNumberElement: document.getElementById('game-number'),
            scoreElement: document.getElementById('score'),
            endReasonElement: document.getElementById('end-reason'),
            endReasonContainer: document.getElementById('end-reason-container'),
            
            // Sections containers
            progressSection: document.getElementById('progress-section'),
            llmInfoSection: document.getElementById('llm-info-section'),
            replayControlsSection: document.getElementById('replay-controls-section'),
            
            // Progress
            progressElement: document.getElementById('progress'),
            progressBar: document.getElementById('progress-bar'),
            
            // Status indicators
            pausedIndicator: document.getElementById('paused-indicator'),
            gameOverIndicator: document.getElementById('game-over-indicator'),
            
            // Controls
            playPauseButton: document.getElementById('play-pause'),
            prevGameButton: document.getElementById('prev-game'),
            nextGameButton: document.getElementById('next-game'),
            restartButton: document.getElementById('restart'),
            speedUpButton: document.getElementById('speed-up'),
            speedDownButton: document.getElementById('speed-down'),
            speedValueElement: document.getElementById('speed-value'),
            
            // LLM info
            primaryLlmElement: document.getElementById('primary-llm'),
            secondaryLlmElement: document.getElementById('secondary-llm')
        };
    }
    
    update(state) {
        if (!state) return;
        
        // Update game information
        this.updateGameInfo(state);
        
        // Update progress
        this.updateProgress(state);
        
        // Update status indicators
        this.updateStatusIndicators(state);
        
        // Update controls
        this.updateControls(state);
        
        // Update LLM information
        this.updateLlmInfo(state);
        
        // Update document title
        document.title = `Snake Game ${state.game_number || 1} - Score: ${state.score || 0}`;
        
        // Redraw game canvas
        this.drawGame(state);
    }
    
    updateGameInfo(state) {
        // Game number
        if (this.elements.gameNumberElement) {
            if (state.total_games && state.total_games > 0) {
                this.elements.gameNumberElement.textContent = `${state.game_number}/${state.total_games}`;
            } else {
                this.elements.gameNumberElement.textContent = state.game_number;
            }
        }
        // Score - display in "X/Y" format if max_score is available
        if (this.elements.scoreElement) {
            if (state.max_score && state.max_score > 0) {
                this.elements.scoreElement.textContent = `${state.score}/${state.max_score}`;
            } else {
                this.elements.scoreElement.textContent = state.score;
            }
        }
        // End Reason
        if (this.elements.endReasonElement && this.elements.endReasonContainer) {
            // Always show the end reason container
            this.elements.endReasonContainer.style.display = 'block';
            // Set the text content to the state value or fallback to "-"
            const reason = state.game_end_reason || state.end_reason || '-';
            this.elements.endReasonElement.textContent = reason;
        }
    }
    
    updateProgress(state) {
        // Progress text
        if (this.elements.progressElement) {
            this.elements.progressElement.textContent = 
                `${state.move_index || 0}/${state.total_moves || 0}`;
        }
        
        // Progress bar
        if (this.elements.progressBar && state.total_moves > 0) {
            const progressPercent = ((state.move_index || 0) / state.total_moves) * 100;
            this.elements.progressBar.style.width = `${progressPercent}%`;
        }
    }
    
    updateStatusIndicators(state) {
        if (!this.elements.pausedIndicator || !this.elements.gameOverIndicator) return;
        
        // Hide both indicators initially
        this.elements.pausedIndicator.style.display = 'none';
        this.elements.gameOverIndicator.style.display = 'none';
        
        // Check if game is over
        const isGameOver = state.game_over;
        
        if (isGameOver) {
            this.elements.gameOverIndicator.style.display = 'block';
        } else if (state.paused) {
            this.elements.pausedIndicator.style.display = 'block';
        }
    }
    
    updateControls(state) {
        // Play/pause button
        if (this.elements.playPauseButton) {
            this.elements.playPauseButton.textContent = state.paused ? 'Play' : 'Pause';
        }
        
        // Speed display
        const pause = state.pause_between_moves ?? state.move_pause;
        if (pause) {
            this.elements.speedValueElement.textContent = `${pause.toFixed(1)}s`;
        }
    }
    
    updateLlmInfo(state) {
        // Primary LLM
        if (this.elements.primaryLlmElement && state.primary_llm) {
            this.elements.primaryLlmElement.textContent = state.primary_llm;
        }
        
        // Secondary LLM
        if (this.elements.secondaryLlmElement) {
            const secondaryLlm = state.parser_llm || state.secondary_llm || 'None';
            this.elements.secondaryLlmElement.textContent = secondaryLlm;
        }
    }
    
    drawGame(state) {
        if (!this.canvas || !this.ctx || !state) return;
        
        const gridSize = state.grid_size || 10;
        const maxSize = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.7);
        const pixelSize = Math.floor(maxSize / gridSize);
        
        // Canvas sizing
        this.canvas.width = pixelSize * gridSize;
        this.canvas.height = pixelSize * gridSize;
        
        // === DEBUG: Log color info ===
        if (state.colors) {
            console.log('[ReplayView] Received colors from backend:', state.colors);
        } else {
            console.warn('[ReplayView] No colors field in state!');
        }
        // Update colors if provided, with strict mapping
        if (state.colors) {
            const colorKeys = ['snake_head', 'snake_body', 'apple', 'background', 'grid'];
            colorKeys.forEach(key => {
                if (!(key in state.colors)) {
                    console.warn(`[ReplayView] Missing color key from backend: ${key}`);
                }
            });
            COLORS.SNAKE_HEAD = rgbArrayToHex(state.colors.snake_head) || COLORS.SNAKE_HEAD;
            COLORS.SNAKE_BODY = rgbArrayToHex(state.colors.snake_body) || COLORS.SNAKE_BODY;
            COLORS.APPLE = rgbArrayToHex(state.colors.apple) || COLORS.APPLE;
            COLORS.BACKGROUND = rgbArrayToHex(state.colors.background) || COLORS.BACKGROUND;
            COLORS.GRID = rgbArrayToHex(state.colors.grid) || COLORS.GRID;
        }
        // === END DEBUG ===
        
        // Clear background
        this.ctx.fillStyle = COLORS.BACKGROUND;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        drawGrid(this.ctx, gridSize, pixelSize);
        
        // Draw snake
        if (Array.isArray(state.snake_positions) && state.snake_positions.length) {
            // Body
            for (let i = 0; i < state.snake_positions.length - 1; i++) {
                const [x, yGame] = state.snake_positions[i];
                drawRect(this.ctx, x, (gridSize - 1) - yGame, COLORS.SNAKE_BODY, pixelSize);
            }
            // Head
            const [hx, hyGame] = state.snake_positions[state.snake_positions.length - 1];
            drawRect(this.ctx, hx, (gridSize - 1) - hyGame, COLORS.SNAKE_HEAD, pixelSize);
        }
        
        // Draw apple
        if (Array.isArray(state.apple_position) && state.apple_position.length === 2) {
            const [ax, ayGame] = state.apple_position;
            drawRect(this.ctx, ax, (gridSize - 1) - ayGame, COLORS.APPLE, pixelSize);
        }
    }
    
    showGameContainer() {
        const loadingMessage = document.getElementById('loading-message');
        const gameContainer = document.getElementById('game-container');
        
        if (loadingMessage) loadingMessage.style.display = 'none';
        if (gameContainer) gameContainer.style.display = 'flex';
    }
    
    showError(message) {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.innerHTML = `<p class="error-message">${message}</p>`;
        }
    }
    
    setButtonLoading(button, loading) {
        if (!button) return;
        
        button.disabled = loading;
        button.style.opacity = loading ? '0.6' : '1';
    }
    
    setAllButtonsLoading(loading) {
        const buttons = [
            this.elements.playPauseButton,
            this.elements.prevGameButton,
            this.elements.nextGameButton,
            this.elements.restartButton,
            this.elements.speedUpButton,
            this.elements.speedDownButton
        ];
        
        buttons.forEach(button => this.setButtonLoading(button, loading));
    }
    
    showReplaySections() {
        const safeShow = (el) => { if (el) el.style.display = 'block'; };
        safeShow(this.elements.progressSection);
        safeShow(this.elements.llmInfoSection);
        safeShow(this.elements.replayControlsSection);
    }
}

// ===== CONTROLLER =====
class ReplayController {
    constructor(model, view) {
        this.model = model;
        this.view = view;
        this.isLoading = false;
        
        // Bind model updates to view
        this.model.addListener((state) => this.view.update(state));
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Button controls
        const safeAdd = (el, evt, fn) => { if (el) el.addEventListener(evt, fn); };
        
        safeAdd(this.view.elements.playPauseButton, 'click', () => this.togglePlayPause());
        safeAdd(this.view.elements.prevGameButton, 'click', () => this.navigateGame('prev'));
        safeAdd(this.view.elements.nextGameButton, 'click', () => this.navigateGame('next'));
        safeAdd(this.view.elements.restartButton, 'click', () => this.restartGame());
        safeAdd(this.view.elements.speedUpButton, 'click', () => this.speedUp());
        safeAdd(this.view.elements.speedDownButton, 'click', () => this.speedDown());
        
        // Keyboard controls
        document.addEventListener('keydown', (event) => this.handleKeyDown(event));
        
        // Window resize
        window.addEventListener('resize', () => {
            if (this.model.state) {
                this.view.drawGame(this.model.state);
            }
        });
    }
    
    async start() {
        // Start polling for state updates
        this.model.startPolling();
        
        // Initial state fetch
        await this.model.fetchState();
    }
    
    async togglePlayPause() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.view.setAllButtonsLoading(true);
        
        const command = this.model.state?.paused ? 'play' : 'pause';
        await this.model.sendCommand(command);
        
        this.isLoading = false;
        this.view.setAllButtonsLoading(false);
    }
    
    async navigateGame(direction) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.view.setAllButtonsLoading(true);
        
        const command = direction === 'next' ? 'next_game' : 'prev_game';
        await this.model.sendCommand(command);
        
        this.isLoading = false;
        this.view.setAllButtonsLoading(false);
    }
    
    async restartGame() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.view.setAllButtonsLoading(true);
        
        await this.model.sendCommand('restart_game');
        
        this.isLoading = false;
        this.view.setAllButtonsLoading(false);
    }
    
    async speedUp() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.view.setAllButtonsLoading(true);
        
        await this.model.sendCommand('speed_up');
        
        this.isLoading = false;
        this.view.setAllButtonsLoading(false);
    }
    
    async speedDown() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.view.setAllButtonsLoading(true);
        
        await this.model.sendCommand('speed_down');
        
        this.isLoading = false;
        this.view.setAllButtonsLoading(false);
    }
    
    handleKeyDown(event) {
        // Prevent default behavior for all arrow keys and space
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' '].includes(event.key)) {
            event.preventDefault();
        }
        
        switch (event.key) {
            case ' ':
                this.togglePlayPause();
                break;
            case 'ArrowLeft':
                this.navigateGame('prev');
                break;
            case 'ArrowRight':
                this.navigateGame('next');
                break;
            case 'r':
            case 'R':
                this.restartGame();
                break;
            case 'ArrowUp':
                this.speedUp();
                break;
            case 'ArrowDown':
                this.speedDown();
                break;
        }
    }
    
    stop() {
        this.model.stopPolling();
    }
}

// ===== APPLICATION =====
class ReplayApplication {
    constructor() {
        this.model = new ReplayModel();
        this.view = new ReplayView();
        this.controller = new ReplayController(this.model, this.view);
    }
    
    async start() {
        await this.controller.start();
    }
    
    stop() {
        this.controller.stop();
    }
}

// ===== GLOBAL INSTANCE =====
let replayApp = null;

// ===== INITIALIZATION =====
function init() {
    replayApp = new ReplayApplication();
    replayApp.start();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);