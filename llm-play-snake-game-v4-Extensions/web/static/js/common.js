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