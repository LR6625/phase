/* ============================================================================
   MTG Card Hash Visualization - JavaScript
   ============================================================================ */

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkForNewData();
    updateServerStatus();
    // Check server status every 2 seconds
    setInterval(updateServerStatus, 2000);
});

// ==============================================================================
// Server Status
// ==============================================================================

async function updateServerStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        const statusText = document.getElementById('statusText');
        
        if (status.has_data) {
            statusText.textContent = `✓ ${status.current_card} (${status.timestamp})`;
            document.getElementById('statusIndicator').classList.add('has-data');
        } else {
            statusText.textContent = 'Waiting for data from OhMyOhMy.py...';
            document.getElementById('statusIndicator').classList.remove('has-data');
        }
    } catch (error) {
        console.log('Could not fetch server status');
    }
}

// ==============================================================================
// Event Listeners
// ==============================================================================

function initializeEventListeners() {
    const refreshBtn = document.getElementById('refreshBtn');
    const clearBtn = document.getElementById('clearBtn');

    refreshBtn.addEventListener('click', async () => {
        await refreshData();
    });

    clearBtn.addEventListener('click', () => {
        clearResults();
    });

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

// ==============================================================================
// Data Fetching
// ==============================================================================

async function refreshData() {
    showLoading(true);
    clearError();

    try {
        const response = await fetch('/api/get-latest');
        const data = await response.json();

        if (!data || !data.input_image) {
            throw new Error('No data available. Run OhMyOhMy.py to send card data.');
        }

        displayResults(data);
        updateStatus('✓ Card data loaded successfully');
    } catch (error) {
        showError(error.message);
        updateStatus('⚠ Failed to load data');
    } finally {
        showLoading(false);
    }
}

async function checkForNewData() {
    try {
        const response = await fetch('/api/has-new-data');
        const data = await response.json();
        
        if (data.has_data) {
            updateStatus('✓ Card data ready - click "Refresh Data" to display');
        }
    } catch (error) {
        console.log('Server not ready yet');
    }
}

// ==============================================================================
// Display Functions
// ==============================================================================

function displayResults(data) {
    // Update status cards
    document.getElementById('hammingDistance').textContent = data.hamming_distance || '-';
    document.getElementById('similarity').textContent = data.similarity || '-';
    document.getElementById('quality').textContent = data.quality || '-';

    // Update card info
    const cardInfo = data.card_info || {};
    document.getElementById('cardName').textContent = cardInfo.name || '-';
    document.getElementById('cardSet').textContent = cardInfo.set || '-';
    document.getElementById('cardRarity').textContent = cardInfo.rarity || '-';
    document.getElementById('orientation').textContent = data.orientation || '-';
    document.getElementById('inputHash').textContent = data.input_hash || '-';
    document.getElementById('matchedHash').textContent = data.matched_hash || '-';

    // Update images
    if (data.detected_image) {
        document.getElementById('detectedImage').src = data.detected_image;
    }
    if (data.input_image) {
        document.getElementById('inputImage').src = data.input_image;
    }

    // Update visualizations
    if (data.hash_visualization) {
        const hashViz = data.hash_visualization;
        const hashSrc = hashViz.startsWith('data:') ? hashViz : `data:image/png;base64,${hashViz}`;
        document.getElementById('hashVisualization').src = hashSrc;
    }
    if (data.comparison_visualization) {
        const compViz = data.comparison_visualization;
        const compSrc = compViz.startsWith('data:') ? compViz : `data:image/png;base64,${compViz}`;
        document.getElementById('comparisonVisualization').src = compSrc;
    }

    // Update JSON details
    const jsonData = cardInfo.json_details || cardInfo;
    document.getElementById('cardJsonDetails').textContent = 
        typeof jsonData === 'string' ? jsonData : JSON.stringify(jsonData, null, 2);

    // Show results section
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Switch to overview tab
    switchTab('overview');
}

function clearResults() {
    // Hide results
    document.getElementById('resultsSection').classList.add('hidden');
    
    // Clear images
    document.getElementById('detectedImage').src = '';
    document.getElementById('inputImage').src = '';
    document.getElementById('hashVisualization').src = '';
    document.getElementById('comparisonVisualization').src = '';
    
    // Clear text
    document.getElementById('hammingDistance').textContent = '-';
    document.getElementById('similarity').textContent = '-';
    document.getElementById('quality').textContent = '-';
    
    updateStatus('Cleared. Run OhMyOhMy.py again to send new data.');
}

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });

    // Show selected tab
    const selectedTab = document.getElementById(`${tabName}-tab`);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Add active class to selected button
    document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');
}

// ==============================================================================
// UI State Functions
// ==============================================================================

function showLoading(show) {
    const loadingState = document.getElementById('loadingState');
    if (show) {
        loadingState.classList.remove('hidden');
        document.getElementById('refreshBtn').disabled = true;
    } else {
        loadingState.classList.add('hidden');
        document.getElementById('refreshBtn').disabled = false;
    }
}

function showError(message) {
    const errorState = document.getElementById('errorState');
    document.getElementById('errorMessage').textContent = message;
    errorState.classList.remove('hidden');
}

function clearError() {
    document.getElementById('errorState').classList.add('hidden');
}

function updateStatus(message) {
    document.getElementById('statusText').textContent = message;
}

// Export for direct call
window.clearError = clearError;
