// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global variables
let selectedStock = null;
let currentStockData = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Update threshold displays
    updateThresholdDisplays();
    
    // Add event listeners
    document.getElementById('positiveThreshold').addEventListener('input', updateThresholdDisplays);
    document.getElementById('negativeThreshold').addEventListener('input', updateThresholdDisplays);
    
    // Check API status
    checkApiStatus();
    
    // Add search functionality
    document.getElementById('stockSearch').addEventListener('input', handleSearchInput);
});

// Update threshold value displays
function updateThresholdDisplays() {
    const positiveValue = document.getElementById('positiveThreshold').value;
    const negativeValue = document.getElementById('negativeThreshold').value;
    
    document.getElementById('positiveValue').textContent = positiveValue;
    document.getElementById('negativeValue').textContent = negativeValue;
}

// Check API connection status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
            document.getElementById('apiStatus').innerHTML = '<span class="badge bg-success">Connected</span>';
        } else {
            document.getElementById('apiStatus').innerHTML = '<span class="badge bg-warning">API Error</span>';
        }
    } catch (error) {
        document.getElementById('apiStatus').innerHTML = '<span class="badge bg-danger">Disconnected</span>';
    }
}

// Handle search input
let searchTimeout;
function handleSearchInput() {
    clearTimeout(searchTimeout);
    const query = document.getElementById('stockSearch').value;
    
    if (query.length < 2) {
        hideSearchResults();
        return;
    }
    
    searchTimeout = setTimeout(() => {
        searchStocks(query);
    }, 300);
}

// Search for stocks
async function searchStocks(query = null) {
    const searchQuery = query || document.getElementById('stockSearch').value;
    
    if (!searchQuery.trim()) {
        showAlert('Please enter a search term', 'warning');
        return;
    }
    
    try {
        showLoading();
        const response = await fetch(`${API_BASE_URL}/stocks/search?query=${encodeURIComponent(searchQuery)}`);
        
        if (response.ok) {
            const data = await response.json();
            displaySearchResults(data.stocks);
        } else {
            throw new Error('Search failed');
        }
    } catch (error) {
        showAlert('Error searching stocks: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Display search results
function displaySearchResults(stocks) {
    const resultsContainer = document.getElementById('searchResults');
    
    if (stocks.length === 0) {
        resultsContainer.innerHTML = '<div class="search-result-item text-muted">No stocks found</div>';
    } else {
        resultsContainer.innerHTML = stocks.map(stock => 
            `<div class="search-result-item" onclick="selectStock('${stock}')">${stock}</div>`
        ).join('');
    }
    
    resultsContainer.style.display = 'block';
}

// Hide search results
function hideSearchResults() {
    document.getElementById('searchResults').style.display = 'none';
}

// Select a stock
async function selectStock(symbol) {
    selectedStock = symbol;
    document.getElementById('selectedSymbol').textContent = symbol;
    document.getElementById('selectedStock').style.display = 'block';
    document.getElementById('stockSearch').value = symbol;
    hideSearchResults();
    
    // Fetch stock data
    await fetchStockData(symbol);
}

// Clear stock selection
function clearSelection() {
    selectedStock = null;
    currentStockData = null;
    document.getElementById('selectedStock').style.display = 'none';
    document.getElementById('stockSearch').value = '';
    document.getElementById('stockInfo').innerHTML = '<p class="text-muted">Select a stock to view information</p>';
    document.getElementById('analysisResults').style.display = 'none';
}

// Fetch stock data
async function fetchStockData(symbol) {
    try {
        showLoading();
        const response = await fetch(`${API_BASE_URL}/stocks/${symbol}/data`);
        
        if (response.ok) {
            const data = await response.json();
            currentStockData = data;
            displayStockInfo(data);
        } else {
            throw new Error('Failed to fetch stock data');
        }
    } catch (error) {
        showAlert('Error fetching stock data: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Display stock information
function displayStockInfo(data) {
    const stockInfo = document.getElementById('stockInfo');
    
    const priceChange = data.price_change;
    const changePercent = data.price_change_percent;
    const changeClass = priceChange >= 0 ? 'text-success' : 'text-danger';
    const changeIcon = priceChange >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
    
    stockInfo.innerHTML = `
        <div class="row text-center">
            <div class="col-6">
                <h6>Current Price</h6>
                <h4 class="text-primary">$${data.current_price.toFixed(2)}</h4>
            </div>
            <div class="col-6">
                <h6>Change</h6>
                <h5 class="${changeClass}">
                    <i class="fas ${changeIcon}"></i> $${priceChange.toFixed(2)} (${changePercent.toFixed(2)}%)
                </h5>
            </div>
        </div>
        <hr>
        <div class="row text-center">
            <div class="col-6">
                <h6>Reference Price</h6>
                <p class="mb-0">$${data.reference_price.toFixed(2)}</p>
            </div>
            <div class="col-6">
                <h6>Volume</h6>
                <p class="mb-0">${data.volume.toLocaleString()}</p>
            </div>
        </div>
    `;
}

// Analyze selected stock
async function analyzeStock() {
    if (!selectedStock) {
        showAlert('Please select a stock first', 'warning');
        return;
    }
    
    const positiveThreshold = parseFloat(document.getElementById('positiveThreshold').value);
    const negativeThreshold = parseFloat(document.getElementById('negativeThreshold').value);
    const enableTrading = document.getElementById('enableTrading').checked;
    
    try {
        showLoading();
        
        const response = await fetch(`${API_BASE_URL}/stocks/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: selectedStock,
                positive_threshold: positiveThreshold,
                negative_threshold: negativeThreshold,
                enable_trading: enableTrading
            })
        });
        
        if (response.ok) {
            const analysis = await response.json();
            displayAnalysisResults(analysis);
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
    } catch (error) {
        showAlert('Error analyzing stock: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Analyze custom news
async function analyzeCustomNews() {
    const symbol = document.getElementById('customStockSymbol').value.trim().toUpperCase();
    const newsText = document.getElementById('customNewsText').value.trim();
    
    if (!symbol || !newsText) {
        showAlert('Please enter both stock symbol and news text', 'warning');
        return;
    }
    
    const positiveThreshold = parseFloat(document.getElementById('positiveThreshold').value);
    const negativeThreshold = parseFloat(document.getElementById('negativeThreshold').value);
    const enableTrading = document.getElementById('enableTrading').checked;
    
    try {
        showLoading();
        
        const response = await fetch(`${API_BASE_URL}/news/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                news_text: newsText,
                positive_threshold: positiveThreshold,
                negative_threshold: negativeThreshold,
                enable_trading: enableTrading
            })
        });
        
        if (response.ok) {
            const analysis = await response.json();
            displayAnalysisResults(analysis);
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'News analysis failed');
        }
    } catch (error) {
        showAlert('Error analyzing news: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Display analysis results
function displayAnalysisResults(analysis) {
    // Update result displays
    document.getElementById('sentimentResult').textContent = analysis.sentiment.toUpperCase();
    document.getElementById('sentimentResult').className = `sentiment-${analysis.sentiment}`;
    
    document.getElementById('confidenceResult').textContent = (analysis.score * 100).toFixed(1) + '%';
    
    document.getElementById('decisionResult').textContent = analysis.decision;
    document.getElementById('decisionResult').className = `sentiment-${analysis.decision.toLowerCase()}`;
    
    // Display chart if available
    if (analysis.chart_data) {
        createChart(analysis.chart_data, analysis.symbol);
    }
    
    // Display news articles if available
    if (analysis.news_articles && analysis.news_articles.length > 0) {
        displayNewsArticles(analysis.news_articles);
    }
    
    // Show results section
    document.getElementById('analysisResults').style.display = 'block';
    
    // Scroll to results
    document.getElementById('analysisResults').scrollIntoView({ behavior: 'smooth' });
}

// Create candlestick chart
function createChart(chartData, symbol) {
    const candles = chartData.candles;
    const signals = chartData.signals || [];
    
    if (!candles || candles.length === 0) {
        document.getElementById('chartContainer').innerHTML = '<p class="text-muted">No chart data available</p>';
        return;
    }
    
    // Prepare data for Plotly
    const times = candles.map(c => c.timestamp);
    const opens = candles.map(c => c.open);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const closes = candles.map(c => c.close);
    
    const candlestickData = {
        x: times,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        type: 'candlestick',
        name: 'Price'
    };
    
    const traces = [candlestickData];
    
    // Add signal markers
    signals.forEach(signal => {
        const signalTime = signal.timestamp;
        const signalPrice = signal.price;
        const signalType = signal.type;
        
        // Find closest candle time
        const closestIndex = times.reduce((closest, time, index) => {
            const currentDiff = Math.abs(new Date(time) - new Date(signalTime));
            const closestDiff = Math.abs(new Date(times[closest]) - new Date(signalTime));
            return currentDiff < closestDiff ? index : closest;
        }, 0);
        
        traces.push({
            x: [times[closestIndex]],
            y: [signalPrice],
            mode: 'markers+text',
            marker: {
                symbol: signalType === 'BUY' ? 'triangle-up' : 'triangle-down',
                color: signalType === 'BUY' ? 'green' : 'red',
                size: 15
            },
            text: [signalType],
            textposition: signalType === 'BUY' ? 'top center' : 'bottom center',
            name: signalType,
            showlegend: false
        });
    });
    
    const layout = {
        title: `${symbol} Stock Chart with Trading Signals`,
        xaxis: { title: 'Time' },
        yaxis: { title: 'Price ($)' },
        height: 500,
        showlegend: false
    };
    
    Plotly.newPlot('chartContainer', traces, layout, {responsive: true});
}

// Display news articles
function displayNewsArticles(articles) {
    const newsSection = document.getElementById('newsSection');
    const newsArticles = document.getElementById('newsArticles');
    
    newsArticles.innerHTML = articles.map(article => `
        <div class="news-item">
            <h6><a href="${article.url}" target="_blank" class="text-decoration-none">${article.title}</a></h6>
            <p class="text-muted mb-1">${article.description}</p>
            <small class="text-muted">
                <i class="fas fa-newspaper"></i> ${article.source} | 
                <i class="fas fa-clock"></i> ${new Date(article.publishedAt).toLocaleDateString()}
            </small>
        </div>
    `).join('');
    
    newsSection.style.display = 'block';
}

// Show loading modal
function showLoading() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

// Hide loading modal
function hideLoading() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Close search results when clicking outside
document.addEventListener('click', function(event) {
    const searchContainer = document.querySelector('.search-container');
    const searchResults = document.getElementById('searchResults');
    
    if (!searchContainer.contains(event.target)) {
        searchResults.style.display = 'none';
    }
});
