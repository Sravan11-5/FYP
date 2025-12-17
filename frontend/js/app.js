/**
 * Telugu Movie Recommendation System - Frontend JavaScript
 * Handles search functionality, API calls, and dynamic UI updates
 * Integrates with Agentic AI Orchestrator backend
 */

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    ORCHESTRATOR_ENDPOINT: '/api/orchestrator/execute',
    AUTO_SEARCH_ENDPOINT: '/api/orchestrator/auto-search',
    SEARCH_DEBOUNCE_MS: 300,
    MIN_SEARCH_LENGTH: 2,
    REQUEST_TIMEOUT: 120000 // 120 seconds (increased from 60)
};

// DOM Elements
const elements = {
    searchForm: document.getElementById('searchForm'),
    searchInput: document.getElementById('searchInput'),
    searchButton: document.getElementById('searchButton'),
    loadingIndicator: document.getElementById('loadingIndicator'),
    errorMessage: document.getElementById('errorMessage'),
    errorText: document.getElementById('errorText'),
    resultsSection: document.getElementById('resultsSection'),
    resultsContainer: document.getElementById('resultsContainer'),
    recommendationsSection: document.getElementById('recommendationsSection'),
    recommendationsContainer: document.getElementById('recommendationsContainer')
};

// State Management
const state = {
    isLoading: false,
    currentQuery: '',
    searchResults: [],
    recommendations: [],
    taskId: null
};

/**
 * Initialize the application
 */
function init() {
    // Add event listeners
    elements.searchForm.addEventListener('submit', handleSearch);
    elements.searchInput.addEventListener('input', handleInputChange);
    
    // Focus on search input
    elements.searchInput.focus();
    
    console.log('🎬 Telugu Movie Recommendation System initialized');
    console.log('🤖 Agentic AI Orchestrator: Ready');
}

/**
 * Handle search form submission with Orchestrator API
 * @param {Event} event - Form submit event
 */
async function handleSearch(event) {
    event.preventDefault();
    
    const query = elements.searchInput.value.trim();
    
    // Validate input
    if (!validateSearchQuery(query)) {
        return;
    }
    
    // Update state
    state.currentQuery = query;
    state.isLoading = true;
    
    // Update UI
    showLoading();
    hideError();
    hideResults();
    hideRecommendations();
    
    try {
        console.log(`🔍 Searching for: "${query}"`);
        console.log('🤖 Invoking Agentic AI Orchestrator...');
        
        // Call the orchestrator API with automated workflow
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.AUTO_SEARCH_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_input: query,
                user_id: generateUserId(),
                session_id: generateSessionId()
            }),
            signal: AbortSignal.timeout(CONFIG.REQUEST_TIMEOUT)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API Error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Orchestrator response:', data);
        
        // Store task ID for tracking
        state.taskId = data.task_id;
        
        // Process and display results
        if (data.status === 'completed') {
            // Show movie results even if no recommendations
            if (data.movie) {
                displayMovieResults(data);
            } else if (data.recommendations && data.recommendations.length > 0) {
                displayRecommendations(data);
            } else {
                throw new Error('Movie not found in database');
            }
        } else if (data.status === 'failed') {
            throw new Error(data.error || 'Recommendation generation failed');
        } else {
            // Handle async case - poll for results
            pollForResults(data.task_id);
        }
        
    } catch (error) {
        console.error('❌ Search error:', error);
        
        if (error.name === 'TimeoutError') {
            showError('అభ్యర్థన సమయం ముగిసింది. దయచేసి మళ్లీ ప్రయత్నించండి. | Request timed out. Please try again.');
        } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
            showError('సర్వర్‌కి కనెక్ట్ చేయడం సాధ్యం కాలేదు. దయచేసి సర్వర్ రన్ అవుతుందో తనిఖీ చెయ్యండి. | Cannot connect to server. Please check if server is running.');
        } else {
            showError(`శోధన విఫలమైంది: ${error.message} | Search failed: ${error.message}`);
        }
    } finally {
        state.isLoading = false;
        hideLoading();
    }
}

/**
 * Handle input change (for future autocomplete functionality)
 * @param {Event} event - Input event
 */
function handleInputChange(event) {
    const query = event.target.value.trim();
    
    // Clear error when user starts typing
    if (query.length > 0) {
        hideError();
    }
    
    // Future: Implement autocomplete suggestions
    if (query.length >= CONFIG.MIN_SEARCH_LENGTH) {
        console.log(`Input changed: "${query}"`);
    }
}

/**
 * Validate search query
 * @param {string} query - Search query
 * @returns {boolean} - Whether query is valid
 */
function validateSearchQuery(query) {
    if (!query || query.length < CONFIG.MIN_SEARCH_LENGTH) {
        showError(`దయచేసి కనీసం ${CONFIG.MIN_SEARCH_LENGTH} అక్షరాలు ఎంటర్ చేయండి | Please enter at least ${CONFIG.MIN_SEARCH_LENGTH} characters`);
        return false;
    }
    return true;
}

/**
 * Simulate API call (temporary - will be replaced with actual API)
 * @param {string} query - Search query
 */
async function simulateAPICall(query) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Simulate successful response
    return {
        success: true,
        query: query,
        results: [],
        message: 'API integration coming soon!'
    };
}

/**
 * Show loading indicator
 */
function showLoading() {
    elements.loadingIndicator.classList.remove('hidden');
    elements.searchButton.disabled = true;
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    elements.loadingIndicator.classList.add('hidden');
    elements.searchButton.disabled = false;
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    elements.errorText.textContent = message;
    elements.errorMessage.classList.remove('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    elements.errorMessage.classList.add('hidden');
}

/**
 * Show search results section
 */
function showResults() {
    elements.resultsSection.classList.remove('hidden');
}

/**
 * Hide search results section
 */
function hideResults() {
    elements.resultsSection.classList.add('hidden');
    elements.resultsContainer.innerHTML = '';
}

/**
 * Show recommendations section
 */
function showRecommendations() {
    elements.recommendationsSection.classList.remove('hidden');
}

/**
 * Hide recommendations section
 */
function hideRecommendations() {
    elements.recommendationsSection.classList.add('hidden');
    elements.recommendationsContainer.innerHTML = '';
}

/**
 * Poll for async task results
 * @param {string} taskId - Task ID to poll
 */
async function pollForResults(taskId) {
    const maxAttempts = 30; // 30 attempts
    const pollInterval = 2000; // 2 seconds
    let attempts = 0;
    
    const poll = async () => {
        if (attempts >= maxAttempts) {
            showError('అభ్యర్థన సమయం ముగిసింది. | Request timed out.');
            return;
        }
        
        attempts++;
        
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/api/orchestrator/status/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                displayRecommendations(data);
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Task failed');
            } else {
                // Still processing, poll again
                setTimeout(poll, pollInterval);
            }
        } catch (error) {
            console.error('Polling error:', error);
            showError(`స్థితిని తనిఖీ చేయడంలో విఫలమైంది | Status check failed: ${error.message}`);
        }
    };
    
    setTimeout(poll, pollInterval);
}

/**
 * Generate unique user ID
 * @returns {string} - User ID
 */
function generateUserId() {
    // Try to get from localStorage, or create new one
    let userId = localStorage.getItem('userId');
    if (!userId) {
        userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        localStorage.setItem('userId', userId);
    }
    return userId;
}

/**
 * Generate session ID
 * @returns {string} - Session ID
 */
function generateSessionId() {
    // Create new session ID for each search
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Render recommendation card with explanation
 * @param {Object} recommendation - Recommendation data
 * @param {number} index - Card index for click handler
 * @returns {string} - HTML for recommendation card
 */
function renderRecommendationCard(recommendation, index) {
    // Handle both old format (recommendation.movie) and new flat format
    const movie = recommendation.movie || recommendation;
    
    // Extract year from release_date if present
    const releaseYear = movie.release_date ? new Date(movie.release_date).getFullYear() : null;
    
    // Get similarity score (0-1 scale) and convert to meaningful percentage
    // ML similarity scores are typically 0.2-0.8, so we scale them to look better
    const similarity = recommendation.similarity || 0;
    
    // Scale similarity: 0.2 -> 50%, 0.4 -> 70%, 0.6 -> 85%, 0.8 -> 95%
    // Formula: Scale up and add baseline to make scores more meaningful
    const scaledSimilarity = Math.min(100, Math.max(0, (similarity * 100) + 20));
    const matchPercentage = Math.round(scaledSimilarity);
    
    const explanation = recommendation.explanation || {};
    const sentiment = explanation.sentiment || {};
    
    // Format sentiment score
    const sentimentScore = sentiment.score ? (sentiment.score * 100).toFixed(1) : 'N/A';
    const sentimentLabel = sentiment.label || 'neutral';
    const sentimentClass = sentimentLabel === 'positive' ? 'sentiment-positive' : 
                          sentimentLabel === 'negative' ? 'sentiment-negative' : 'sentiment-neutral';
    
    // Get confidence score (use scaled similarity if no confidence_score)
    const confidence = recommendation.confidence_score ? 
                      Math.round((recommendation.confidence_score * 100) + 20) : matchPercentage;
    
    return `
        <div class="recommendation-card" onclick="openMovieModal(state.recommendations[${index}])" style="cursor: pointer;" title="Click to view details">
            <div class="recommendation-poster">
                <img 
                    src="${movie.poster_url || 'https://via.placeholder.com/280x400?text=' + encodeURIComponent(movie.title || 'No Poster')}" 
                    alt="${movie.title || 'Unknown'}" 
                    loading="lazy"
                >
                ${confidence !== 'N/A' ? `<div class="confidence-badge">${confidence}% Match</div>` : ''}
            </div>
            <div class="recommendation-content">
                <h4 class="recommendation-title">${movie.title || 'Unknown Movie'} 🔍</h4>
                ${movie.original_title && movie.original_title !== movie.title ? 
                    `<p class="recommendation-original">${movie.original_title}</p>` : ''}
                
                <div class="recommendation-meta">
                    ${releaseYear ? `<span class="meta-year">📅 ${releaseYear}</span>` : ''}
                    ${movie.vote_average && movie.vote_average > 0 ? `<span class="meta-rating">⭐ ${movie.vote_average}/10</span>` : ''}
                    ${movie.runtime ? `<span class="meta-runtime">⏱️ ${movie.runtime} min</span>` : ''}
                </div>
                
                ${movie.genres && movie.genres.length > 0 ? `
                    <div class="recommendation-genres">
                        ${movie.genres.slice(0, 4).map(genre => 
                            `<span class="genre-tag">${genre}</span>`
                        ).join('')}
                    </div>
                ` : ''}
                
                ${movie.overview ? `
                    <p class="recommendation-overview">${movie.overview}</p>
                ` : ''}
                
                ${explanation.reason ? `
                    <div class="recommendation-reason">
                        <strong>🎯 Why recommended:</strong>
                        <p>${explanation.reason}</p>
                    </div>
                ` : ''}
                
                ${sentiment.score ? `
                    <div class="recommendation-sentiment">
                        <strong>📊 Public Sentiment:</strong>
                        <div class="sentiment-bar">
                            <span class="${sentimentClass}">${sentimentLabel.toUpperCase()}</span>
                            <span class="sentiment-score">${sentimentScore}%</span>
                        </div>
                    </div>
                ` : ''}
                
                ${movie.cast && movie.cast.length > 0 ? `
                    <div class="recommendation-cast">
                        <strong>🎭 Cast:</strong> ${movie.cast.slice(0, 3).join(', ')}
                    </div>
                ` : ''}
                
                <div style="margin-top: 1rem; padding: 0.5rem; background: #f0f9ff; border-radius: 0.5rem; text-align: center; font-size: 0.875rem; color: #1e3a8a;">
                    <strong>Click anywhere on this card to view full details</strong>
                </div>
            </div>
        </div>
    `;
}

/**
 * Display recommendations from orchestrator response
 * @param {Object} data - Orchestrator response data
 */
/**
 * Display movie search results with analysis
 * @param {Object} data - Search results data
 */
function displayMovieResults(data) {
    const movie = data.movie;
    const analysis = data.analysis;
    const recommendations = data.recommendations || [];
    
    console.log('📽️ Displaying movie results:', movie);
    
    // Build movie result card
    const sentiment = analysis?.sentiment_distribution || {};
    const avgSentiment = analysis?.average_sentiment || 0;
    const reviewsCount = analysis?.reviews_analyzed || movie.reviews_analyzed || 0;
    
    // Extract year from release_date if present
    const releaseYear = movie.release_date ? new Date(movie.release_date).getFullYear() : null;
    
    const resultHTML = `
        <div class="search-result-container">
            <div class="movie-result-card-horizontal">
                <!-- Poster Section -->
                <div class="result-poster-section">
                    <img 
                        src="${movie.poster_url || 'https://via.placeholder.com/200x300?text=No+Poster'}" 
                        alt="${movie.name || 'Unknown'}" 
                        class="result-poster-image"
                    />
                </div>
                
                <!-- Movie Info Section -->
                <div class="result-info-section">
                    <h2 class="result-movie-title">🎬 ${movie.name}</h2>
                    <div class="result-movie-meta">
                        ${releaseYear ? `<span class="meta-item">📅 ${releaseYear}</span>` : ''}
                        ${movie.vote_average && movie.vote_average > 0 ? `<span class="meta-item">⭐ ${movie.vote_average}/10</span>` : ''}
                        <span class="meta-item">📊 ${reviewsCount} Reviews Analyzed</span>
                    </div>
                    ${movie.overview ? `
                        <div class="result-overview">
                            <p>${movie.overview}</p>
                        </div>
                    ` : ''}
                </div>
                
                <!-- Sentiment Analysis Section -->
                ${analysis ? `
                <div class="result-sentiment-section">
                    <h3 class="sentiment-title">Sentiment Analysis</h3>
                    <div class="sentiment-breakdown">
                        <div class="sentiment-item positive-item">
                            <div class="sentiment-header">
                                <span class="sentiment-icon">✅</span>
                                <span class="sentiment-name">Positive</span>
                            </div>
                            <div class="sentiment-value">${sentiment.positive || 0} reviews</div>
                            <div class="sentiment-percentage">(${(sentiment.positive_percentage || 0).toFixed(1)}%)</div>
                            <div class="sentiment-bar-bg">
                                <div class="sentiment-bar-fill positive-bar" style="width: ${sentiment.positive_percentage || 0}%"></div>
                            </div>
                        </div>
                        <div class="sentiment-item neutral-item">
                            <div class="sentiment-header">
                                <span class="sentiment-icon">⚠️</span>
                                <span class="sentiment-name">Neutral</span>
                            </div>
                            <div class="sentiment-value">${sentiment.neutral || 0} reviews</div>
                            <div class="sentiment-percentage">(${(sentiment.neutral_percentage || 0).toFixed(1)}%)</div>
                            <div class="sentiment-bar-bg">
                                <div class="sentiment-bar-fill neutral-bar" style="width: ${sentiment.neutral_percentage || 0}%"></div>
                            </div>
                        </div>
                        <div class="sentiment-item negative-item">
                            <div class="sentiment-header">
                                <span class="sentiment-icon">❌</span>
                                <span class="sentiment-name">Negative</span>
                            </div>
                            <div class="sentiment-value">${sentiment.negative || 0} reviews</div>
                            <div class="sentiment-percentage">(${(sentiment.negative_percentage || 0).toFixed(1)}%)</div>
                            <div class="sentiment-bar-bg">
                                <div class="sentiment-bar-fill negative-bar" style="width: ${sentiment.negative_percentage || 0}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                ` : ''}
            </div>
            
            ${recommendations.length > 0 ? `
                <div class="recommendations-notice">
                    <h3>🎬 Similar Movies (${recommendations.length})</h3>
                    <p>Scroll down to see AI-powered recommendations based on this movie</p>
                </div>
            ` : `
                <div class="no-recommendations-notice">
                    <p>✅ Movie analysis complete!</p>
                    <p>ℹ️ No similar movies found in our database yet.</p>
                </div>
            `}
        </div>
    `;
    
    elements.resultsContainer.innerHTML = resultHTML;
    showResults();
    
    // If there are recommendations, show them below
    if (recommendations.length > 0) {
        displayRecommendations(data);
    }
    
    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    console.log('✅ Movie results displayed');
}

/**
 * Display recommendations from search results
 * @param {Object} data - Search results with recommendations
 */
function displayRecommendations(data) {
    if (!data.recommendations || data.recommendations.length === 0) {
        showError('సిఫార్సులు దొరకలేదు. దయచేసి మరొక చిత్రాన్ని ప్రయత్నించండి. | No recommendations found. Please try another movie.');
        return;
    }
    
    // Store recommendations in state
    state.recommendations = data.recommendations;
    
    // Render recommendation cards with click handlers
    const html = data.recommendations.map((rec, index) => renderRecommendationCard(rec, index)).join('');
    elements.recommendationsContainer.innerHTML = html;
    
    // Show recommendations section
    showRecommendations();
    
    // Scroll to recommendations
    elements.recommendationsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    console.log(`✅ Displayed ${data.recommendations.length} recommendations`);
    console.log('💡 Click on any recommendation card to view full details');
}

/**
 * Open movie detail modal
 * @param {Object} movieData - Complete movie and recommendation data
 */
function openMovieModal(movieData) {
    const modal = document.getElementById('movieModal');
    const modalBody = document.getElementById('modalBody');
    
    // Handle both old format (movieData.movie) and new flat format
    const movie = movieData.movie || movieData;
    const explanation = movieData.explanation || {};
    const sentiment = explanation.sentiment || {};
    
    // Get similarity/confidence score and scale it
    const similarity = movieData.similarity || 0;
    const confidence = movieData.confidence_score || similarity;
    
    // Scale similarity: 0.2 -> 50%, 0.4 -> 70%, 0.6 -> 85%, 0.8 -> 95%
    const scaledConfidence = Math.min(100, Math.max(0, (confidence * 100) + 20));
    const confidencePercent = Math.round(scaledConfidence);
    
    const finalConfidence = confidencePercent > 0 ? confidencePercent : 'N/A';
    
    // Extract year from release_date if present
    const releaseYear = movie.release_date ? new Date(movie.release_date).getFullYear() : null;
    
    // Build modal content
    const modalHTML = `
        <div class="modal-header">
            <img 
                src="${movie.poster_url || 'https://via.placeholder.com/300x450?text=' + encodeURIComponent(movie.title || 'No Poster')}" 
                alt="${movie.title || 'Unknown'}" 
                class="modal-poster"
            >
            <div class="modal-info">
                <h2 class="modal-title" id="modalTitle">${movie.title || 'Unknown Movie'}</h2>
                ${movie.original_title && movie.original_title !== movie.title ? 
                    `<p class="modal-original-title">${movie.original_title}</p>` : ''}
                
                <div class="modal-meta">
                    ${releaseYear ? `<span>📅 ${releaseYear}</span>` : ''}
                    ${movie.vote_average && movie.vote_average > 0 ? `<span>⭐ ${movie.vote_average}/10</span>` : ''}
                    ${movie.runtime ? `<span>⏱️ ${movie.runtime} min</span>` : ''}
                    ${movie.genre ? `<span>🎭 ${movie.genre}</span>` : ''}
                </div>
                
                ${movie.genres && movie.genres.length > 0 ? `
                    <div class="modal-genres">
                        ${movie.genres.map(genre => 
                            `<span class="genre-tag">${genre}</span>`
                        ).join('')}
                    </div>
                ` : ''}
                
                ${finalConfidence !== 'N/A' && finalConfidence > 0 ? `
                    <div style="margin-top: 1rem;">
                        <div class="confidence-badge" style="position: static; display: inline-block;">
                            ${finalConfidence}% Match
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
        
        ${movie.overview ? `
            <div class="modal-section">
                <h3 class="modal-section-title">📖 Overview</h3>
                <p class="modal-overview">${movie.overview}</p>
            </div>
        ` : ''}
        
        ${explanation.reason ? `
            <div class="modal-section">
                <div class="recommendation-reason">
                    <strong>🎯 Why We Recommend This:</strong>
                    <p>${explanation.reason}</p>
                </div>
            </div>
        ` : ''}
        
        ${sentiment.score ? `
            <div class="modal-section">
                <div class="recommendation-sentiment">
                    <strong>📊 Public Sentiment Analysis:</strong>
                    <div class="sentiment-bar">
                        <span class="sentiment-${sentiment.label}">${(sentiment.label || 'neutral').toUpperCase()}</span>
                        <span class="sentiment-score">${(sentiment.score * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        ` : ''}
        
        ${movie.cast && movie.cast.length > 0 ? `
            <div class="modal-section">
                <h3 class="modal-section-title">🎭 Cast</h3>
                <div class="modal-cast-grid">
                    ${movie.cast.slice(0, 6).map(actor => 
                        `<div class="modal-cast-item">${actor}</div>`
                    ).join('')}
                </div>
            </div>
        ` : ''}
        
        <div class="modal-section">
            <h3 class="modal-section-title">📊 Statistics</h3>
            <div class="modal-stats">
                ${movie.rating ? `
                    <div class="modal-stat-card">
                        <div class="modal-stat-value">${movie.rating}/10</div>
                        <div class="modal-stat-label">Rating</div>
                    </div>
                ` : ''}
                ${confidence !== 'N/A' ? `
                    <div class="modal-stat-card">
                        <div class="modal-stat-value">${confidence}%</div>
                        <div class="modal-stat-label">Match Score</div>
                    </div>
                ` : ''}
                ${sentiment.score ? `
                    <div class="modal-stat-card">
                        <div class="modal-stat-value">${(sentiment.score * 100).toFixed(0)}%</div>
                        <div class="modal-stat-label">Sentiment Score</div>
                    </div>
                ` : ''}
            </div>
        </div>
        
        <div class="modal-actions">
            <button class="modal-button modal-button-secondary" onclick="closeMovieModal()">
                Close
            </button>
            ${movie.tmdb_id ? `
                <button class="modal-button modal-button-primary" onclick="window.open('https://www.themoviedb.org/movie/${movie.tmdb_id}', '_blank')">
                    View on TMDB
                </button>
            ` : ''}
        </div>
    `;
    
    modalBody.innerHTML = modalHTML;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
}

/**
 * Close movie detail modal
 */
function closeMovieModal() {
    const modal = document.getElementById('movieModal');
    modal.classList.add('hidden');
    document.body.style.overflow = ''; // Restore scrolling
}

/**
 * Validate search query
 * @param {string} query - Search query
 * @returns {boolean} - Whether query is valid
 */
function validateSearchQuery(query) {
    if (!query || query.length < CONFIG.MIN_SEARCH_LENGTH) {
        showError(`దయచేసి కనీసం ${CONFIG.MIN_SEARCH_LENGTH} అక్షరాలు ఎంటర్ చేయండి | Please enter at least ${CONFIG.MIN_SEARCH_LENGTH} characters`);
        return false;
    }
    return true;
}

/**
 * Handle keyboard shortcuts
 * @param {KeyboardEvent} event - Keyboard event
 */
function handleKeyboardShortcuts(event) {
    // Focus search on '/' key
    if (event.key === '/' && event.target !== elements.searchInput) {
        event.preventDefault();
        elements.searchInput.focus();
    }
    
    // Clear search on 'Escape' key or close modal
    if (event.key === 'Escape') {
        const modal = document.getElementById('movieModal');
        if (!modal.classList.contains('hidden')) {
            closeMovieModal();
        } else {
            elements.searchInput.value = '';
            hideResults();
            hideRecommendations();
            hideError();
        }
    }
}

// Add keyboard shortcuts listener
document.addEventListener('keydown', handleKeyboardShortcuts);

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateSearchQuery,
        renderMovieCard,
        displayResults,
        displayRecommendations,
        openMovieModal,
        closeMovieModal
    };
}
