// Agent Recommendation System - Frontend JavaScript

let eventSource = null;

document.addEventListener('DOMContentLoaded', function() {
    const recommendBtn = document.getElementById('recommend-btn');
    const goalInput = document.getElementById('goal-input');
    
    recommendBtn.addEventListener('click', handleRecommend);
    goalInput.addEventListener('keypress', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            handleRecommend();
        }
    });
});

function handleRecommend() {
    const goal = document.getElementById('goal-input').value.trim();
    const topK = parseInt(document.getElementById('top-k-input').value) || 5;
    
    if (!goal) {
        alert('Please enter a goal');
        return;
    }
    
    // Hide previous results/errors
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('error-section').style.display = 'none';
    
    // Show progress
    document.getElementById('progress-section').style.display = 'block';
    
    // Reset progress
    updateProgress(0, 'Initializing...');
    resetSteps();
    
    // Disable button
    const btn = document.getElementById('recommend-btn');
    btn.disabled = true;
    btn.textContent = 'Processing...';
    
    // Close previous event source if exists
    if (eventSource) {
        eventSource.close();
    }
    
    // Create new event source for Server-Sent Events
    eventSource = new EventSource(`/api/recommend?goal=${encodeURIComponent(goal)}&top_k=${topK}`);
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleProgressUpdate(data);
        } catch (e) {
            console.error('Error parsing event data:', e);
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('EventSource error:', error);
        // Check if connection is in a bad state
        if (eventSource.readyState === EventSource.CLOSED) {
            showError('Connection closed. The server may have encountered an error. Please check the server logs and try again.');
        } else if (eventSource.readyState === EventSource.CONNECTING) {
            // Still connecting, wait a bit more
            console.log('EventSource is still connecting...');
        } else {
            showError('Connection error. Please check that the server is running and try again.');
        }
        cleanup();
    };
}

function handleProgressUpdate(data) {
    if (data.error) {
        showError(data.message || 'An error occurred');
        cleanup();
        return;
    }
    
    // Update progress bar
    updateProgress(data.progress || 0, data.message || '');
    
    // Update active step
    updateStep(data.step);
    
    // Handle completion
    if (data.step === 'complete' && data.result) {
        displayResults(data.result);
        cleanup();
    }
    
    // Handle goal spec summary
    if (data.goal_spec) {
        displayGoalSummary(data.goal_spec);
    }
}

function updateProgress(percentage, message) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const statusMessage = document.getElementById('status-message');
    
    progressBar.style.width = percentage + '%';
    progressText.textContent = Math.round(percentage) + '%';
    statusMessage.textContent = message;
}

function resetSteps() {
    const steps = document.querySelectorAll('.step');
    steps.forEach(step => {
        step.classList.remove('active', 'completed');
    });
}

function updateStep(stepName) {
    // Mark all previous steps as completed
    const stepOrder = ['init', 'goal_spec', 'retrieval', 'ranking', 'explanation'];
    const currentIndex = stepOrder.indexOf(stepName);
    
    stepOrder.forEach((step, index) => {
        const stepElement = document.getElementById(`step-${step}`);
        if (stepElement) {
            if (index < currentIndex) {
                stepElement.classList.remove('active');
                stepElement.classList.add('completed');
                stepElement.querySelector('.step-icon').textContent = '✓';
            } else if (index === currentIndex) {
                stepElement.classList.add('active');
                stepElement.classList.remove('completed');
            } else {
                stepElement.classList.remove('active', 'completed');
            }
        }
    });
}

function displayGoalSummary(goalSpec) {
    const summaryDiv = document.getElementById('goal-summary');
    const contentDiv = document.getElementById('goal-summary-content');
    
    contentDiv.innerHTML = '';
    
    if (goalSpec.capabilities && goalSpec.capabilities.length > 0) {
        const capabilitiesDiv = document.createElement('div');
        capabilitiesDiv.style.width = '100%';
        capabilitiesDiv.innerHTML = '<strong>Capabilities:</strong><br>';
        goalSpec.capabilities.forEach(cap => {
            const tag = document.createElement('span');
            tag.className = 'capability-tag';
            tag.textContent = cap;
            capabilitiesDiv.appendChild(tag);
            capabilitiesDiv.appendChild(document.createTextNode(' '));
        });
        contentDiv.appendChild(capabilitiesDiv);
    }
    
    if (goalSpec.keywords && goalSpec.keywords.length > 0) {
        const keywordsDiv = document.createElement('div');
        keywordsDiv.style.width = '100%';
        keywordsDiv.style.marginTop = '10px';
        keywordsDiv.innerHTML = '<strong>Keywords:</strong><br>';
        goalSpec.keywords.forEach(keyword => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.textContent = keyword;
            keywordsDiv.appendChild(tag);
            keywordsDiv.appendChild(document.createTextNode(' '));
        });
        contentDiv.appendChild(keywordsDiv);
    }
    
    summaryDiv.style.display = 'block';
}

function displayResults(result) {
    // Hide progress
    document.getElementById('progress-section').style.display = 'none';
    
    // Show results
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    
    // Display overall explanation
    if (result.explanation) {
        document.getElementById('explanation-text').textContent = result.explanation;
        document.getElementById('overall-explanation').style.display = 'block';
    }
    
    // Display goal summary if not already shown
    if (result.goal_spec) {
        displayGoalSummary(result.goal_spec);
    }
    
    // Display recommendations
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = '';
    
    if (!result.recommendations || result.recommendations.length === 0) {
        recommendationsList.innerHTML = '<p>No recommendations found.</p>';
        return;
    }
    
    result.recommendations.forEach((rec, index) => {
        const card = createRecommendationCard(rec, index + 1);
        recommendationsList.appendChild(card);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createRecommendationCard(rec, index) {
    const card = document.createElement('div');
    card.className = 'recommendation-card';
    
    const header = document.createElement('div');
    header.className = 'recommendation-header';
    
    const title = document.createElement('div');
    title.className = 'recommendation-title';
    title.textContent = `${index}. ${rec.agent.name || 'Unknown Agent'}`;
    
    const scores = document.createElement('div');
    scores.className = 'recommendation-scores';
    
    // Similarity score
    if (rec.similarity_score !== undefined) {
        const simBadge = document.createElement('span');
        simBadge.className = 'score-badge score-similarity';
        simBadge.textContent = `Similarity: ${(rec.similarity_score * 100).toFixed(1)}%`;
        scores.appendChild(simBadge);
    }
    
    // Compatibility score
    if (rec.compatibility_score !== undefined) {
        const compBadge = document.createElement('span');
        compBadge.className = 'score-badge score-compatibility';
        compBadge.textContent = `Compatibility: ${(rec.compatibility_score * 100).toFixed(1)}%`;
        scores.appendChild(compBadge);
    }
    
    // Reuse difficulty
    if (rec.reuse_difficulty) {
        const diffBadge = document.createElement('span');
        diffBadge.className = `score-difficulty difficulty-${rec.reuse_difficulty}`;
        diffBadge.textContent = `Difficulty: ${rec.reuse_difficulty}`;
        scores.appendChild(diffBadge);
    }
    
    header.appendChild(title);
    header.appendChild(scores);
    
    card.appendChild(header);
    
    // Description
    if (rec.agent.description) {
        const desc = document.createElement('div');
        desc.className = 'recommendation-description';
        desc.textContent = rec.agent.description;
        card.appendChild(desc);
    }
    
    // Explanation
    if (rec.explanation) {
        const explanation = document.createElement('div');
        explanation.className = 'recommendation-explanation';
        explanation.textContent = rec.explanation;
        card.appendChild(explanation);
    }
    
    // Capabilities
    if (rec.agent.capabilities && rec.agent.capabilities.length > 0) {
        const capsDiv = document.createElement('div');
        capsDiv.className = 'recommendation-capabilities';
        capsDiv.innerHTML = '<strong>Capabilities: </strong>';
        rec.agent.capabilities.forEach(cap => {
            const tag = document.createElement('span');
            tag.className = 'capability-tag';
            tag.textContent = cap;
            capsDiv.appendChild(tag);
            capsDiv.appendChild(document.createTextNode(' '));
        });
        card.appendChild(capsDiv);
    }
    
    // Required modifications
    if (rec.required_modifications && rec.required_modifications.length > 0) {
        const modDiv = document.createElement('div');
        modDiv.innerHTML = '<strong>Required Modifications:</strong>';
        const modList = document.createElement('ul');
        modList.className = 'modifications-list';
        rec.required_modifications.forEach(mod => {
            const li = document.createElement('li');
            li.textContent = mod;
            modList.appendChild(li);
        });
        modDiv.appendChild(modList);
        card.appendChild(modDiv);
    }
    
    return card;
}

function showError(message) {
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
    
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function cleanup() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    const btn = document.getElementById('recommend-btn');
    btn.disabled = false;
    btn.textContent = 'Get Recommendations';
}

