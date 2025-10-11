// Global state
let isCalibrating = false;
let isMonitoring = false;
let calibrationAngles = [];
let calibrationPhase = null;
let calibrationTimer = null;
let postureChart = null;
let statisticsInterval = null;

// DOM Elements
const startCalibrationBtn = document.getElementById('startCalibrationBtn');
const calibrationInstructions = document.getElementById('calibrationInstructions');
const calibrateGoodBtn = document.getElementById('calibrateGoodBtn');
const calibrateBadBtn = document.getElementById('calibrateBadBtn');
const startMonitoringBtn = document.getElementById('startMonitoringBtn');
const stopMonitoringBtn = document.getElementById('stopMonitoringBtn');
const resetBtn = document.getElementById('resetBtn');
const systemStatus = document.getElementById('systemStatus');
const videoOverlay = document.getElementById('videoOverlay');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeChart();
    updateStatistics();
    
    // Remove overlay after video loads
    setTimeout(() => {
        videoOverlay.classList.add('hidden');
    }, 2000);
    
    // Event listeners
    startCalibrationBtn.addEventListener('click', startCalibration);
    calibrateGoodBtn.addEventListener('click', () => startPhaseCalibration('good'));
    calibrateBadBtn.addEventListener('click', () => startPhaseCalibration('bad'));
    startMonitoringBtn.addEventListener('click', startMonitoring);
    stopMonitoringBtn.addEventListener('click', stopMonitoring);
    resetBtn.addEventListener('click', resetSystem);
});

// Initialize Chart
function initializeChart() {
    const ctx = document.getElementById('postureChart').getContext('2d');
    postureChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Good Posture', 'Poor Posture'],
            datasets: [{
                data: [0, 0],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(16, 185, 129, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        padding: 20,
                        font: {
                            size: 14,
                            family: 'Inter'
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return `${label}: ${value.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Update statistics
async function updateStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const data = await response.json();
        
        // Update status badge
        if (data.is_monitoring) {
            systemStatus.className = 'status-badge monitoring';
            systemStatus.querySelector('.status-text').textContent = 'Monitoring Active';
        } else if (data.is_calibrated) {
            systemStatus.className = 'status-badge calibrated';
            systemStatus.querySelector('.status-text').textContent = 'Calibrated';
        } else {
            systemStatus.className = 'status-badge';
            systemStatus.querySelector('.status-text').textContent = 'Not Calibrated';
        }
        
        // Update current angle
        document.getElementById('currentAngle').textContent = 
            data.current_angle ? data.current_angle.toFixed(1) : '--';
        
        // Update statistics
        document.getElementById('goodPosturePercent').textContent = 
            `${data.good_percentage.toFixed(1)}%`;
        document.getElementById('badPosturePercent').textContent = 
            `${data.bad_percentage.toFixed(1)}%`;
        document.getElementById('slouchCount').textContent = data.slouch_count;
        document.getElementById('thresholdAngle').textContent = 
            data.threshold ? `${data.threshold.toFixed(1)}°` : '--°';
        
        // Update time displays
        document.getElementById('goodPostureTime').textContent = 
            formatTime(data.good_posture_time);
        document.getElementById('badPostureTime').textContent = 
            formatTime(data.bad_posture_time);
        document.getElementById('sessionTimer').textContent = 
            formatTime(data.total_time);
        
        // Update chart
        postureChart.data.datasets[0].data = [
            data.good_percentage,
            data.bad_percentage
        ];
        postureChart.update('none');
        
        // Update button states
        startMonitoringBtn.disabled = !data.is_calibrated || data.is_monitoring;
        stopMonitoringBtn.disabled = !data.is_monitoring;
        
    } catch (error) {
        console.error('Error updating statistics:', error);
    }
    
    // Schedule next update
    setTimeout(updateStatistics, 500);
}

// Format time in MM:SS
function formatTime(seconds) {
    if (!seconds || seconds < 0) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Start calibration process
async function startCalibration() {
    try {
        const response = await fetch('/api/start_calibration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isCalibrating = true;
            calibrationInstructions.classList.remove('hidden');
            startCalibrationBtn.disabled = true;
            calibrateGoodBtn.disabled = false;
            startMonitoringBtn.disabled = true;
            
            showNotification('Calibration started! Follow the instructions.', 'info');
        }
    } catch (error) {
        console.error('Error starting calibration:', error);
        showNotification('Failed to start calibration', 'error');
    }
}

// Start phase calibration
async function startPhaseCalibration(phase) {
    const isGoodPhase = phase === 'good';
    const button = isGoodPhase ? calibrateGoodBtn : calibrateBadBtn;
    const countdownElement = document.getElementById(isGoodPhase ? 'goodCountdown' : 'badCountdown');
    
    button.disabled = true;
    calibrationAngles = [];
    
    let countdown = 5;
    countdownElement.textContent = `(${countdown}s)`;
    
    // Collect angles for 5 seconds
    const angleCollectionInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/current_angle');
            const data = await response.json();
            if (data.angle) {
                calibrationAngles.push(data.angle);
            }
        } catch (error) {
            console.error('Error collecting angle:', error);
        }
    }, 200);
    
    // Countdown timer
    calibrationTimer = setInterval(() => {
        countdown--;
        countdownElement.textContent = `(${countdown}s)`;
        
        if (countdown <= 0) {
            clearInterval(calibrationTimer);
            clearInterval(angleCollectionInterval);
            countdownElement.textContent = '';
            finishPhaseCalibration(phase);
        }
    }, 1000);
}

// Finish phase calibration
async function finishPhaseCalibration(phase) {
    const isGoodPhase = phase === 'good';
    const endpoint = isGoodPhase ? '/api/calibrate_good' : '/api/calibrate_bad';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ angles: calibrationAngles })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            if (isGoodPhase) {
                showNotification(`Good posture calibrated: ${data.angle.toFixed(1)}°`, 'success');
                calibrateBadBtn.disabled = false;
                document.getElementById('step2').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                showNotification('Calibration complete!', 'success');
                isCalibrating = false;
                calibrationInstructions.classList.add('hidden');
                startCalibrationBtn.disabled = false;
                startMonitoringBtn.disabled = false;
                
                // Reset buttons
                calibrateGoodBtn.disabled = true;
                calibrateBadBtn.disabled = true;
            }
        }
    } catch (error) {
        console.error('Error finishing calibration:', error);
        showNotification('Calibration failed', 'error');
        
        // Re-enable button for retry
        if (isGoodPhase) {
            calibrateGoodBtn.disabled = false;
        } else {
            calibrateBadBtn.disabled = false;
        }
    }
}

// Start monitoring
async function startMonitoring() {
    try {
        const response = await fetch('/api/start_monitoring', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isMonitoring = true;
            startMonitoringBtn.disabled = true;
            stopMonitoringBtn.disabled = false;
            startCalibrationBtn.disabled = true;
            
            showNotification('Monitoring started!', 'success');
        }
    } catch (error) {
        console.error('Error starting monitoring:', error);
        showNotification('Failed to start monitoring', 'error');
    }
}

// Stop monitoring
async function stopMonitoring() {
    try {
        const response = await fetch('/api/stop_monitoring', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isMonitoring = false;
            startMonitoringBtn.disabled = false;
            stopMonitoringBtn.disabled = true;
            startCalibrationBtn.disabled = false;
            
            showNotification('Monitoring stopped', 'info');
        }
    } catch (error) {
        console.error('Error stopping monitoring:', error);
        showNotification('Failed to stop monitoring', 'error');
    }
}

// Reset system
async function resetSystem() {
    if (!confirm('Are you sure you want to reset? All calibration data and statistics will be lost.')) {
        return;
    }
    
    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isCalibrating = false;
            isMonitoring = false;
            calibrationInstructions.classList.add('hidden');
            
            startCalibrationBtn.disabled = false;
            calibrateGoodBtn.disabled = true;
            calibrateBadBtn.disabled = true;
            startMonitoringBtn.disabled = true;
            stopMonitoringBtn.disabled = true;
            
            // Reset chart
            postureChart.data.datasets[0].data = [0, 0];
            postureChart.update();
            
            showNotification('System reset complete', 'info');
        }
    } catch (error) {
        console.error('Error resetting system:', error);
        showNotification('Failed to reset system', 'error');
    }
}

// Show notification (simple implementation)
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        font-weight: 500;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
