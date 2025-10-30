// Main JavaScript for Curva.io
let currentData = null;
let currentChart = null;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const contextInput = document.getElementById('contextInput');
const processBtn = document.getElementById('processBtn');
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const chartContainer = document.getElementById('chartContainer');
const chartType = document.getElementById('chartType');
const aiSuggestion = document.getElementById('aiSuggestion');
const suggestionContent = document.getElementById('suggestionContent');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const backToUpload = document.getElementById('backToUpload');
const toastContainer = document.getElementById('toastContainer');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    animateStats();
});

// Event Listeners
function setupEventListeners() {
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Process button
    processBtn.addEventListener('click', processData);
    
    // Chart type change
    chartType.addEventListener('change', () => {
        if (currentData) {
            updateVisualization(chartType.value);
        }
    });
    
    // Export buttons
    document.querySelectorAll('.export-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const format = e.currentTarget.dataset.format;
            exportChart(format);
        });
    });
    
    // Back to upload
    backToUpload.addEventListener('click', () => {
        resetToUpload();
    });
    
    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const validTypes = ['text/csv', 'application/vnd.ms-excel', 
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    
    const fileExt = file.name.split('.').pop().toLowerCase();
    const validExtensions = ['csv', 'xlsx', 'xls'];
    
    if (!validExtensions.includes(fileExt)) {
        showToast('Por favor selecciona un archivo CSV o Excel', 'error');
        return;
    }
    
    // Update UI to show file name
    uploadArea.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-file-alt"></i>
        </div>
        <h3>Archivo seleccionado</h3>
        <p>${file.name}</p>
        <span class="file-types">Tamaño: ${formatFileSize(file.size)}</span>
    `;
    
    // Enable process button
    processBtn.disabled = false;
    
    // Store file for processing
    window.selectedFile = file;
}

// Process Data
async function processData() {
    if (!window.selectedFile) {
        showToast('Por favor selecciona un archivo primero', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', window.selectedFile);
    formData.append('context', contextInput.value);
    
    showLoading('Analizando tus datos con IA...');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentData = JSON.parse(result.data);
            
            // Update step indicators
            updateStepIndicator(2);
            
            // Switch to visualization view
            step1.classList.add('hidden');
            step2.classList.remove('hidden');
            
            // Populate chart types
            populateChartTypes(result.visualization_types, result.ai_suggestion.visualization_type);
            
            // Show AI suggestion
            if (result.ai_suggestion) {
                showAISuggestion(result.ai_suggestion);
            }
            
            // Display initial chart
            if (result.initial_chart) {
                displayChart(result.initial_chart);
            }
            
            showToast('Datos procesados exitosamente', 'success');
        } else {
            showToast(result.error || 'Error al procesar los datos', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error al conectar con el servidor', 'error');
    } finally {
        hideLoading();
    }
}

// Update Visualization
async function updateVisualization(vizType) {
    if (!currentData) return;
    
    showLoading('Generando visualización...');
    
    try {
        const response = await fetch('/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data: currentData,
                type: vizType
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayChart(result.chart);
        } else {
            showToast('Error al crear la visualización', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error al generar la visualización', 'error');
    } finally {
        hideLoading();
    }
}

// Display Chart
function displayChart(chartJson) {
    if (!chartJson) return;
    
    currentChart = JSON.parse(chartJson);
    
    // Clear container
    chartContainer.innerHTML = '';
    
    // Create Plotly chart
    Plotly.newPlot('chartContainer', currentChart.data, currentChart.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'curva_io_chart',
            height: 800,
            width: 1200,
            scale: 1
        }
    });
}

// Export Chart
async function exportChart(format) {
    if (!currentChart) {
        showToast('No hay gráfico para exportar', 'error');
        return;
    }
    
    showLoading('Preparando descarga...');
    
    try {
        const response = await fetch('/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                chart: JSON.stringify(currentChart),
                format: format
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Create download link
            const link = document.createElement('a');
            link.href = `data:image/${format};base64,${result.data}`;
            link.download = result.filename;
            link.click();
            
            showToast('Descarga iniciada', 'success');
            updateStepIndicator(3);
        } else {
            showToast('Error al exportar el gráfico', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error al preparar la descarga', 'error');
    } finally {
        hideLoading();
    }
}

// UI Helper Functions
function populateChartTypes(types, selected) {
    chartType.innerHTML = '';
    
    for (const [key, label] of Object.entries(types)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = label;
        if (key === selected) {
            option.selected = true;
        }
        chartType.appendChild(option);
    }
}

function showAISuggestion(suggestion) {
    let content = `
        <strong>Tipo recomendado:</strong> ${suggestion.visualization_type}<br>
        <strong>Razón:</strong> ${suggestion.reasoning}<br>
    `;
    
    if (suggestion.insights) {
        content += `<strong>Insights:</strong> ${suggestion.insights}`;
    }
    
    suggestionContent.innerHTML = content;
    aiSuggestion.classList.remove('hidden');
}

function updateStepIndicator(stepNumber) {
    document.querySelectorAll('.step').forEach((step, index) => {
        if (index < stepNumber) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

function resetToUpload() {
    // Reset UI
    step2.classList.add('hidden');
    step1.classList.remove('hidden');
    
    // Reset upload area
    uploadArea.innerHTML = `
        <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" hidden>
        <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
        </div>
        <h3>Arrastra tu archivo aquí</h3>
        <p>o haz clic para seleccionar</p>
        <span class="file-types">Soportamos CSV, XLSX, XLS</span>
    `;
    
    // Reset file input
    const newFileInput = document.getElementById('fileInput');
    newFileInput.addEventListener('change', handleFileSelect);
    
    // Clear data
    currentData = null;
    currentChart = null;
    window.selectedFile = null;
    contextInput.value = '';
    processBtn.disabled = true;
    
    // Reset step indicator
    updateStepIndicator(1);
}

// Loading Functions
function showLoading(text = 'Procesando...') {
    loadingText.textContent = text;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';
    
    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Animate Stats
function animateStats() {
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const statNumbers = entry.target.querySelectorAll('.stat-number');
                statNumbers.forEach(stat => {
                    const target = stat.textContent;
                    if (target.includes('%')) {
                        animateValue(stat, 0, 100, 2000, '%');
                    } else if (target.includes('+')) {
                        animateValue(stat, 0, parseInt(target), 2000, '+');
                    }
                });
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    const statsSection = document.querySelector('.stats');
    if (statsSection) {
        observer.observe(statsSection);
    }
}

function animateValue(element, start, end, duration, suffix = '') {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            element.textContent = end + suffix;
            clearInterval(timer);
        } else {
            element.textContent = Math.round(current) + suffix;
        }
    }, 16);
}

// Initialize smooth animations on scroll
document.addEventListener('DOMContentLoaded', () => {
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.feature-card, .stat');
        
        elements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementBottom = element.getBoundingClientRect().bottom;
            
            if (elementTop < window.innerHeight && elementBottom > 0) {
                element.style.animation = 'fadeInUp 0.8s ease both';
            }
        });
    };
    
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Initial check
});
