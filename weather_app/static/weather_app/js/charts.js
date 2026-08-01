document.addEventListener("DOMContentLoaded", function() {
    const labelsStr = document.getElementById('chart-labels').textContent;
    const dataStr = document.getElementById('chart-data').textContent;
    
    if (labelsStr && dataStr) {
        const labels = JSON.parse(labelsStr);
        const data = JSON.parse(dataStr);
        
        const ctx = document.getElementById('tempChart').getContext('2d');
        
        // Determine line color based on theme
        const isCloudy = document.body.classList.contains('theme-cloudy');
        const color = isCloudy ? 'rgba(50, 50, 50, 1)' : 'rgba(255, 255, 255, 1)';
        const bgColor = isCloudy ? 'rgba(50, 50, 50, 0.2)' : 'rgba(255, 255, 255, 0.2)';
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Temp (°C)',
                    data: data,
                    borderColor: color,
                    backgroundColor: bgColor,
                    borderWidth: 3,
                    tension: 0.4, // smooth curves
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: color }
                    }
                },
                scales: {
                    x: {
                        grid: { color: isCloudy ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)' },
                        ticks: { color: color }
                    },
                    y: {
                        grid: { color: isCloudy ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)' },
                        ticks: { color: color }
                    }
                }
            }
        });
    }
});
