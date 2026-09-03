function fmt(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
}

function pct(part, total) {
    if (!total) return '';
    return Math.round((part / total) * 100) + '%';
}

function updateDashboard(d) {
    document.getElementById('total').textContent = fmt(d.total_requests);
    document.getElementById('local').textContent = fmt(d.local_requests);
    document.getElementById('cloud').textContent = fmt(d.cloud_requests);
    document.getElementById('hybrid').textContent = fmt(d.hybrid_requests);
    document.getElementById('saved').textContent = fmt(d.tokens_saved);
    document.getElementById('cost').textContent = '$' + d.total_cost_usd.toFixed(4);
    document.getElementById('roi').textContent = d.roi_percent.toFixed(0) + '%';
    document.getElementById('avg-time').textContent = Math.round(d.avg_response_time_ms) + 'ms';

    document.getElementById('local-pct').textContent = pct(d.local_requests, d.total_requests);
    document.getElementById('cloud-pct').textContent = pct(d.cloud_requests, d.total_requests);

    var total = d.total_requests || 1;
    document.getElementById('bar-local').style.width = (d.local_requests / total * 100) + '%';
    document.getElementById('bar-hybrid').style.width = (d.hybrid_requests / total * 100) + '%';
    document.getElementById('bar-cloud').style.width = (d.cloud_requests / total * 100) + '%';

    var delegation = Math.round(d.delegation_rate * 100);
    document.getElementById('delegation-fill').style.width = delegation + '%';
    document.getElementById('delegation-pct').textContent = delegation + '%';
}

fetch('/health').then(function(r) { return r.json(); }).then(function(d) {
    var el = document.getElementById('status');
    el.textContent = d.status;
    el.className = 'status ' + d.status;
});

fetch('/api/dashboard').then(function(r) { return r.json(); }).then(updateDashboard);
