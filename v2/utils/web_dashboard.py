"""
Web Dashboard for Mesh Network Monitoring
Simple Flask-based dashboard to monitor mesh status
"""
from flask import Flask, render_template_string, jsonify
import json
import threading
from datetime import datetime

app = Flask(__name__)

# Global state (updated by mesh controller)
mesh_state = {
    'coordinator_id': None,
    'nodes': {},
    'stats': {},
    'tasks': [],
    'last_update': None
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pluginfer Mesh Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .nodes-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .nodes-section h2 {
            margin-bottom: 20px;
            color: #333;
        }
        .node {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .node-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .node-id {
            font-weight: bold;
            font-size: 1.1em;
        }
        .node-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .status-online {
            background: #d4edda;
            color: #155724;
        }
        .status-offline {
            background: #f8d7da;
            color: #721c24;
        }
        .node-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .detail {
            font-size: 0.9em;
            color: #666;
        }
        .detail strong {
            color: #333;
            display: block;
            margin-bottom: 5px;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.2s;
        }
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(102, 126, 234, 0.6);
        }
        .footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.9;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Pluginfer Mesh Dashboard</h1>
            <p>Real-time monitoring of distributed compute network</p>
            <p style="margin-top: 10px; color: #666;">
                <span id="last-update">Loading...</span>
            </p>
            <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Nodes</h3>
                <div class="value" id="total-nodes">-</div>
            </div>
            <div class="stat-card">
                <h3>Online Nodes</h3>
                <div class="value" id="online-nodes">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Tasks</h3>
                <div class="value" id="total-tasks">-</div>
            </div>
            <div class="stat-card">
                <h3>Revenue Earned</h3>
                <div class="value" id="revenue">$0.00</div>
            </div>
        </div>

        <div class="nodes-section">
            <h2>📊 Connected Nodes</h2>
            <div id="nodes-list">
                <div class="loading">Loading nodes...</div>
            </div>
        </div>

        <div class="footer">
            <p>Pluginfer Mesh Network | Auto-refreshes every 5 seconds</p>
        </div>
    </div>

    <script>
        function loadData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update stats
                    document.getElementById('total-nodes').textContent = data.stats.total_nodes || 0;
                    document.getElementById('online-nodes').textContent = data.stats.online_nodes || 0;
                    document.getElementById('total-tasks').textContent = data.stats.total_tasks || 0;
                    document.getElementById('revenue').textContent = '$' + (data.stats.revenue || 0).toFixed(2);
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = 
                        'Last updated: ' + new Date().toLocaleTimeString();
                    
                    // Update nodes list
                    const nodesList = document.getElementById('nodes-list');
                    if (data.nodes.length === 0) {
                        nodesList.innerHTML = '<div class="loading">No nodes connected</div>';
                    } else {
                        nodesList.innerHTML = data.nodes.map(node => `
                            <div class="node">
                                <div class="node-header">
                                    <div class="node-id">🖥️ ${node.node_id}</div>
                                    <div class="node-status status-${node.status}">
                                        ${node.status === 'online' ? '✅ Online' : '❌ Offline'}
                                    </div>
                                </div>
                                <div class="node-details">
                                    <div class="detail">
                                        <strong>GPU</strong>
                                        ${node.gpu_type || 'CPU Only'}
                                    </div>
                                    <div class="detail">
                                        <strong>Compute Power</strong>
                                        ${node.compute_power || 0}
                                    </div>
                                    <div class="detail">
                                        <strong>Tasks Completed</strong>
                                        ${node.tasks_completed || 0}
                                    </div>
                                    <div class="detail">
                                        <strong>Revenue</strong>
                                        $${(node.revenue || 0).toFixed(2)}
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    }
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                });
        }

        // Load data on page load
        loadData();
        
        // Auto-refresh every 5 seconds
        setInterval(loadData, 5000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """API endpoint for mesh status"""
    return jsonify({
        'stats': mesh_state.get('stats', {}),
        'nodes': list(mesh_state.get('nodes', {}).values()),
        'last_update': mesh_state.get('last_update')
    })

def update_mesh_state(coordinator):
    """Update mesh state from coordinator"""
    try:
        stats = coordinator.get_mesh_stats()
        
        mesh_state['coordinator_id'] = coordinator.node_id
        mesh_state['stats'] = {
            'total_nodes': len(coordinator.nodes),
            'online_nodes': sum(1 for n in coordinator.nodes.values() if n.get('status') == 'online'),
            'total_tasks': coordinator.tasks_completed,
            'revenue': coordinator.revenue_earned
        }
        
        mesh_state['nodes'] = {
            node_id: {
                'node_id': node_id,
                'status': node.get('status', 'online'),
                'gpu_type': node.get('capabilities', {}).get('gpu_type', 'CPU'),
                'compute_power': node.get('compute_power', 0),
                'tasks_completed': node.get('tasks_completed', 0),
                'revenue': node.get('revenue', 0)
            }
            for node_id, node in coordinator.nodes.items()
        }
        
        mesh_state['last_update'] = datetime.now().isoformat()
        
    except Exception as e:
        print(f"Error updating mesh state: {e}")

def start_dashboard(coordinator, host='0.0.0.0', port=5000):
    """
    Start web dashboard.
    
    Args:
        coordinator: MeshNetworkController instance
        host: Dashboard host
        port: Dashboard port
    """
    # Update state periodically
    def update_loop():
        while True:
            update_mesh_state(coordinator)
            import time
            time.sleep(5)
    
    # Start update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    print(f"\n🌐 Dashboard running at: http://{host}:{port}")
    print(f"   Open this URL in your browser to monitor the mesh\n")
    
    # Start Flask app
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    print("Web Dashboard Demo")
    print("="*70)
    print("\nTo use with mesh coordinator:")
    print("""
from core import MeshNetworkController
from utils.web_dashboard import start_dashboard

coordinator = MeshNetworkController('0.0.0.0', 9999, 'coordinator')
coordinator.start()

# Start dashboard
start_dashboard(coordinator, host='0.0.0.0', port=5000)
    """)
