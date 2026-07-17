
"""
Professional Web UI - Marketplace & Dashboard
Beautiful, modern interface for Pluginfer marketplace
"""
from flask import Flask, render_template, jsonify, request, render_template_string
import json
import os
import threading
import time
from datetime import datetime

app = Flask(__name__, template_folder='templates')
# Global reference to the mesh controller (injected on startup)
mesh_controller = None

def attach_controller(controller):
    """
    Attach the mesh controller instance to the Flask app.
    This allows the UI to interact with the backend node logic.
    """
    global mesh_controller
    mesh_controller = controller
    print(f"WEB UI: Generic Controller Attached: {controller.node_id if hasattr(controller, 'node_id') else 'Unknown'}")

@app.route('/')
def home():
    """Main Dashboard"""
    return render_template('dashboard.html')

@app.route('/client')
def client_portal_route():
    """Client Portal for submitting tasks"""
    return render_template('client.html')

@app.route('/explorer')
def explorer_route():
    """Public Block Explorer"""
    return render_template('explorer.html')

@app.route('/api/stats')
def get_stats():
    """Unified API for Advanced Dashboard"""
    if not mesh_controller:
        return jsonify({'error': 'Node starting...'})
        
    try:
        # Gather Data from all Sub-Systems
        wallet_bal = 0.0
        if hasattr(mesh_controller, 'ledger') and hasattr(mesh_controller, 'wallet'):
            wallet_bal = mesh_controller.ledger.get_balance(mesh_controller.wallet.address)
            
        rep_score = 0.0
        if hasattr(mesh_controller, 'reputation'):
            rep_score = mesh_controller.reputation.get_score()
            
        role = "FOLLOWER"
        term = 0
        if hasattr(mesh_controller, 'election'):
            role = mesh_controller.election.state
            term = mesh_controller.election.term
            
        peers = []
        if hasattr(mesh_controller, 'nodes'):
            for pid, pdata in mesh_controller.nodes.items():
                peers.append({
                    'id': pid,
                    'host': pdata.get('host', 'unknown'),
                    'port': pdata.get('port', 0),
                    'reputation': 50 # Placeholder if not synced
                })
            
        # [FIX] Get Hardware Info
        hardware_info = "Unknown"
        if hasattr(mesh_controller, 'local_hardware'):
            h = mesh_controller.local_hardware
            hardware_info = f"{h.get('name', 'CPU')} ({h.get('type', 'cpu')})"

        return jsonify({
            'node_id': mesh_controller.node_id,
            'balance': round(wallet_bal, 4),
            'staked_balance': mesh_controller.staking.get_staking_info(mesh_controller.wallet.address).get('amount', '0.0') if hasattr(mesh_controller, 'staking') and mesh_controller.staking.get_staking_info(mesh_controller.wallet.address) else '0.0',
            'reputation': rep_score,
            'tasks': mesh_controller.reputation.tasks_completed if hasattr(mesh_controller, 'reputation') else 0,
            'role': role,
            'term': term,
            'peers': peers,
            'hardware': hardware_info,
            'logs': ["[SYSTEM] Node Running..."] 
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/explorer/data')
def api_explorer_data():
    """API: Get Full Chain Data"""
    if not mesh_controller:
        return jsonify({'error': 'Node not connected'}), 503
        
    # Serialize chain
    if hasattr(mesh_controller, 'ledger'):
        chain_data = [b.to_dict() for b in mesh_controller.ledger.chain]
        return jsonify({'chain': chain_data})
    return jsonify({'chain': []})

@app.route('/api/marketplace/stats')
def api_marketplace_stats():
    """API endpoint for LIVE marketplace statistics"""
    if not mesh_controller:
        return jsonify({'error': 'Node not connected'}), 503
        
    # PULL REAL DATA
    stats = mesh_controller.get_mesh_stats()
    
    # Format for UI
    return jsonify({
        'available_nodes': stats.get('total_nodes', 1),
        'tasks_completed': stats.get('tasks_completed', 0),
        'total_earned': stats.get('revenue_earned', 0.0),
        'node_id': stats.get('node_id'),
        'mode': stats.get('mode'),
        'queue_size': stats.get('queue_size', 0),
        'gpu_stats': stats.get('gpu_stats', {}),
        'nodes_list': stats.get('nodes_list', []),
        'ledger_height': mesh_controller.ledger.get_height() if hasattr(mesh_controller, 'ledger') else 0,
        'is_paused': getattr(mesh_controller, 'paused', False),
        'rep_score': mesh_controller.reputation.get_score() if hasattr(mesh_controller, 'reputation') else 0
    })

@app.route('/api/node/control', methods=['POST'])
def api_node_control():
    """Control the node (Stop/Pause/Resume)"""
    if not mesh_controller:
        return jsonify({'error': 'Node not connected'}), 503
        
    data = request.json
    action = data.get('action')
    
    if action == 'pause':
        if hasattr(mesh_controller, 'pause'):
            mesh_controller.pause()
            return jsonify({'status': 'success', 'message': 'Node Paused'})
    elif action == 'resume':
        if hasattr(mesh_controller, 'resume'):
            mesh_controller.resume()
            return jsonify({'status': 'success', 'message': 'Node Resumed'})
    elif action == 'stop':
        def delayed_stop():
            time.sleep(1)
            mesh_controller.stop()
            os._exit(0)
            
        threading.Thread(target=delayed_stop).start()
        return jsonify({'status': 'success', 'message': 'Node Stopping...'})
        
    return jsonify({'error': 'Invalid action'}), 400

@app.route('/api/plugins')
def api_list_plugins():
    """List loaded plugins"""
    if not mesh_controller:
        return jsonify({'error': 'Node not connected'}), 503
    
    try:
        raw_plugins = mesh_controller.plugin_registry.list_plugins()
        # Transform [('name', {config})] -> [{'name':..., 'version':...}]
        plugins = []
        for p_name, p_config in raw_plugins:
            plugins.append({
                'name': p_name,
                'version': p_config.get('version', '0.0.0'),
                'description': p_config.get('description', ''),
                'type': p_config.get('type', 'AI')
            })
            
        return jsonify({'plugins': plugins, 'count': len(plugins)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit_job', methods=['POST'])
def api_submit_job():
    """Handle job submission from Client Portal"""
    try:
        if not mesh_controller:
            return jsonify({'error': 'Node not ready'}), 503
            
        plugin_name = request.form.get('plugin')
        input_data = {}
        
        # 1. Handle File Uploads (Generic)
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                import base64
                file_content = file.read()
                encoded_string = base64.b64encode(file_content).decode('utf-8')
                
                input_data['data'] = encoded_string
                input_data['image_data'] = encoded_string 
                input_data['file_data'] = encoded_string
                input_data['filename'] = file.filename
                input_data['content_type'] = file.content_type
                input_data['size_bytes'] = len(file_content)

        # 2. Handle Text / Code Inputs
        if request.form.get('text'):
            input_data['text'] = request.form.get('text')
            
        if request.form.get('code'):
            import base64
            code_str = request.form.get('code')
            input_data['code'] = base64.b64encode(code_str.encode('utf-8')).decode('utf-8')
            
        if request.form.get('function_name'):
            input_data['function_name'] = request.form.get('function_name')
            
        if request.form.get('args'):
            try:
                import json
                input_data['args'] = json.loads(request.form.get('args'))
            except:
                input_data['args'] = {}

        # 3. Submit to Mesh
        task_id = mesh_controller.submit_task(plugin_name, input_data)
        
        # 4. Wait for result (Short polling for demo)
        result = None
        for _ in range(30): # Wait up to 3 seconds
            result = mesh_controller.get_result(task_id, timeout=0.1)
            if result: break
            time.sleep(0.1)
            
        if result:
             return jsonify({
                'status': 'success',
                'task_id': task_id,
                'result': result
            })
        else:
            return jsonify({
                'status': 'success',
                'task_id': task_id,
                'result': {'status': 'processing', 'message': 'Task queued.'}
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/task_status/<task_id>', methods=['GET'])
def api_task_status(task_id):
    """Check status of a specific task (Polling Endpoint)"""
    if not mesh_controller:
        return jsonify({'error': 'Node not ready'}), 503
    
    # Check if result is ready
    result = mesh_controller.get_result(task_id, timeout=0.01)
    
    if result:
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'result': result
        })
    else:
        return jsonify({
            'status': 'processing',
            'task_id': task_id,
            'message': 'Task is still running...'
        })

def start_professional_ui(host='0.0.0.0', port=8000):
    """
    Start the professional web UI.
    """
    print("PROFESSIONAL WEB UI STARTING")
    
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    start_professional_ui()
