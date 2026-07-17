# Mesh Networking Guide

## 🌐 Distributed Computing with Pluginfer

Pluginfer's mesh networking enables you to connect multiple machines (with different GPUs or CPUs) into a unified compute cluster. This turns separate computers into a powerful, distributed AI inference system.

---

## What is Mesh Networking?

Mesh networking allows you to:
- **Pool Resources**: Combine GPUs/CPUs from multiple machines
- **Scale Horizontally**: Add more machines to increase capacity
- **Distribute Workload**: Automatically balance tasks across nodes
- **Earn Revenue**: Contributors get paid for compute time
- **Fault Tolerant**: System continues if nodes go offline

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  COORDINATOR NODE                    │
│  (Orchestrates tasks, manages mesh)                  │
│  • Task queue management                            │
│  • Load balancing                                   │
│  • Node health monitoring                           │
│  • Revenue distribution                             │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
           │                      │
    ┌──────▼──────┐        ┌─────▼──────┐
    │ WORKER NODE │        │ WORKER NODE│
    │ GPU: CUDA   │        │ GPU: MPS    │
    │ 2x RTX 4090 │        │ Apple M2    │
    └──────┬──────┘        └─────┬──────┘
           │                     │
           └──────────┬──────────┘
                      │
               ┌──────▼──────┐
               │ WORKER NODE │
               │ CPU: 16 core│
               │ No GPU      │
               └─────────────┘
```

---

## Quick Start

### 1. Start Coordinator (Main Machine)

```python
from core.mesh_controller import MeshNetworkController

# Start coordinator
coordinator = MeshNetworkController(
    host='0.0.0.0',  # Listen on all interfaces
    port=9999,
    mode='coordinator'
)
coordinator.start()

print(f"Coordinator started on port 9999")
print(f"Workers can connect to: {your_ip}:9999")
```

### 2. Start Workers (Other Machines)

```python
from core.mesh_controller import (
    MeshNetworkController,
    get_system_capabilities,
    connect_to_mesh
)

# Get this machine's capabilities
capabilities = get_system_capabilities()

# Connect to coordinator
connect_to_mesh(
    coordinator_host='192.168.1.100',  # Coordinator IP
    coordinator_port=9999,
    capabilities=capabilities
)

# Start worker
worker = MeshNetworkController(
    host='0.0.0.0',
    port=10000,
    mode='worker'
)
worker.start()

print("Worker ready! Processing tasks...")
```

### 3. Submit Tasks

```python
# On coordinator - submit batch of tasks
batch = [
    {'data': [i]} for i in range(1000)
]

task_ids = coordinator.distribute_batch(
    plugin_name='SimpleAI',
    batch=batch,
    strategy='load_balanced'
)

print(f"Distributed {len(task_ids)} tasks!")
```

---

## Node Modes

### Coordinator Mode
- Manages task distribution
- Monitors node health
- Handles load balancing
- Tracks revenue
- Does NOT execute tasks

```python
coordinator = MeshNetworkController(
    host='0.0.0.0',
    port=9999,
    mode='coordinator'
)
```

### Worker Mode
- Executes tasks
- Reports results
- Monitors health
- Earns revenue
- Does NOT coordinate

```python
worker = MeshNetworkController(
    host='0.0.0.0',
    port=10000,
    mode='worker'
)
```

### Hybrid Mode
- Can coordinate AND execute
- Best for small clusters
- Self-sufficient node
- Full flexibility

```python
hybrid = MeshNetworkController(
    host='0.0.0.0',
    port=9999,
    mode='hybrid'
)
```

---

## Distribution Strategies

### 1. Load Balanced (Recommended)
Distributes tasks based on compute power.

```python
task_ids = coordinator.distribute_batch(
    plugin_name='ImageClassifier',
    batch=batch,
    strategy='load_balanced'
)
```

**How it works:**
- Calculates compute score for each node
- GPU nodes get more tasks than CPU nodes
- Multi-GPU nodes get proportionally more
- Automatic scaling based on capacity

**Example Distribution:**
```
Node 1: 2x RTX 4090 (score: 250) → 60% of tasks
Node 2: Apple M2    (score: 100) → 30% of tasks  
Node 3: CPU 16-core (score: 50)  → 10% of tasks
```

### 2. Round Robin
Simple rotation through nodes.

```python
task_ids = coordinator.distribute_batch(
    plugin_name='TextProcessor',
    batch=batch,
    strategy='round_robin'
)
```

**When to use:**
- All nodes have similar power
- Simple distribution needed
- Testing/debugging

### 3. Fastest First
Uses only the fastest nodes.

```python
task_ids = coordinator.distribute_batch(
    plugin_name='FastInference',
    batch=batch,
    strategy='fastest_first'
)
```

**When to use:**
- Time-critical tasks
- Few but powerful nodes
- Maximum throughput needed

---

## Compute Power Scoring

Nodes are scored based on their capabilities:

```python
Base Score: 10

GPU Bonuses:
+ 100 for CUDA (NVIDIA)
+ 80 for MPS (Apple Silicon)
+ 70 for ROCm (AMD)
+ 50 per additional GPU

CPU:
+ 5 per core

Memory:
+ 1 per GB
```

**Examples:**
```
2x RTX 4090, 16 cores, 32GB RAM:
  100 (CUDA) + 50 (2nd GPU) + 80 (cores) + 32 (RAM) = 262

Apple M2, 8 cores, 16GB RAM:
  80 (MPS) + 40 (cores) + 16 (RAM) = 136

CPU only, 8 cores, 16GB RAM:
  40 (cores) + 16 (RAM) = 56
```

---

## Revenue System

Contributors earn revenue for processing tasks.

### How it Works

1. **Task Pricing**: Each task has a value (e.g., $0.01)
2. **Execution**: Worker processes task
3. **Payment**: Worker earns the task value
4. **Tracking**: System tracks earnings per node

### Example

```python
# Check earnings
stats = coordinator.get_mesh_stats()
print(f"Revenue earned: ${stats['revenue_earned']:.4f}")

# Per-node earnings
for node_id, node in coordinator.nodes.items():
    earnings = calculate_node_earnings(node)
    print(f"{node_id}: ${earnings:.2f}")
```

### Revenue Distribution

Based on work completed:
```
If 1000 tasks at $0.01 each = $10 total

Node 1: Completed 600 tasks → $6.00
Node 2: Completed 300 tasks → $3.00  
Node 3: Completed 100 tasks → $1.00
```

---

## Real-World Examples

### Example 1: Personal Cluster
**Scenario**: You have 3 machines at home
- Gaming PC (2x RTX 4090)
- MacBook Pro (M2)
- Old Linux server (CPU only)

**Setup:**
```python
# Gaming PC - Coordinator
coordinator = MeshNetworkController('0.0.0.0', 9999, 'hybrid')
coordinator.start()

# MacBook - Worker (auto-connect)
# Linux - Worker (auto-connect)

# Submit 10,000 image classifications
batch = load_images(10000)
task_ids = coordinator.distribute_batch('ImageNet', batch, 'load_balanced')

# Gaming PC does 70%, Mac does 25%, Linux does 5%
```

### Example 2: Startup AI Service
**Scenario**: Small company with:
- 1 cloud server (AWS with 4x A100)
- 2 employee laptops (contributing when idle)

**Benefits:**
- Use expensive GPUs efficiently
- Utilize idle laptop time
- Pay employees for compute contribution
- Scale up/down based on demand

### Example 3: Community Compute Pool
**Scenario**: 50 people pool their GPUs
- Everyone contributes compute
- Everyone can use the pool
- Revenue shared based on contribution

**Implementation:**
```python
# Each person runs a worker
worker = MeshNetworkController('0.0.0.0', 10000, 'worker')
capabilities = get_system_capabilities()
connect_to_mesh('community.example.com', 9999, capabilities)
worker.start()

# Central coordinator manages distribution
# Revenue tracked per node
# Monthly payouts based on compute provided
```

---

## Security Considerations

### Phase 1 (Current)
- Local network only
- Trusted nodes
- No encryption
- Basic authentication

### Phase 2 (Planned)
- TLS encryption
- Node authentication
- Task verification
- Sandboxed execution
- Tamper detection

---

## Monitoring & Management

### Check Mesh Status

```python
coordinator.print_mesh_status()
```

Output:
```
🌐 MESH NETWORK STATUS
======================================================================

Node ID: abc123def456
Mode: COORDINATOR

📊 Network:
  Total Nodes: 3
  Online: 3
  Offline: 0
  Total Compute Power: 386

⚙️  Tasks:
  Completed: 1000
  Failed: 5
  In Queue: 0

💰 Revenue: $10.0000

🖥️  Connected Nodes:
  ✅ 192.168.1.100:10000
     Compute: 250 | cuda
  ✅ 192.168.1.101:10001
     Compute: 100 | mps
  ✅ 192.168.1.102:10002
     Compute: 36 | CPU
```

### Node Health

Automatic health checks:
- Every 10 seconds
- Marks offline if not seen in 30s
- Auto-reconnect on recovery

### Task Tracking

```python
# Submit task
task_id = coordinator.submit_task('MyPlugin', {'data': [1,2,3]})

# Check result
result = coordinator.get_result(task_id, timeout=60)

if result:
    print(f"Task completed: {result}")
else:
    print("Task timed out or failed")
```

---

## Performance Tips

### 1. Network Optimization
- Use gigabit ethernet or better
- Minimize latency between nodes
- Co-locate nodes if possible
- Use dedicated network for mesh

### 2. Task Sizing
- Batch small tasks (reduces overhead)
- Don't send huge data (use references)
- Balance task complexity
- Monitor queue depth

### 3. Node Configuration
- Pin workers to specific GPUs
- Set resource limits
- Monitor temperature
- Use power limits if needed

### 4. Load Balancing
- Profile node performance first
- Adjust compute scores manually if needed
- Monitor task completion times
- Rebalance if needed

---

## Scaling Guidelines

### Small Scale (2-5 nodes)
- 1 coordinator
- Hybrid mode acceptable
- Simple load balancing
- Local network

### Medium Scale (5-20 nodes)
- Dedicated coordinator
- Separate workers
- Advanced load balancing
- May need multiple coordinators

### Large Scale (20+ nodes)
- Multiple coordinators
- Hierarchical structure
- Geographic distribution
- Professional networking

---

## Troubleshooting

### Nodes Not Connecting

**Check:**
1. Firewall allows port 9999
2. Coordinator is running
3. Correct IP address
4. Network connectivity

```bash
# Test connection
telnet coordinator_ip 9999
```

### Tasks Not Processing

**Check:**
1. Worker nodes are online
2. Plugins are installed on workers
3. Queue has tasks
4. No errors in logs

```python
# Check queue
stats = coordinator.get_mesh_stats()
print(f"Queue size: {stats['queue_size']}")

# Check nodes
for node in coordinator.nodes.values():
    print(f"{node.node_id}: {node.status}")
```

### Poor Performance

**Possible causes:**
1. Network latency too high
2. Tasks too small (overhead)
3. Unbalanced load
4. Slow nodes dominating

**Solutions:**
- Increase task batch size
- Adjust compute scores
- Remove slow nodes
- Use faster network

---

## API Reference

### MeshNetworkController

```python
class MeshNetworkController:
    def __init__(self, host: str, port: int, mode: str)
    
    def start()
        """Start the node"""
    
    def stop()
        """Stop the node"""
    
    def register_node(host, port, capabilities) -> str
        """Register a worker node"""
    
    def submit_task(plugin_name, input_data, priority=5) -> str
        """Submit a single task"""
    
    def distribute_batch(plugin_name, batch, strategy) -> List[str]
        """Distribute batch of tasks"""
    
    def get_result(task_id, timeout=60) -> Optional[Dict]
        """Get task result"""
    
    def get_mesh_stats() -> Dict
        """Get mesh statistics"""
    
    def print_mesh_status()
        """Print status to console"""
```

### Utility Functions

```python
def get_system_capabilities() -> Dict
    """Get current system's capabilities"""

def connect_to_mesh(coordinator_host, coordinator_port, capabilities) -> bool
    """Connect to a mesh as worker"""
```

---

## Future Enhancements (Phase 2)

- [ ] Web dashboard for monitoring
- [ ] Automatic node discovery
- [ ] Dynamic pricing
- [ ] Task prioritization
- [ ] Geographic routing
- [ ] Failover coordinators
- [ ] Blockchain integration for payments
- [ ] Smart contracts
- [ ] Reputation system
- [ ] SLA guarantees

---

## Examples

See `examples/example_mesh_network.py` for:
1. Coordinator setup
2. Worker setup
3. Hybrid node
4. Real-world inference farm scenario

Run:
```bash
python examples/example_mesh_network.py
```

---

## Summary

Mesh networking transforms Pluginfer from a single-machine system into a distributed compute platform. Key benefits:

✅ **Pool Resources**: Combine multiple machines
✅ **Scale Easily**: Add/remove nodes dynamically
✅ **Optimize Costs**: Use idle compute time
✅ **Earn Revenue**: Get paid for contributing
✅ **Fault Tolerant**: System continues if nodes fail
✅ **Load Balanced**: Automatic task distribution
✅ **Hardware Agnostic**: Mix NVIDIA, AMD, Apple, CPU

Start small with 2-3 machines and scale up as needed!

---

## Support

- **Documentation**: This guide
- **Examples**: `examples/example_mesh_network.py`
- **API Reference**: See above
- **Issues**: GitHub Issues

---

**🚀 Ready to build your compute mesh? Start with the Quick Start section!**
