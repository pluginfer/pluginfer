"""RETIRED in CP-1.

The original test imported `core.mesh_controller.MeshNetworkController`
and `find_coordinator`. Both were archived under `_archive_v2/` when
the mesh layer was rewritten as `core.complete_mesh_controller.
CompleteMeshController` plus `core.discovery` (the new LAN-UDP
coordinator-discovery module).

The discovery feature is now exercised end-to-end in:
  - tests/test_e2e_product.py::test_two_node_mesh_forms (in-process)
  - tests/test_brain_full_integration.py (full chain)

Keeping this file so git history preserves the deletion intent.
Pytest finds zero tests here, which is the point.
"""
