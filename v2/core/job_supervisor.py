"""
Job Supervisor (MapReduce Orchestrator)
Handles composite tasks: Split -> Distribute -> Wait -> Join -> Result
"""
import time
import logging
import uuid
import threading
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class JobSupervisor:
    def __init__(self, controller):
        self.controller = controller
        self.active_jobs: Dict[str, Dict] = {} # {job_id: {state, chunks, results, ...}}
        self.lock = threading.Lock()
        self.running = True
        
        # Start background monitor for timeouts / completions
        self.thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self.thread.start()

    def submit_composite_job(self, split_plugin: str, join_plugin: str, 
                             input_data: Dict[str, Any], priority: int = 5) -> str:
        """
        Start a new composite job.
        1. Run Splitter (Locally or remotely)
        2. Create sub-tasks for each chunk
        3. Wait for results
        4. Run Joiner
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        with self.lock:
            self.active_jobs[job_id] = {
                'id': job_id,
                'status': 'Splitting',
                'created_at': time.time(),
                'split_plugin': split_plugin,
                'join_plugin': join_plugin,
                'input_data': input_data,
                'priority': priority,
                'sub_tasks': {}, # {task_id: {status, result}}
                'chunks': [],
                'final_result': None,
                'progress': 0,
                'eta': 0
            }
            
        logger.info(f"JOB [{job_id}] Started. Running Splitter '{split_plugin}'...")
        
        # Run Splitter (Synchronous for now, or could be a task)
        # Ideally, we run this via the controller's plugin engine to get acceleration
        try:
            # We use the generic execute method but force local? 
            # Or just submit it as a high priority task and wait?
            # Let's run locally for simplicity / stability of the supervisor
            
            splitter = self.controller.plugin_registry.get_plugin(split_plugin)
            if not splitter:
                self._fail_job(job_id, f"Splitter plugin '{split_plugin}' not found")
                return job_id

            # Execute Split
            split_result = self.controller.inference_engine.run(splitter, input_data)
            
            # Expecting result to contain a list of chunks under 'segments' or similar
            # Adaptation for 'video_splitter': returns {'segments': [...]}
            chunks = split_result.get('segments', [])
            if not chunks:
                 self._fail_job(job_id, "Splitter produced no chunks")
                 return job_id
                 
            logger.info(f"JOB [{job_id}] Split into {len(chunks)} chunks. Distributing...")
            
            with self.lock:
                self.active_jobs[job_id]['status'] = 'Distributing'
                self.active_jobs[job_id]['chunks'] = chunks
                
            # Fan-Out
            for i, chunk in enumerate(chunks):
                # Construct sub-task input
                # The 'processor' plugin is implied... wait.
                # The user request said "split it into differnt jobs". 
                # We need to know WHICH plugin processes the chunks!
                # Let's assume input_data has 'process_plugin' or we infer it.
                # For video: split -> (nothing?) -> join? 
                # Usually: video_splitter -> process_plugin (e.g. grayscale) -> video_joiner
                
                process_plugin = input_data.get('process_plugin')
                if not process_plugin:
                    # If no processor, maybe the joiner takes the raw splits? (Unlikely)
                    # Let's assume the user wants to process them.
                    self._fail_job(job_id, "No 'process_plugin' specified in input_data")
                    return job_id
                    
                sub_input = chunk.copy() # Contains file path etc
                # Pass original params if needed?
                
                sub_task_id = self.controller.submit_task(
                    plugin_name=process_plugin,
                    input_data=sub_input,
                    priority=priority
                )
                
                with self.lock:
                    self.active_jobs[job_id]['sub_tasks'][sub_task_id] = {
                        'status': 'pending',
                        'chunk_index': i
                    }
            
            with self.lock:
                self.active_jobs[job_id]['status'] = 'Processing'
                self.active_jobs[job_id]['total_tasks'] = len(chunks)
                
        except Exception as e:
            logger.error(f"JOB [{job_id}] Failed during split: {e}")
            self._fail_job(job_id, str(e))
            
        return job_id

    def _monitor_jobs(self):
        """Monitor progress of composite jobs"""
        while self.running:
            try:
                # Iterate copy of keys to avoid locking issues during iteration
                job_ids = list(self.active_jobs.keys())
                
                for job_id in job_ids:
                    job = self.active_jobs[job_id]
                    
                    if job['status'] in ['Completed', 'Failed']:
                        continue
                        
                    if job['status'] == 'Processing':
                        # Check sub-tasks
                        completed_count = 0
                        failed_count = 0
                        results = []
                        
                        for tid, info in job['sub_tasks'].items():
                            # Poll controller for task status
                            # Hack: Access controller's internal state or Ledger?
                            # Optimally controller should callback/notify us.
                            # For now, we poll controller.results or similar
                            
                            # Check controller internal results cache (we need to ensure it keeps them!)
                            # Assuming controller.results[tid] exists
                            if tid in self.controller.results:
                                res = self.controller.results[tid]
                                if res['status'] == 'success':
                                    info['status'] = 'completed'
                                    info['result'] = res['result']
                                else:
                                    info['status'] = 'failed'
                                    info['error'] = res.get('error')
                                    failed_count += 1
                                    
                            if info['status'] == 'completed':
                                completed_count += 1
                                results.append(info['result'])
                                
                        # Update Progress
                        total = len(job['sub_tasks'])
                        if total > 0:
                            progress = (completed_count / total) * 100
                            job['progress'] = progress
                            # Simple ETA: (time_elapsed / pct) * remaining_pct
                            elapsed = time.time() - job['created_at']
                            if progress > 10:
                                job['eta'] = (elapsed / progress) * (100 - progress)
                        
                        # Check Completion
                        if completed_count == total:
                            # All done -> JOIN
                            self._run_joiner(job_id, results)
                        elif failed_count > 0:
                            # Retry logic? For now, fail hard.
                            self._fail_job(job_id, "One or more sub-tasks failed.")
                            
            except Exception as e:
                logger.error(f"Job Monitor Error: {e}")
                
            time.sleep(1.0)

    def _run_joiner(self, job_id, results):
        """Run the Joiner plugin"""
        logger.info(f"JOB [{job_id}] All tasks done. Joining...")
        with self.lock:
             self.active_jobs[job_id]['status'] = 'Joining'
             
        try:
            job = self.active_jobs[job_id]
            joiner_name = job['join_plugin']
            joiner = self.controller.plugin_registry.get_plugin(joiner_name)
            
            # Prepare Join Input
            # usually {'segments': [results...], 'original_filename': ...}
            join_input = {
                'segments': results,
                'original_filename': job['input_data'].get('filename', 'output')
            }
            
            # Run Joiner
            join_result = self.controller.inference_engine.run(joiner, join_input)
            
            # Finish
            with self.lock:
                self.active_jobs[job_id]['status'] = 'Completed'
                self.active_jobs[job_id]['final_result'] = join_result
                self.active_jobs[job_id]['progress'] = 100
                self.active_jobs[job_id]['eta'] = 0
                
            logger.info(f"JOB [{job_id}] COMPLETED SUCCESSFULLY.")
            
        except Exception as e:
             logger.error(f"JOB [{job_id}] Join Failed: {e}")
             self._fail_job(job_id, f"Join failed: {e}")

    def _fail_job(self, job_id, error):
        with self.lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'Failed'
                self.active_jobs[job_id]['error'] = error
        logger.error(f"JOB [{job_id}] FAILED: {error}")
