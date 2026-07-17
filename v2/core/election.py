
"""
Decentralized Leader Election (Raft-inspired)
Ensures the Mesh never dies. If the Coordinator fails, a Worker promotes itself.
Selection Criteria: Highest Staked Amount + Longest Uptime.
"""
import time
import threading
import logging
import random
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LeaderElection:
    def __init__(self, node_id: str, ledger, net_manager, reputation_manager, on_promote_callback, peer_count_fn=None):
        self.node_id = node_id
        self.ledger = ledger
        self.net_manager = net_manager
        self.reputation = reputation_manager
        self.on_promote = on_promote_callback
        self.get_peer_count = peer_count_fn if peer_count_fn else lambda: 0
        
        # States: FOLLOWER, CANDIDATE, LEADER
        self.state = "FOLLOWER"
        self.leader_id = None
        self.last_heartbeat = time.time()
        
        # Raft Parameters
        self.HEARTBEAT_TIMEOUT = 5.0 
        self.ELECTION_TIMEOUT = random.uniform(5.0, 8.0) 
        
        self.votes_received = 0
        self.term = 0
        
        # Background Monitor
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def _monitor_loop(self):
        """Watchdog: Checks if Leader is alive"""
        while self.running:
            time.sleep(1.0)
            
            if self.state == "LEADER":
                self._send_heartbeat()
            elif self.state == "FOLLOWER":
                if time.time() - self.last_heartbeat > self.HEARTBEAT_TIMEOUT:
                    logger.warning(f"[ELECTION] Leader Dead? Timeout reached. Starting Election.")
                    self._start_election()
                    
    def _start_election(self):
        """Promote self to CANDIDATE and ask for votes"""
        self.state = "CANDIDATE"
        self.term += 1
        self.votes_received = 1 
        
        my_score = self.reputation.get_score()
        logger.info(f"[ELECTION] I am Candidate for Term {self.term} (Score: {my_score})")
        
        # [FIX] Check for Immediate Win (Single Node)
        if self._check_win():
            return

        msg = {
            "type": "ELECTION_VOTE_REQUEST",
            "candidate_id": self.node_id,
            "term": self.term,
            "score": str(my_score)
        }
        self.net_manager.broadcast_message(msg)
        
    def _check_win(self):
        """Check if we have majority votes"""
        peers = self.get_peer_count()
        total_votes = peers + 1 # Myself + Peers
        majority = (total_votes // 2) + 1
        
        if self.votes_received >= majority:
            self.state = "LEADER"
            self.leader_id = self.node_id
            logger.info(f"[ELECTION] 👑 WON ELECTION! Votes: {self.votes_received}/{total_votes}")
            if self.on_promote:
                self.on_promote()
            self._send_heartbeat()
            return True
        return False

    def handle_message(self, msg: Dict[str, Any]):
        """Process Election Messages"""
        m_type = msg.get('type')
        
        if m_type == "LEADER_HEARTBEAT":
            self.last_heartbeat = time.time()
            self.leader_id = msg.get('leader_id')
            if self.state != "FOLLOWER":
                logger.info(f"[ELECTION] New Leader Found: {self.leader_id}. Stepping down.")
                self.state = "FOLLOWER"
                
        elif m_type == "ELECTION_VOTE_REQUEST":
            # Decide to vote based on REPUTATION SCORE
            candidate_score = float(msg.get('score', 0))
            my_score = self.reputation.get_score()
            candidate_id = msg.get('candidate_id')
            
            # Meritocratic Vote: Only vote if they are better or equal
            if candidate_score >= my_score:
                logger.info(f"[ELECTION] Voted for {candidate_id}")
                # Send ACK
                ack = {
                    "type": "ELECTION_VOTE_ACK",
                    "voter_id": self.node_id,
                    "candidate_id": candidate_id
                }
                # Since we don't have direct send, broadcast ACK (lazy but works for mesh)
                self.net_manager.broadcast_message(ack)
        
        elif m_type == "ELECTION_VOTE_ACK":
             if self.state == "CANDIDATE" and msg.get('candidate_id') == self.node_id:
                 self.votes_received += 1
                 logger.info(f"[ELECTION] Received Vote from {msg.get('voter_id')}")
                 self._check_win()
                
    def stop(self):
        self.running = False

    def _send_heartbeat(self):
        """Leader: I am alive"""
        msg = {
            "type": "LEADER_HEARTBEAT",
            "leader_id": self.node_id,
            "term": self.term
        }
        self.net_manager.broadcast_message(msg)
