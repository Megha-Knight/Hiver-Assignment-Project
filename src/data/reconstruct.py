"""Conversation reconstruction module for customer support tweets.

Reconstructs multi-turn conversational threads, enforces strict chronological order,
prevents duplicates, detects cycles defensively, and flags broken reference links.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TweetNode:
    """Internal graph node representation of a single tweet."""
    tweet_id: int
    author_id: str
    inbound: bool
    created_at_str: str
    timestamp: float
    text: str
    parent_id: Optional[int] = None
    response_ids: List[int] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """A single turn in a reconstructed conversation."""
    turn_index: int
    tweet_id: int
    author_id: str
    role: str  # 'customer' or 'support'
    created_at: str
    timestamp: float
    text: str

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "tweet_id": self.tweet_id,
            "author_id": self.author_id,
            "role": self.role,
            "created_at": self.created_at,
            "timestamp": self.timestamp,
            "text": self.text,
        }


@dataclass
class ReconstructedConversation:
    """A complete reconstructed multi-turn conversation thread."""
    conversation_id: str
    brand: str
    root_tweet_id: int
    turns: List[ConversationTurn]
    turn_count: int
    is_broken: bool
    broken_reasons: List[str] = field(default_factory=list)
    has_cycle: bool = False
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "root_tweet_id": self.root_tweet_id,
            "turn_count": self.turn_count,
            "is_broken": self.is_broken,
            "broken_reasons": self.broken_reasons,
            "has_cycle": self.has_cycle,
            "duration_seconds": round(self.duration_seconds, 1),
            "turns": [turn.to_dict() for turn in self.turns],
        }


class ConversationReconstructor:
    """Reconstructs conversational threads from tweet nodes with defensive checks."""

    DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"

    def __init__(self, brand: str):
        self.brand = brand
        self.nodes: Dict[int, TweetNode] = {}
        self.forward_edges: Dict[int, Set[int]] = {}
        self.backward_edges: Dict[int, Optional[int]] = {}

    def parse_timestamp(self, dt_str: str) -> float:
        """Parses Twitter datetime string into epoch timestamp."""
        try:
            dt = datetime.strptime(dt_str.strip(), self.DATE_FORMAT)
            return dt.timestamp()
        except Exception:
            return 0.0

    def add_tweet(
        self,
        tweet_id: int,
        author_id: str,
        inbound: bool,
        created_at: str,
        text: str,
        parent_id: Optional[int] = None,
        response_ids: Optional[List[int]] = None,
    ) -> None:
        """Adds a tweet node and registers references."""
        ts = self.parse_timestamp(created_at)
        resps = response_ids or []

        node = TweetNode(
            tweet_id=tweet_id,
            author_id=author_id,
            inbound=inbound,
            created_at_str=created_at,
            timestamp=ts,
            text=text,
            parent_id=parent_id,
            response_ids=resps,
        )
        self.nodes[tweet_id] = node

        # Index forward responses
        if tweet_id not in self.forward_edges:
            self.forward_edges[tweet_id] = set()
        for r in resps:
            self.forward_edges[tweet_id].add(r)

        # Index backward parent
        if parent_id is not None:
            self.backward_edges[tweet_id] = parent_id
            if parent_id not in self.forward_edges:
                self.forward_edges[parent_id] = set()
            self.forward_edges[parent_id].add(tweet_id)

    def reconstruct_all(self) -> List[ReconstructedConversation]:
        """Reconstructs all conversation threads from registered nodes using O(V+E) BFS."""
        # Find connected components via undirected adjacency graph
        adj: Dict[int, Set[int]] = {tid: set() for tid in self.nodes}
        for tid, node in self.nodes.items():
            if node.parent_id is not None:
                adj[tid].add(node.parent_id)
                if node.parent_id in adj:
                    adj[node.parent_id].add(tid)
            for r in node.response_ids:
                adj[tid].add(r)
                if r in adj:
                    adj[r].add(tid)

        visited_nodes: Set[int] = set()
        conversations: List[ReconstructedConversation] = []

        for start_id in sorted(self.nodes.keys()):
            if start_id in visited_nodes:
                continue

            # O(1) dequeue BFS
            queue = deque([start_id])
            visited_nodes.add(start_id)
            component_ids: List[int] = []

            is_broken = False
            broken_reasons: List[str] = []
            edge_count = 0

            while queue:
                curr_id = queue.popleft()
                if curr_id in self.nodes:
                    component_ids.append(curr_id)
                    curr_node = self.nodes[curr_id]

                    # Check parent link integrity
                    if curr_node.parent_id is not None:
                        edge_count += 1
                        if curr_node.parent_id not in self.nodes:
                            is_broken = True
                            broken_reasons.append(f"missing_parent_{curr_node.parent_id}")

                    # Check child response link integrity
                    for r_id in curr_node.response_ids:
                        if r_id not in self.nodes:
                            is_broken = True
                            broken_reasons.append(f"missing_child_{r_id}")

                for neighbor in adj.get(curr_id, set()):
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append(neighbor)

            if not component_ids:
                continue

            # Defensive cycle check: if undirected graph has edges >= vertices, cycle exists
            has_cycle = edge_count >= len(component_ids) and len(component_ids) > 1

            # Retrieve nodes and sort strictly chronologically by timestamp
            component_nodes = [self.nodes[tid] for tid in component_ids if tid in self.nodes]
            component_nodes.sort(key=lambda n: (n.timestamp, n.tweet_id))

            # Build conversation turns
            turns: List[ConversationTurn] = []
            for idx, n in enumerate(component_nodes, start=1):
                role = "customer" if n.inbound else "support"
                turns.append(
                    ConversationTurn(
                        turn_index=idx,
                        tweet_id=n.tweet_id,
                        author_id=n.author_id,
                        role=role,
                        created_at=n.created_at_str,
                        timestamp=n.timestamp,
                        text=n.text,
                    )
                )

            root_id = component_nodes[0].tweet_id
            conv_id = f"conv_{self.brand}_{root_id}"

            duration = 0.0
            if len(turns) > 1:
                duration = max(0.0, turns[-1].timestamp - turns[0].timestamp)

            conversations.append(
                ReconstructedConversation(
                    conversation_id=conv_id,
                    brand=self.brand,
                    root_tweet_id=root_id,
                    turns=turns,
                    turn_count=len(turns),
                    is_broken=is_broken,
                    broken_reasons=list(set(broken_reasons)),
                    has_cycle=has_cycle,
                    duration_seconds=duration,
                )
            )

        logger.info(
            f"Reconstructed {len(conversations):,} conversations for brand {self.brand}"
        )
        return conversations
