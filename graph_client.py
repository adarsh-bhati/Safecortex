"""FalkorDB client: the shared blackboard for all SafeCortex agents."""
import hashlib
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from falkordb import FalkorDB

LOCK_TIMEOUT_SECONDS = 60

_embedder = None
_embedder_failed = False


def embed_text(text: str) -> List[float]:
    """Embed text with sentence-transformers; fall back to a deterministic hash vector."""
    global _embedder, _embedder_failed
    if _embedder is None and not _embedder_failed and os.getenv("SAFECORTEX_EMBEDDINGS", "1") == "1":
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder_failed = True
    if _embedder is not None:
        return [float(x) for x in _embedder.encode(text)]
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:16]]


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class GraphClient:
    def __init__(self, host: str = "localhost", port: int = 6379, graph_name: str = "safecortex"):
        self.host, self.port, self.graph_name = host, port, graph_name
        self.db = None
        self.graph = None

    # ---------- connection / schema ----------
    def connect(self):
        self.db = FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)
        self.initialize_schema()
        return self

    def q(self, query: str, params: Optional[Dict[str, Any]] = None):
        return self.graph.query(query, params or {})

    def initialize_schema(self):
        for stmt in (
            "CREATE INDEX FOR (t:Task) ON (t.status)",
            "CREATE INDEX FOR (t:Task) ON (t.id)",
            "CREATE INDEX FOR (o:Observation) ON (o.timestamp)",
            "CREATE INDEX FOR (g:Goal) ON (g.id)",
            "CREATE INDEX FOR (b:Belief) ON (b.id)",
            "CREATE INDEX FOR (c:Constraint) ON (c.id)",
            "CREATE INDEX FOR (a:Agent) ON (a.id)",
        ):
            try:
                self.q(stmt)
            except Exception:
                pass  # index already exists

    def clear(self):
        try:
            self.graph.delete()
        except Exception:
            pass
        self.graph = self.db.select_graph(self.graph_name)
        self.initialize_schema()

    # ---------- agents ----------
    def register_agent(self, agent_id: str, role: str, status: str = "idle"):
        self.q("MERGE (a:Agent {id:$id}) SET a.role=$role, a.status=$status",
               {"id": agent_id, "role": role, "status": status})

    def set_agent_status(self, agent_id: str, status: str):
        self.q("MATCH (a:Agent {id:$id}) SET a.status=$status", {"id": agent_id, "status": status})

    def get_agents(self) -> List[Dict]:
        r = self.q("MATCH (a:Agent) RETURN a.id, a.role, a.status ORDER BY a.id")
        return [{"id": x[0], "role": x[1], "status": x[2]} for x in r.result_set]

    # ---------- goals / tasks ----------
    def add_goal(self, description: str, goal_id: Optional[str] = None) -> str:
        gid = goal_id or _id("goal")
        self.q("CREATE (:Goal {id:$id, description:$d, status:'active', created_at:$ts})",
               {"id": gid, "d": description, "ts": time.time()})
        return gid

    def get_blocked_tasks_for_latest_goal(self, limit: int = 50) -> List[Dict]:
        """Blocked tasks belonging to the most recently created Goal."""
        r = self.q(
            "MATCH (g:Goal) WITH g ORDER BY coalesce(g.created_at, 0) DESC LIMIT 1 "
            "MATCH (g)-[:HAS_TASK]->(t:Task {status:'blocked'}) "
            "RETURN t.id, t.description, t.assigned_to LIMIT $n", {"n": limit})
        return [{"id": x[0], "description": x[1], "assigned_to": x[2]} for x in r.result_set]

    def get_active_goals(self) -> List[Dict]:
        r = self.q("MATCH (g:Goal {status:'active'}) RETURN g.id, g.description")
        return [{"id": x[0], "description": x[1]} for x in r.result_set]

    def set_goal_status(self, goal_id: str, status: str):
        self.q("MATCH (g:Goal {id:$id}) SET g.status=$s", {"id": goal_id, "s": status})

    def add_task(self, description: str, goal_id: Optional[str] = None,
                 depends_on: Optional[List[str]] = None, task_id: Optional[str] = None,
                 status: str = "pending") -> str:
        tid = task_id or _id("task")
        self.q("CREATE (:Task {id:$id, description:$d, status:$st, assigned_to:'', lock_timestamp:0})",
               {"id": tid, "d": description, "st": status})
        if goal_id:
            self.q("MATCH (g:Goal {id:$g}), (t:Task {id:$t}) MERGE (g)-[:HAS_TASK]->(t)", {"g": goal_id, "t": tid})
        for dep in depends_on or []:
            self.q("MATCH (a:Task {id:$a}), (b:Task {id:$b}) MERGE (a)-[:DEPENDS_ON]->(b)", {"a": tid, "b": dep})
        return tid

    def count_open_tasks(self) -> int:
        r = self.q("MATCH (t:Task) WHERE t.status IN ['pending','verifying','verified','locked'] RETURN count(t)")
        return r.result_set[0][0]

    def get_pending_tasks(self, limit: int = 10) -> List[Dict]:
        """Pending tasks whose dependencies are all done."""
        self.release_stale_locks()
        r = self.q(
            "MATCH (t:Task {status:'pending'}) "
            "OPTIONAL MATCH (t)-[:DEPENDS_ON]->(d:Task) WHERE d.status <> 'done' "
            "WITH t, count(d) AS open WHERE open = 0 "
            "RETURN t.id, t.description, t.status ORDER BY t.id LIMIT $n", {"n": limit})
        return [{"id": x[0], "description": x[1], "status": x[2]} for x in r.result_set]

    def get_tasks_by_status(self, status: str, limit: int = 10) -> List[Dict]:
        r = self.q("MATCH (t:Task {status:$s}) RETURN t.id, t.description, t.assigned_to LIMIT $n",
                   {"s": status, "n": limit})
        return [{"id": x[0], "description": x[1], "assigned_to": x[2]} for x in r.result_set]

    def lock_task(self, task_id: str, agent_id: str, from_status: str = "pending",
                  to_status: str = "locked") -> bool:
        """Atomic compare-and-set. FalkorDB serializes write queries, and the status
        predicate + SET happen inside one query, so two agents cannot both win."""
        r = self.q(
            "MATCH (t:Task {id:$id}) WHERE t.status=$from "
            "SET t.status=$to, t.assigned_to=$agent, t.lock_timestamp=$ts RETURN t.id",
            {"id": task_id, "from": from_status, "to": to_status, "agent": agent_id, "ts": time.time()})
        if not r.result_set:
            return False
        self.q("MATCH (t:Task {id:$id}), (a:Agent {id:$a}) "
               "OPTIONAL MATCH (t)-[old:ASSIGNED_TO]->(:Agent) DELETE old "
               "MERGE (t)-[:ASSIGNED_TO]->(a)", {"id": task_id, "a": agent_id})
        return True

    def release_stale_locks(self):
        cut = {"cut": time.time() - LOCK_TIMEOUT_SECONDS}
        # Executor locks came from 'verified'; verifier locks came from 'pending'.
        self.q("MATCH (t:Task {status:'locked'}) WHERE t.lock_timestamp < $cut "
               "SET t.status='verified', t.assigned_to=''", cut)
        self.q("MATCH (t:Task {status:'verifying'}) WHERE t.lock_timestamp < $cut "
               "SET t.status='pending', t.assigned_to=''", cut)

    def set_task_status(self, task_id: str, status: str):
        self.q("MATCH (t:Task {id:$id}) SET t.status=$s", {"id": task_id, "s": status})

    def get_task(self, task_id: str) -> Optional[Dict]:
        r = self.q("MATCH (t:Task {id:$id}) RETURN t.id, t.description, t.status", {"id": task_id})
        if not r.result_set:
            return None
        x = r.result_set[0]
        return {"id": x[0], "description": x[1], "status": x[2]}

    # ---------- constraints ----------
    def add_constraint(self, rule: str, ctype: str = "forbidden_keyword", cid: Optional[str] = None) -> str:
        cid = cid or _id("constraint")
        self.q("MERGE (c:Constraint {id:$id}) SET c.rule=$rule, c.type=$type",
               {"id": cid, "rule": rule, "type": ctype})
        return cid

    def get_constraints(self) -> List[Dict]:
        r = self.q("MATCH (c:Constraint) RETURN c.id, c.rule, c.type")
        return [{"id": x[0], "rule": x[1], "type": x[2]} for x in r.result_set]

    def add_violation(self, task_id: str, constraint_id: str):
        self.q("MATCH (t:Task {id:$t}), (c:Constraint {id:$c}) MERGE (t)-[:VIOLATES]->(c)",
               {"t": task_id, "c": constraint_id})

    # ---------- observations / beliefs ----------
    def add_observation(self, content: str, agent_id: Optional[str] = None,
                        task_id: Optional[str] = None) -> str:
        oid = _id("obs")
        self.q("CREATE (:Observation {id:$id, content:$c, timestamp:$ts, embedding:$emb})",
               {"id": oid, "c": content, "ts": time.time(), "emb": embed_text(content)})
        if agent_id:
            self.q("MATCH (a:Agent {id:$a}), (o:Observation {id:$o}) MERGE (a)-[:PRODUCED]->(o)",
                   {"a": agent_id, "o": oid})
        if task_id:
            self.q("MATCH (t:Task {id:$t}), (o:Observation {id:$o}) MERGE (t)-[:YIELDED]->(o)",
                   {"t": task_id, "o": oid})
        return oid

    def get_recent_observations(self, limit: int = 10, since: float = 0.0) -> List[Dict]:
        r = self.q("MATCH (o:Observation) WHERE o.timestamp > $since "
                   "OPTIONAL MATCH (a:Agent)-[:PRODUCED]->(o) "
                   "RETURN o.id, o.content, o.timestamp, a.id ORDER BY o.timestamp DESC LIMIT $n",
                   {"since": since, "n": limit})
        return [{"id": x[0], "content": x[1], "timestamp": x[2], "agent": x[3]} for x in r.result_set]

    def add_belief(self, statement: str, confidence: float = 0.5, agent_id: Optional[str] = None) -> str:
        bid = _id("belief")
        self.q("CREATE (:Belief {id:$id, statement:$s, confidence:$c})",
               {"id": bid, "s": statement, "c": float(confidence)})
        if agent_id:
            self.q("MATCH (a:Agent {id:$a}), (b:Belief {id:$b}) MERGE (a)-[:PRODUCED]->(b)",
                   {"a": agent_id, "b": bid})
        return bid

    def get_beliefs(self) -> List[Dict]:
        r = self.q("MATCH (b:Belief) RETURN b.id, b.statement, b.confidence")
        return [{"id": x[0], "statement": x[1], "confidence": x[2]} for x in r.result_set]

    def update_belief_confidence(self, belief_id: str, confidence: float):
        self.q("MATCH (b:Belief {id:$id}) SET b.confidence=$c",
               {"id": belief_id, "c": max(0.0, min(1.0, float(confidence)))})

    def link_observation(self, obs_id: str, belief_id: str, supports: bool = True):
        rel = "SUPPORTS" if supports else "CONTRADICTS"
        self.q(f"MATCH (o:Observation {{id:$o}}), (b:Belief {{id:$b}}) MERGE (o)-[:{rel}]->(b)",
               {"o": obs_id, "b": belief_id})

    # ---------- visualization ----------
    def get_graph_data_for_vis(self) -> Dict[str, List[Dict]]:
        nodes, edges = [], []
        r = self.q("MATCH (n) RETURN labels(n)[0], n.id, "
                   "coalesce(n.description, n.content, n.statement, n.rule, n.role), "
                   "n.status, n.confidence")
        for label, nid, text, status, conf in r.result_set:
            nodes.append({"id": nid, "type": label, "label": text or nid, "status": status, "confidence": conf})
        r = self.q("MATCH (a)-[e]->(b) RETURN a.id, b.id, type(e)")
        for src, dst, rel in r.result_set:
            edges.append({"source": src, "target": dst, "type": rel})
        return {"nodes": nodes, "edges": edges}
