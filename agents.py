"""SafeCortex agents. They never talk to each other: all coordination goes through FalkorDB."""
import abc
import asyncio
import json
import logging
import random
import re
import sys
import time
from typing import Any, List, Optional

from graph_client import GraphClient
from llm_router import LLMRouter

log = logging.getLogger("agents")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM reply."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in LLM output")


class BaseAgent(abc.ABC):
    role = "base"

    def __init__(self, agent_id: str, graph: GraphClient, llm: LLMRouter, interval: float = 2.0):
        self.id, self.graph, self.llm, self.interval = agent_id, graph, llm, interval
        self.graph.register_agent(self.id, self.role)

    def log(self, msg: str):
        print(f"{time.strftime('%H:%M:%S')} [{self.id}] {msg}", flush=True)

    @abc.abstractmethod
    def step(self) -> bool:
        """Do one unit of work (blocking). Return True if work was done."""

    async def run(self, stop: Optional[asyncio.Event] = None):
        while not (stop and stop.is_set()):
            try:
                self.graph.set_agent_status(self.id, "working")
                did = await asyncio.to_thread(self.step)
                self.graph.set_agent_status(self.id, "active" if did else "idle")
            except Exception as e:
                self.log(f"error: {e}")
                try:
                    self.graph.set_agent_status(self.id, "error")
                except Exception:
                    pass
            await asyncio.sleep(self.interval)


class PlannerAgent(BaseAgent):
    role = "planner"

    SYSTEM = ("You are a task planner. Decompose the goal into 3-6 concrete tasks. Reply with ONLY a JSON array. "
              'Each item: {"key": "t1", "description": "...", "depends_on": ["t0"]}. '
              "depends_on lists keys of earlier tasks (may be empty).")

    def step(self) -> bool:
        if self.graph.count_open_tasks() > 0:
            self.maybe_complete_goals()
            return False
        goals = self.graph.get_active_goals()
        if not goals:
            return False
        goal = goals[0]
        has_tasks = self.graph.q("MATCH (:Goal {id:$g})-[:HAS_TASK]->(t) RETURN count(t)",
                                 {"g": goal["id"]}).result_set[0][0]
        if has_tasks:
            self.maybe_complete_goals()
            return False
        self.plan(goal["id"], goal["description"])
        return True

    def maybe_complete_goals(self):
        # An active goal with tasks and no open ones is finished.
        self.graph.q("MATCH (g:Goal {status:'active'})-[:HAS_TASK]->(:Task) "
                     "WITH DISTINCT g "
                     "OPTIONAL MATCH (g)-[:HAS_TASK]->(t:Task) "
                     "WHERE t.status IN ['pending','verifying','verified','locked'] "
                     "WITH g, count(t) AS open WHERE open = 0 "
                     "SET g.status='completed'")

    def plan(self, goal_id: str, description: str) -> List[str]:
        self.log(f"planning goal: {description}")
        try:
            tasks = extract_json(self.llm.generate(f"Goal: {description}", self.SYSTEM))
            assert isinstance(tasks, list) and tasks
        except Exception as e:
            self.log(f"LLM planning failed ({e}); using fallback plan")
            tasks = [{"key": "t1", "description": f"Research background for: {description}", "depends_on": []},
                     {"key": "t2", "description": f"Analyze findings for: {description}", "depends_on": ["t1"]},
                     {"key": "t3", "description": f"Summarize results for: {description}", "depends_on": ["t2"]}]
        key_to_id, ids = {}, []
        for t in tasks:
            deps = [key_to_id[k] for k in t.get("depends_on", []) if k in key_to_id]
            tid = self.graph.add_task(str(t.get("description", "")), goal_id, deps, status="planning")
            key_to_id[t.get("key", tid)] = tid
            ids.append(tid)
            self.graph.q("MATCH (a:Agent {id:$a}), (t:Task {id:$t}) MERGE (a)-[:PRODUCED]->(t)",
                         {"a": self.id, "t": tid})
        for tid in ids:  # release only once every dependency edge exists
            self.graph.set_task_status(tid, "pending")
        self.log(f"wrote {len(ids)} tasks to graph")
        return ids


class SafetyVerifierAgent(BaseAgent):
    role = "safety_verifier"

    def step(self) -> bool:
        did = False
        self.graph.release_stale_locks()
        # Tasks that depend on a blocked task can never run.
        self.graph.q("MATCH (t:Task)-[:DEPENDS_ON]->(b:Task {status:'blocked'}) "
                     "WHERE t.status IN ['pending','verified'] SET t.status='blocked'")
        # Verify every pending task (even if deps unfinished) so violations surface early.
        pending = self.graph.get_tasks_by_status("pending", 10)
        for t in pending:
            if not self.graph.lock_task(t["id"], self.id, "pending", "verifying"):
                continue
            did = True
            violated = self.check(t)
            if violated:
                for c in violated:
                    self.graph.add_violation(t["id"], c["id"])
                self.graph.set_task_status(t["id"], "blocked")
                self.graph.add_observation(f"BLOCKED task '{t['description']}' - violates: "
                                           + "; ".join(c["rule"] for c in violated), self.id, t["id"])
                self.log(f"BLOCKED {t['id']} violates {[c['id'] for c in violated]}")
            else:
                self.graph.set_task_status(t["id"], "verified")
                self.log(f"approved {t['id']}")
        return did

    @staticmethod
    def _stem(word: str) -> str:
        """Light stem so 'delete' also catches 'deletion', 'deleting', 'deleted'."""
        w = word.lower().strip()
        for suf in ("ing", "ion", "ed", "es", "s", "e"):
            if len(w) - len(suf) >= 4 and w.endswith(suf):
                return w[:-len(suf)]
        return w

    def check(self, task: dict) -> List[dict]:
        desc = task["description"].lower()
        violated = []
        for c in self.graph.get_constraints():
            if c["type"] == "forbidden_keyword":
                if self._stem(c["rule"]) in desc:
                    violated.append(c)
            elif c["type"] == "policy":
                try:
                    ans = self.llm.generate(
                        f"Policy: {c['rule']}\nTask: {task['description']}\n"
                        "Does the task violate the policy? Answer only YES or NO.",
                        "You are a strict safety auditor.")
                    if ans.strip().upper().startswith("YES"):
                        violated.append(c)
                except Exception as e:
                    self.log(f"policy check failed open->closed ({e}); blocking")
                    violated.append(c)  # fail closed
        return violated


class ExecutorAgent(BaseAgent):
    role = "executor"

    def step(self) -> bool:
        # Only run tasks the SafetyVerifier has cleared and whose dependencies are done.
        ready = self.graph.q(
            "MATCH (t:Task {status:'verified'}) "
            "OPTIONAL MATCH (t)-[:DEPENDS_ON]->(d:Task) WHERE d.status <> 'done' "
            "WITH t, count(d) AS open WHERE open = 0 "
            "RETURN t.id, t.description LIMIT 5").result_set
        for tid, desc in ready:
            if not self.graph.lock_task(tid, self.id, "verified", "locked"):
                continue
            self.log(f"locked {tid}: {desc}")
            result = self.mock_action(desc)
            self.graph.add_observation(result, self.id, tid)
            self.graph.set_task_status(tid, "done")
            self.log(f"done {tid}")
            return True
        return False

    def mock_action(self, desc: str) -> str:
        time.sleep(random.uniform(0.5, 1.5))
        kind = random.choice(["API call returned 200 OK", "Document retrieved (3 pages)",
                              "Search returned 12 relevant results"])
        return f"Executed '{desc}': {kind}."


class MemoryCuratorAgent(BaseAgent):
    role = "memory_curator"

    SYSTEM = ("You curate an agent memory. Given numbered observations and existing beliefs, reply with ONLY JSON: "
              '{"new_beliefs":[{"statement":"...","confidence":0.0-1.0,"supported_by":[obs numbers]}],'
              '"supports":[{"belief_id":"...","obs":[numbers]}],'
              '"contradicts":[{"belief_id":"...","obs":[numbers]}]}. '
              "Only report a contradiction if an observation clearly conflicts with the belief.")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_ts = 0.0

    def step(self) -> bool:
        obs = self.graph.get_recent_observations(limit=8, since=self.last_ts)
        if len(obs) < 2:
            return False
        obs = list(reversed(obs))
        self.last_ts = max(o["timestamp"] for o in obs)
        beliefs = self.graph.get_beliefs()
        prompt = ("Observations:\n" + "\n".join(f"{i}. {o['content']}" for i, o in enumerate(obs)) +
                  "\n\nExisting beliefs:\n" + ("\n".join(f"{b['id']}: {b['statement']} (conf {b['confidence']})"
                                                         for b in beliefs) or "none"))
        try:
            data = extract_json(self.llm.generate(prompt, self.SYSTEM))
        except Exception as e:
            self.log(f"curation failed: {e}")
            return False
        known = {b["id"]: b for b in beliefs}

        def pick(idx):
            return [obs[i] for i in idx if isinstance(i, int) and 0 <= i < len(obs)]

        for nb in data.get("new_beliefs", []):
            bid = self.graph.add_belief(nb["statement"], nb.get("confidence", 0.6), self.id)
            for o in pick(nb.get("supported_by", [])):
                self.graph.link_observation(o["id"], bid, True)
            self.log(f"new belief: {nb['statement']}")
        for s in data.get("supports", []):
            b = known.get(s.get("belief_id"))
            if b:
                for o in pick(s.get("obs", [])):
                    self.graph.link_observation(o["id"], b["id"], True)
                self.graph.update_belief_confidence(b["id"], b["confidence"] + 0.1)
        for c in data.get("contradicts", []):
            b = known.get(c.get("belief_id"))
            if b:
                for o in pick(c.get("obs", [])):
                    self.graph.link_observation(o["id"], b["id"], False)
                self.graph.update_belief_confidence(b["id"], b["confidence"] - 0.25)
                self.log(f"contradiction on belief {b['id']}: confidence lowered")
        return True
