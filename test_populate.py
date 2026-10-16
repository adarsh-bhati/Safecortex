"""Populate dummy data, exercise locking, and render the graph to graph_preview.html."""
import os
os.environ["SAFECORTEX_EMBEDDINGS"] = "0"  # fast hash embeddings for the smoke test

from pyvis.network import Network

from graph_client import GraphClient

g = GraphClient(graph_name="safecortex_test").connect()
g.clear()
for aid, role in (("planner-1", "planner"), ("executor-1", "executor"), ("executor-2", "executor"),
                  ("safety-1", "safety_verifier"), ("curator-1", "memory_curator")):
    g.register_agent(aid, role)

goal = g.add_goal("Research coffee market")
t1 = g.add_task("Search market reports", goal)
t2 = g.add_task("Analyze competitors", goal, [t1])
t3 = g.add_task("Delete old records", goal, [t1])
c = g.add_constraint("delete", "forbidden_keyword")
g.add_violation(t3, c)
g.set_task_status(t3, "blocked")

assert [t["id"] for t in g.get_pending_tasks()] == [t1], "only dependency-free task should be ready"
assert g.lock_task(t1, "executor-1") is True
assert g.lock_task(t1, "executor-2") is False, "second lock must fail"
o1 = g.add_observation("Market grew 8% in 2025", "executor-1", t1)
o2 = g.add_observation("Market shrank 3% in 2025", "executor-2", t1)
b = g.add_belief("Coffee market is growing", 0.7, "curator-1")
g.link_observation(o1, b, True)
g.link_observation(o2, b, False)
g.update_belief_confidence(b, 0.45)

data = g.get_graph_data_for_vis()
print(len(data["nodes"]), "nodes,", len(data["edges"]), "edges")
colors = {"Task": "#3b82f6", "Observation": "#22c55e", "Belief": "#eab308", "Constraint": "#ef4444",
          "Goal": "#a855f7", "Agent": "#9ca3af"}
net = Network(height="700px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
for n in data["nodes"]:
    net.add_node(n["id"], label=str(n["label"])[:25], color=colors[n["type"]])
for e in data["edges"]:
    net.add_edge(e["source"], e["target"], label=e["type"])
net.write_html("graph_preview.html")
print("OK: wrote graph_preview.html")
