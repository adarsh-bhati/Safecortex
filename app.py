"""SafeCortex Streamlit UI: submit goals, watch the FalkorDB blackboard evolve."""
import asyncio
import threading
import time

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from streamlit_autorefresh import st_autorefresh

from graph_client import GraphClient
from llm_router import LLMRouter
from main import build_agents

st.set_page_config(page_title="SafeCortex", layout="wide")

COLORS = {"Task": "#3b82f6", "Observation": "#22c55e", "Belief": "#eab308", "Constraint": "#ef4444",
          "Goal": "#a855f7", "Agent": "#9ca3af"}


@st.cache_resource
def start_system():
    graph = GraphClient().connect()
    if not graph.get_constraints():
        graph.add_constraint("delete", "forbidden_keyword", "c-no-delete")
        graph.add_constraint("password", "forbidden_keyword", "c-no-secrets")
    agents = build_agents(graph, LLMRouter())

    def runner():
        async def go():
            await asyncio.gather(*(a.run() for a in agents))
        asyncio.run(go())

    threading.Thread(target=runner, daemon=True).start()
    return GraphClient().connect()  # separate connection for the UI thread


graph = start_system()
st_autorefresh(interval=2000, key="live_refresh")

blocked = graph.get_tasks_by_status("blocked", 50)
if blocked:
    st.error(f"🚨 SAFETY ALERT: {len(blocked)} task(s) BLOCKED by the Safety Verifier for violating constraints!")


def render_graph(height: int = 650):
    data = graph.get_graph_data_for_vis()
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    for n in data["nodes"]:
        label = str(n["label"])
        short = label if len(label) <= 28 else label[:25] + "..."
        color = COLORS.get(n["type"], "#999")
        if n["type"] == "Task" and n["status"] == "blocked":
            color = "#7f1d1d"
        net.add_node(n["id"], label=short, title=f"{n['type']} [{n['status'] or ''}]\n{label}",
                     color=color, shape="dot" if n["type"] != "Agent" else "triangle",
                     size=22 if n["type"] in ("Goal", "Task") else 16)
    for e in data["edges"]:
        net.add_edge(e["source"], e["target"], label=e["type"], font={"size": 8, "color": "#aaa"})
    net.set_options('{"physics":{"solver":"forceAtlas2Based"},"edges":{"arrows":{"to":{"enabled":true}}}}')
    components.html(net.generate_html(), height=height + 20, scrolling=False)


with st.sidebar:
    st.title("🧠 SafeCortex")
    goal = st.text_area("Goal", placeholder="e.g. Research competitors for a coffee startup")
    if st.button("Submit", type="primary") and goal.strip():
        gid = graph.add_goal(goal.strip())
        graph.q("MATCH (g:Goal {id:$id}) SET g.created_at=$ts", {"id": gid, "ts": time.time()})
        st.success("Goal written to graph")
    if st.button("🧹 Clear old tasks"):
        # Wipe stale run data (keeps agents + constraints) so old blocked tasks stop triggering the banner.
        graph.q("MATCH (n) WHERE n:Task OR n:Goal OR n:Observation OR n:Belief DETACH DELETE n")
        st.rerun()
    st.subheader("Constraints")
    new_rule = st.text_input("Forbidden keyword")
    if st.button("Add constraint") and new_rule.strip():
        graph.add_constraint(new_rule.strip(), "forbidden_keyword")
    st.subheader("Agents")
    icons = {"idle": "🟡", "working": "🟢", "active": "🟢", "error": "🔴"}
    for a in graph.get_agents():
        st.write(f"{icons.get(a['status'], '⚪')} **{a['id']}** — {a['role']} ({a['status']})")

st.header("Blackboard graph")
st.caption("🔵 Task  🟢 Observation  🟡 Belief  🔴 Constraint  🟣 Goal  ▲ Agent  (dark red = blocked task)")

render_graph()
st.subheader("Thought process")
for o in graph.get_recent_observations(12):
    ts = time.strftime("%H:%M:%S", time.localtime(o["timestamp"]))
    st.markdown(f"`{ts}` **{o['agent'] or 'system'}** — {o['content']}")
beliefs = graph.get_beliefs()
if beliefs:
    st.subheader("Beliefs")
    for b in beliefs:
        st.progress(min(max(float(b["confidence"] or 0), 0.0), 1.0), text=b["statement"])
