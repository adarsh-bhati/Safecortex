# 🧠 SafeCortex

## Demo Video

[![🎥 Watch the Demo Video](https://img.shields.io/badge/🎥_Watch_the_Demo_Video-SafeCortex-red?style=for-the-badge)](https://drive.google.com/drive/folders/1-K2r6mQ7fViXbXg-ier5qJU4i1BIJT8N?usp=sharing)

▶️ [Click here to play the video](https://drive.google.com/drive/folders/1-K2r6mQ7fViXbXg-ier5qJU4i1BIJT8N?usp=sharing)

**A multi-agent system where the graph is the memory, the coordinator, and the safety layer.**
Built for the *Graph Hacks* hackathon (WeMakeDevs × FalkorDB).

Most multi-agent frameworks pass text between agents and hope nothing is lost or invented. SafeCortex takes the opposite approach: agents never talk to each other. They read and write a shared **FalkorDB knowledge graph** (the *blackboard*), so every task, observation, belief, rule, and violation is a queryable, auditable node or edge.

---

## Features

### 🗂️ Blackboard pattern
- Four independent agents (Planner, SafetyVerifier, Executor, MemoryCurator) coordinate **only** through FalkorDB.
- Swap, add, or scale agents (we run two executors in parallel) without changing any other component.
- Every state change is a graph mutation, so the live graph is always the true system state.
- Atomic task locking via a single compare-and-set Cypher query: no two executors can claim the same task. Stale locks are released after a timeout.

### 🛡️ Safety Verifier
- Safety rules are `Constraint` nodes in the graph, not prompt text that can be truncated or ignored.
- Every task passes `pending → verifying → verified` **before** an executor may touch it; the executor only accepts `verified` tasks, making unsafe execution structurally impossible.
- Violations are first-class: a `(Task)-[:VIOLATES]->(Constraint)` edge records exactly *what* was blocked and *why*. Tasks that depend on a blocked task are blocked too.
- Two constraint types: deterministic `forbidden_keyword` and LLM-judged `policy`. If a policy check fails, the verifier **fails closed**.

### 🧩 Memory Curator (anti-hallucination)
- Distills observations into `Belief` nodes with confidence scores and `SUPPORTS` evidence edges.
- A conflicting observation creates a `CONTRADICTS` edge and lowers the belief's confidence, so weak or disputed claims are visible rather than silently trusted.
- Observations store sentence-transformer embeddings for future semantic retrieval.

### 🔀 LLM Router
- Ollama is optional and skipped unless `OLLAMA_ENABLED=true` is set in `.env`.
- The router now attempts openai/gpt-oss-120b (Groq) and gemini-2.5-flash (Gemini) first. The older Llama 3.3 and Gemini 1.5 Flash models, which now return 404 due to API deprecations, are kept at the end of each fallback list.
- Exponential backoff on rate limits and timeouts.
- The planner has an offline fallback plan if all providers fail.

### 📊 Live UI
- Streamlit dashboard with a real-time PyVis graph: Tasks 🔵, Observations 🟢, Beliefs 🟡, Constraints 🔴, Goals 🟣, Agents ▲ (blocked tasks in dark red).
- "Thought process" feed of recent observations, belief confidence bars, agent status indicators, and live constraint editing.

---

## Architecture

```
                  ┌────────────────────────────────────┐
   User Goal ───▶ │          FalkorDB  (Blackboard)    │ ◀─── Streamlit UI
                  │                                    │      (PyVis graph,
                  │  Goal ─HAS_TASK▶ Task ─DEPENDS_ON▶ Task    thought feed)
                  │  Task ─ASSIGNED_TO▶ Agent          │
                  │  Agent ─PRODUCED▶ Observation/Task │
                  │  Observation ─SUPPORTS/CONTRADICTS▶ Belief
                  │  Task ─VIOLATES▶ Constraint        │
                  └──▲───────▲────────▲────────▲───────┘
                     │       │        │        │      (read / write only)
               ┌─────┴──┐ ┌──┴─────┐ ┌┴───────┐ ┌┴──────────┐
               │Planner │ │ Safety │ │Executor│ │  Memory   │
               │        │ │Verifier│ │  (×2)  │ │  Curator  │
               └────┬───┘ └───┬────┘ └────────┘ └─────┬─────┘
                    └─────────┴───────────┬───────────┘
                                          ▼
               ┌────────────────────────────────────────────────────┐
               │ LLM Router                                         │
               │ Groq (primary) → Gemini (fallback) →               │
               │ Ollama (optional, if OLLAMA_ENABLED=true)          │
               └────────────────────────────────────────────────────┘
```

**Task lifecycle:** `pending → verifying → verified → locked → done` (or `blocked`).

| Agent | Reads | Writes |
|---|---|---|
| Planner | Active `Goal`s, open task count | `Task` nodes + `DEPENDS_ON` edges |
| SafetyVerifier | `pending` tasks, `Constraint`s | `verified`/`blocked` status, `VIOLATES` edges |
| Executor | `verified` tasks with all dependencies `done` | `Observation` nodes, `done` status |
| MemoryCurator | Recent `Observation`s, existing `Belief`s | `Belief` nodes, `SUPPORTS`/`CONTRADICTS` edges, confidence updates |

**Graph schema:** nodes `Goal`, `Task`, `Agent`, `Observation`, `Belief`, `Constraint`; edges `DEPENDS_ON`, `ASSIGNED_TO`, `PRODUCED`, `SUPPORTS`, `CONTRADICTS`, `VIOLATES` (plus `HAS_TASK` and `YIELDED` for goal/observation linkage). Indexes on `Task(status)` and `Observation(timestamp)`.

### Project layout

| File | Purpose |
|---|---|
| `graph_client.py` | FalkorDB client: schema, atomic locking, all Cypher |
| `llm_router.py` | Multi-provider LLM routing with backoff |
| `agents.py` | `BaseAgent` + Planner, SafetyVerifier, Executor, MemoryCurator |
| `main.py` | Async orchestrator (CLI) |
| `app.py` | Streamlit UI (also launches the agents) |
| `test_populate.py` | Smoke test: dummy data, lock test, graph export |
| `blog_post.md` | Technical write-up |

---

## Setup

**Prerequisites:** Python 3.11+, Docker, and a Groq and/or Gemini API key (Ollama is optional).

1. **Start FalkorDB**
   ```bash
   docker run -p 6379:6379 -it --rm falkordb/falkordb
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   (`sentence-transformers` pulls in PyTorch, so the first install is large.)
3. **Configure keys** by creating a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_key
   GEMINI_API_KEY=your_gemini_key
   ```
   Optional: run [Ollama](https://ollama.com) with `llama3.2` as a last-resort local fallback.
   Ollama is skipped by default unless `OLLAMA_ENABLED=true` is set in the `.env` file.

---

## How to run

**Web UI (recommended)**: launches the agents and the dashboard together:
```bash
streamlit run app.py
```
Enter a goal in the sidebar and click **Submit**. Watch tasks appear, get verified, execute, and turn into observations and beliefs. Add a forbidden keyword in the sidebar to see the Safety Verifier block matching tasks.

The page auto-refreshes every 2 seconds, so the graph updates live. When the Safety Verifier blocks a task, a red safety violation banner appears at the top of the UI.

**Headless CLI**: console log with timestamps and agent IDs:
```bash
python main.py "Plan a coffee shop product launch"
```

**Smoke test**: populates dummy data in a separate graph, checks lock exclusivity, and writes `graph_preview.html`:
```bash
python test_populate.py
```

Set `SAFECORTEX_EMBEDDINGS=0` to skip loading the sentence-transformer model (uses a cheap hash vector instead).

### Try the safety demo
1. Submit: `Plan a product launch and delete old customer data`.
2. The default `delete` constraint blocks the matching tasks; the graph shows `VIOLATES` edges to the red constraint node and nothing reaches the executor.

---

## Limitations & next steps
- The executor performs mock actions; plug in real tools behind the same verifier.
- Contradiction detection is LLM-based today; stored embeddings enable semantic detection next.
- Human-in-the-loop approval can be added as just another agent writing to the graph.

## Read more
See [`blog_post.md`](blog_post.md): *Why Graph Memory is the Key to Safe Multi-Agent Systems*.

Word count (`wc -w blog_post.md`):
```
1740 blog_post.md
```
