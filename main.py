"""SafeCortex orchestrator: runs all agents concurrently against the FalkorDB blackboard."""
import asyncio
import logging
import sys

from agents import ExecutorAgent, MemoryCuratorAgent, PlannerAgent, SafetyVerifierAgent
from graph_client import GraphClient
from llm_router import LLMRouter


def build_agents(graph: GraphClient, llm: LLMRouter):
    return [
        PlannerAgent("planner-1", graph, llm, interval=2),
        SafetyVerifierAgent("safety-1", graph, llm, interval=1.5),
        ExecutorAgent("executor-1", graph, llm, interval=1.5),
        ExecutorAgent("executor-2", graph, llm, interval=1.7),
        MemoryCuratorAgent("curator-1", graph, llm, interval=8),
    ]


async def main(seed_goal: str = ""):
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
    graph = GraphClient().connect()
    llm = LLMRouter()
    if not graph.get_constraints():
        graph.add_constraint("delete", "forbidden_keyword", "c-no-delete")
        graph.add_constraint("password", "forbidden_keyword", "c-no-secrets")
    if seed_goal:
        graph.add_goal(seed_goal)
    agents = build_agents(graph, llm)
    print("SafeCortex started with agents:", [a.id for a in agents], flush=True)
    await asyncio.gather(*(a.run() for a in agents))


if __name__ == "__main__":
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        print("stopped")
