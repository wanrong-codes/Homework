from agent.llm_client import get_llm_client
from agent.react_agent import ReActAgent
from agent.tools import default_toolset

if __name__ == "__main__":
    llm = get_llm_client()
    agent = ReActAgent(llm, tools=default_toolset())

    question = "帮我查一下 Tokyo 的天气，再算一下 350 * 5 等于多少（这是我 5 天的预算总额）"
    print(f"Question: {question}\n")
    final_answer = agent.run(question)
    print(f"\nFinal Answer: {final_answer}")
