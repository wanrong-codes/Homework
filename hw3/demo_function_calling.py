from agent.function_calling_agent import FunctionCallingAgent
from agent.llm_client import get_llm_client
from agent.tools import CalculatorTool, SearchTool

if __name__ == "__main__":
    llm = get_llm_client()  
    agent = FunctionCallingAgent(llm, tools=[CalculatorTool(), SearchTool()])

    questions = [
        "帮我算一下 (128 + 72) * 3 / 5 等于多少",
        "搜索一下 Eiffel Tower 的基本信息",
        "今天天气怎么样？",  
    ]
    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.run(q)
        print(f"助手: {answer}")
