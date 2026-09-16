from agent.llm_client import get_llm_client
from agent.memory import MemoryStore
from agent.skills import TravelPlanningSkill

if __name__ == "__main__":
    llm = get_llm_client()
    memory = MemoryStore(path="demo_memory_store.json")

    user_id = "user_wanrong"
    memory.add_preference(user_id, "travel_style", "经济型，偏好当地小吃而非高档餐厅")
    memory.add_preference(user_id, "preferred_currency", "CNY")

    print("已存储的用户偏好:")
    print(memory.format_for_prompt(user_id))

    skill = TravelPlanningSkill(llm, memory=memory)
    result = skill.plan_trip(user_id=user_id, destination="Tokyo", days=5, budget=5000)

    print("\n=== 旅行计划 ===")
    print(result)
