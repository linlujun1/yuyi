from llm_service.model_service import ModelService


with ModelService("Qwen2.5-14B-Instruct") as service:
    print("BASE_URL =", service.base_url)
    raise RuntimeError("故意测试异常清理")