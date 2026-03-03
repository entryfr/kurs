import uvicorn
from pathlib import Path

from src.ai_agent_course.secrets_loader import load_local_env


if __name__ == "__main__":
    load_local_env(Path(__file__).resolve().parent / ".env.local")
    # По умолчанию поднимаем новую реализацию V2 (картинка + запрос).
    uvicorn.run("src.ai_agent_course.web:app", host="0.0.0.0", port=8000, reload=False)
