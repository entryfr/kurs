import uvicorn


if __name__ == "__main__":
    # По умолчанию поднимаем новую реализацию V2 (картинка + запрос).
    uvicorn.run("src.ai_agent_course.web:app", host="0.0.0.0", port=8000, reload=False)
