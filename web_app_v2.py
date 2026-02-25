import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.ai_agent_course.web:app", host="0.0.0.0", port=8010, reload=False)

