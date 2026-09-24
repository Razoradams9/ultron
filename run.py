"""Boot the Ultron server: python run.py"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("ultron.server:app", host="127.0.0.1", port=8000)
