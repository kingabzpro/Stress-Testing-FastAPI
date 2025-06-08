import uvicorn
import multiprocessing
import os

if __name__ == "__main__":
    # Optimize based on available resources
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 4)  # Limit workers for stability

    # Set environment variables for better performance
    os.environ["PYTHONUNBUFFERED"] = "1"

    config = uvicorn.Config(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
    )

    server = uvicorn.Server(config)
    server.run()
