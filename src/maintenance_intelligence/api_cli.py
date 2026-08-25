from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the FastAPI application with portfolio-friendly defaults."""
    uvicorn.run(
        "maintenance_intelligence.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
