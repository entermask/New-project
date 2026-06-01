import os
from typing import Optional

from fastapi import Header
from fastapi import HTTPException


API_TOKEN = os.getenv("API_TOKEN", "")


def validate_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token.")
