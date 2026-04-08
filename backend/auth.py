"""
Simple JWT authentication for Logistics Document AI.

Two hardcoded users — no database required.
Passwords are hashed with sha256_crypt (passlib).
Tokens are signed HS256 JWTs with a 8-hour expiry.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ───────────────────────────────────────────────────────────────────

# Secret key for signing JWTs.
# Override via JWT_SECRET env var in production.
_SECRET_KEY = os.environ.get("JWT_SECRET", "logistics-ai-dev-secret-change-in-prod")
_ALGORITHM  = "HS256"
_TOKEN_TTL  = timedelta(hours=8)

pwd_ctx = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# ── User store ────────────────────────────────────────────────────────────────

# Passwords hashed with sha256_crypt.
# To regenerate:  python -c "from passlib.context import CryptContext; print(CryptContext(['sha256_crypt']).hash('yourpassword'))"
_USERS = {
    "arshika": {
        "username":      "arshika",
        "display_name":  "Arshika",
        "password_hash": "$5$rounds=535000$Kqe3i5uJkmNk3Xgx$6yOlWjy83BeK6oJQg0sHc7vRmiTzrK8OOaAh/aMZkw4",
    },
    "testuser": {
        "username":      "testuser",
        "display_name":  "Test User",
        "password_hash": "$5$rounds=535000$ZIYNOSrCPdd15uMn$v7vsO2SxRHDBhtMSsISsud/OkD9mWtAFh9rz47tXPw5",
    },
}

# ── OAuth2 scheme ─────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ── Helpers ───────────────────────────────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Return user dict if credentials are valid, else None."""
    user = _USERS.get(username.lower())
    if not user:
        return None
    if not pwd_ctx.verify(password, user["password_hash"]):
        return None
    return user


def create_access_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + _TOKEN_TTL,
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency — validates JWT and returns the user dict."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = _USERS.get(username)
    if not user:
        raise credentials_exc
    return user
