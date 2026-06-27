"""
Minimal API key auth. Keys are generated once, shown to the operator a single time, and
only the SHA-256 hash is ever stored -- standard practice, the same reason you never see
your own password again after setting it. Verification is a hash lookup, not a secret
comparison of plaintext, so a DB leak doesn't hand out usable keys.

Deliberately not JWT/OAuth: this is a single-operator-dashboard auth model (one key per
client/integration), not a multi-user login system with sessions -- that's a different,
bigger feature (user accounts, password reset, etc.) that wasn't asked for here.
"""
from __future__ import annotations

import hashlib
import secrets

from sqlalchemy.orm import Session

from db.models import ApiKey


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_api_key(db: Session, name: str, facility_id=None) -> str:
    """Returns the raw key -- this is the ONLY time it's available. Store it; if lost,
    revoke and create a new one, the same way you'd rotate any secret."""
    raw_key = f"isi_{secrets.token_urlsafe(32)}"
    db.add(ApiKey(key_hash=_hash(raw_key), name=name, facility_id=facility_id))
    db.commit()
    return raw_key


def verify_api_key(db: Session, raw_key: str) -> ApiKey | None:
    if not raw_key:
        return None
    key = db.query(ApiKey).filter_by(key_hash=_hash(raw_key), revoked=False).one_or_none()
    return key
