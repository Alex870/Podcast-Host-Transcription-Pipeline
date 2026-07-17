"""Local implementation of the ecosystem canonical identity rules."""
import hashlib,json
from typing import Any
def canonical_hash(value:Any)->str: return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
