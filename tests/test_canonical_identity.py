import json,unittest
from pathlib import Path
from podcast_transcribe.canonical_identity import canonical_hash
class IdentityTests(unittest.TestCase):
 def test_published_vector(self):
  v=json.loads((Path(__file__).parent/"fixtures/ecosystem/canonical-identity-v1.json").read_text(encoding="utf-8")); self.assertEqual(v["expected_sha256"],canonical_hash(v["payload"])); self.assertEqual(v["expected_sha256"],canonical_hash(dict(reversed(list(v["payload"].items())))))
if __name__=="__main__": unittest.main()
