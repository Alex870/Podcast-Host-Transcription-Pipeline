# correction-manifest-v1

The transcription repository owns this additive ecosystem contract. UTF-8 JSON is
canonicalized with sorted keys and compact separators. The immutable identity
contains the contract version, producer, source/result transcript hashes,
reviewer pseudonym, accepted corrections, source span IDs, before/after values,
adjudication state, and reason codes. `notes`, display labels, and UI state are
mutable and excluded. Generated or rejected suggestions are never authoritative.

Consumers must reject unknown major versions, stale source hashes, identity
mismatches, and corrections whose `before` value differs from the referenced
transcript. Existing transcript and reviewed-transcript readers are unchanged.
