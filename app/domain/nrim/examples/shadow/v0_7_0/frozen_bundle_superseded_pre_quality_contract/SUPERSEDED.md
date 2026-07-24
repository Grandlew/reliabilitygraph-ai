# Superseded bundle

This bundle is retained only as the signed lineage parent of the current
candidate. It is not a valid serving bundle for the current source tree:
the original strict telemetry contract omitted the observation-quality
category required by the frozen `low_quality_fraction` features.

Use `../frozen_bundle/`. Its signed provenance records the parent manifest,
the exact two corrected source paths, unchanged model/policy artifacts, and
that no locked-test data was read during reissue.
