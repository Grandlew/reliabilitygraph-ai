# NRIM v0.7 pilot preregistration template

This file is a template, not a registered pilot. Use
`app.domain.nrim.shadow.governance.default_protocol` and
`register_protocol` only after the fields below are approved. Registration is
exclusive-create and Ed25519 signed; it cannot be overwritten.

Required before registration:

- frozen bundle hash and externally stored public-key fingerprint;
- pilot start and end dates, with at least eight planned weeks;
- at least five deployment pseudonyms and five topology families;
- deployment owners and data-access approval;
- pseudonymization key owner and secret-manager reference;
- decision interval and per-metric watermark/freshness limits;
- existing monitoring burden definition;
- independent reviewer and disagreement-resolution roster;
- historical replay readiness evidence;
- emergency-stop owner and escalation channel;
- immutable evidence retention and access policy.

The registered protocol freezes:

- hypotheses and inclusion/exclusion rules;
- incident, burden, ranking, support, data-quality, runtime, and human-utility
  metrics;
- deployment/episode independent units and uncertainty calculations;
- fast/slow/dual path-retention rules;
- evidence floor of 3,000 healthy deployment-hours and 30 adjudicated
  incidents;
- all emergency-stop conditions;
- prohibition on model, feature, support, or threshold tuning during v0.7.

Pilot outcomes may be used only after protocol closure to design a
deployment-grouped and time-separated v0.8 development corpus.
