# Security policy

Copyright 2026 Zale Pan.

## Experimental status

This is an experimental source candidate. It is not a production system, provides no service level
agreement, no enterprise security assurance and no guarantee of suitability for any particular
threat model. It has not been independently audited.

## Reporting a vulnerability

No dedicated security contact address is published with this candidate, and none is invented here.
Until a contact channel is published, please report a suspected vulnerability through the same
project channel you used to obtain this candidate, describing:

* the affected version or commit,
* a minimal reproduction,
* the observed and expected behaviour,
* any credentials or personal data removed from the report.

Do not include live secrets, personal data or third-party confidential material. Do not test against
systems you do not own or have permission to test.

## Scope notes

* The optional DeepSeek smoke is disabled by default and never reads a credential unless you
  explicitly select live mode and inject one locally.
* The exporter copies a manifest-declared whitelist only; it performs no upload, no Git operation
  and no network call.
* There is no operating-system sandbox and no guarantee against a hostile concurrent writer.

The whitelisted source exporter and its release tests are internal release tooling and are not
distributed with the public source set.
