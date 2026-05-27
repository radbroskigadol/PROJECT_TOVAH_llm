# LICENSE_OPTIONS.md — Commercial / Research License Options Memo

This is not legal advice and is not a final license agreement. It is a transaction memo describing plausible licensing structures for TOVAH v14.2.6.

The package metadata currently marks the project as **Proprietary**. No open-source license is granted by default unless a separate written agreement says otherwise.

## Option A — Evaluation license

Purpose: let a serious buyer or lab inspect and test the code privately.

Suggested terms:

```text
non-commercial evaluation only
no redistribution
no public posting of source
no derivative commercial use
limited evaluation window
confidentiality required
```

Typical price range:

```text
free under NDA to serious buyer, or low paid evaluation fee
```

## Option B — Nonexclusive research license

Purpose: allow a lab/startup to use the code internally while the author retains ownership and can license elsewhere.

Suggested terms:

```text
nonexclusive
source access
internal research and prototyping rights
no resale or sublicensing without permission
attribution / provenance clause
optional support hours
```

Potential pricing anchor:

```text
$25k-$100k depending on scope, support, exclusivity carveouts, and buyer type
```

## Option C — Commercial product license

Purpose: allow use inside a commercial product without transferring IP ownership.

Suggested terms:

```text
nonexclusive or field-limited exclusive
commercial deployment rights
no source redistribution
support/security obligations negotiated separately
royalty or revenue-share possible
```

Potential pricing anchor:

```text
$75k-$250k+ depending on field, term, support, and deployment rights
```

## Option D — Exclusive source/IP license

Purpose: give one buyer broad exclusive rights while author may or may not retain attribution/research-use rights.

Suggested terms:

```text
exclusive field or global exclusivity
source access
assignment or exclusive license of relevant copyrights/trade-secret materials
clear treatment of future inventions
clear treatment of author research rights
warranty disclaimers
```

Potential pricing anchor:

```text
$250k-$750k+ depending on exclusivity and buyer strategic need
```

## Option E — Full IP transfer

Purpose: buyer acquires the code and associated project IP outright.

Suggested terms:

```text
assignment agreement
representations about authorship/provenance
dependency schedule
warranty disclaimer
transition/handoff support
possible consulting agreement
```

Potential pricing anchor:

```text
$500k+ recommended floor for serious strategic buyer, unless seller wants a quick exit
```

## Recommended default public posture

For public X/Twitter language, keep it flexible:

```text
Considering sale or license of a research-grade autonomous AI kernel. Asking $75k for serious buyers. Full source and test suite available; architecture notes and handoff materials provided privately.
```

Do not publicly post the tarball. Provide source only after buyer screening and agreement on evaluation terms.

## Dependency note

The project depends on third-party packages such as PyTorch and optional OpenAI/python-dotenv/pypdf/pytest extras. A buyer should review each dependency's license separately before commercial deployment.
