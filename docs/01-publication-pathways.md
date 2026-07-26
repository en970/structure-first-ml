# Publication pathways for an unaffiliated researcher

*Compiled 26 July 2026 from the primary author-guideline pages. Policies change; re-check before
submitting. Marked **[verified]** where the source page was fetched during the scan.*

The purpose of this note is to fix, in advance, what artefact each pathway actually requires, so
that the work is shaped toward a submission rather than retrofitted to one afterwards.

---

## Research Notes of the AAS (RNAAS)

**[verified]** — journals.aas.org author guidelines.

- Limit: 1,500 words including a 150-word abstract. One figure **or** one table, not both.
- No publication charge.
- Not peer reviewed; editor-moderated for scope and format.
- Accepted notes appear within roughly 72 hours.
- No affiliation requirement is stated. Independent authors are not barred.
- Scope explicitly welcomes works in progress, null results, comments, and results that would not
  merit a full paper — a single candidate detection is the canonical example given.
- RNAAS does not publish substantially novel theory. A method note must therefore lead with the
  measurement, not the theorem.

**Realistic bar: low.** This is the natural first destination for a candidate list or a compact
methodological result. The failure mode is an editor judging the note insufficiently astronomically
interesting, which argues for leading with the astrophysical consequence rather than the algorithm.

## arXiv

**[verified]** — info.arxiv.org endorsement help page and the arXiv blog post of 2026-01-21.

- A first submission to a category requires endorsement by someone who has published in that
  category within the last five years and holds active endorsement status.
- As of 21 January 2026, arXiv **no longer grants automatic endorsement from an institutional email
  address**. This tightening applies to everyone, which removes the structural disadvantage an
  unaffiliated author previously faced relative to a newly affiliated one, but it also removes the
  institutional shortcut entirely.
- Practical routes: co-author with someone active in the category, or seek endorsement after a
  published RNAAS note exists as evidence of competence.

**Realistic bar: moderate to hard without a collaborator.** Plan for it rather than assume it.

## Zenodo

**[verified]** — Zenodo GitHub integration documentation.

- Enable the repository in Zenodo, cut a public GitHub release, and a versioned DOI is minted
  automatically. A LICENSE file should be present. DOIs cannot be pre-reserved through the GitHub
  route.

**Realistic bar: trivial, and it should be done for every track that produces a catalogue or a
reusable implementation.** A DOI makes the work citable and is the cheapest form of provenance.

## Registries for candidate detections

| Registry | Accepts | Bar |
|---|---|---|
| AAVSO VSX | New variable stars | Moderate — the submitter must verify novelty against existing catalogues first |
| Transient Name Server (TNS) | Transients, classifications | Open to amateurs and professionals; requires registration |
| ExoFOP-TESS | Community TOIs (cTOIs) | Open upload; the TESS team may promote a cTOI to an official TOI |
| Minor Planet Center | Astrometry, minor-planet discoveries | High — requires demonstrated astrometric competence on known objects over multiple nights; built for observers, not archival mining |
| IAU Meteor Data Center | Meteor shower / stream candidates | Formal working-list process; the relevant pathway for T2 |
| SIMBAD | Not a submission portal — ingests published literature | n/a |

## Citizen science and pro-am routes

Zooniverse projects that vet machine-flagged candidates are a genuine, precedented route to
co-authorship: *Cosmic Cataclysms* was built to have volunteers vet machine-flagged transients in
TESS full-frame images, and Planet Hunters TESS has produced peer-reviewed discoveries with
volunteer co-authors. Across NASA-sponsored Zooniverse projects, a substantial fraction of the
resulting publications carry citizen co-authors **[reported]**. For a machine-learning pre-filter,
feeding a human-vetting project is a more realistic path to a refereed paper than submitting one
alone.

---

## What this implies for the tracks here

- **T1 (method).** The primary output is a benchmark and an implementation, which suits a Zenodo
  DOI plus a preprint. An RNAAS note is viable only if the method surfaces a specific object worth
  reporting. Aim the writing at astro-ph.IM.
- **T2 (discovery).** Candidate streams have a formal registry pathway through the IAU Meteor Data
  Center, and a compact candidate list is exactly what RNAAS was designed for.
- **T3 (theory).** Not an astronomy publication. Its natural home is the repository itself and, if
  the result is sharp enough, a workshop paper in machine learning.

In all three cases the immediate, unconditional deliverable is the same: a reproducible public
repository with a DOI, which requires no gatekeeper's permission and is the thing an eventual
collaborator or endorser will actually read.
