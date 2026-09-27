"""Input and output guardrails.

``pii``, ``injection`` and ``intent`` are deliberately **dependency-light**: they
import nothing from ``app.pipeline`` and nothing third-party, so each guard can
be tested on its own with no index, no model, and no LLM. ``intent`` does read
``app.sources`` for the allowlist scheme names and their aliases, which is pure
corpus data with no pipeline dependency — the gate needs to know which five
schemes exist in order to notice when a sixth is asked about.

The composition happens in exactly one place — ``app.pipeline.orchestrator`` and
``app.pipeline.validators``, which import these guards but are never imported
*by* them. That is a narrowing of the literal "must not import each other" rule
in ``app/models.py``, adopted because the alternative (a pipeline that cannot
call the guards, or guards that import the pipeline and form a cycle) is worse:
each guard still has no reverse dependency, and no guard can reach the LLM.
"""
