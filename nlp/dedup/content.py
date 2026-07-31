"""M2's own canonical content representation and fingerprint.

This module exists so that M2's notion of "the same text" is owned by M2.
It deliberately does **not** call
:func:`nlp.embeddings.compose_embedding_text` or
:func:`nlp.embeddings.input_fingerprint`: those define the *encoder's* input
composition, and M1 is free to change them (add a source prefix, change the
separator, truncate) for encoder reasons.  If M2 borrowed them, such a
change would silently move every dedup identity key and quietly alter which
articles merge.  Nothing in :mod:`nlp.dedup` imports :mod:`nlp.embeddings`,
so importing and running M2 never loads an embedding model.
"""

from __future__ import annotations

import hashlib

#: Bumped whenever the canonical content string changes shape.  It is part
#: of the configuration fingerprint, so a bump invalidates cached output.
DEDUP_CONTENT_VERSION = "m2.content.v1"

_FIELD_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"


def canonicalize_dedup_content(
    normalized_title: str, normalized_description: str
) -> str:
    """Return the versioned canonical content string for one item.

    Both inputs are already-normalized keys (see
    :func:`nlp.dedup.structural.analyze_title` and
    :func:`nlp.dedup.text.text_key`).  The separators are control
    characters so no title or description can forge a field boundary.
    """

    return _RECORD_SEPARATOR.join(
        (
            DEDUP_CONTENT_VERSION,
            normalized_title + _FIELD_SEPARATOR + normalized_description,
        )
    )


def dedup_content_fingerprint(
    normalized_title: str, normalized_description: str
) -> str | None:
    """Return the SHA-256 fingerprint of one item's text, or ``None``.

    ``None`` means the item carries no usable text at all, in which case no
    content claim can be made about it.
    """

    if not normalized_title and not normalized_description:
        return None
    canonical = canonicalize_dedup_content(normalized_title, normalized_description)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_identity_key(
    normalized_title: str, normalized_description: str
) -> str | None:
    """Return the tier-3 "identical article text" key, or ``None``.

    A key is only produced when the item has *both* a title and a
    description: without body text a "content hash" is just a title hash,
    and title identity is a weaker signal that lives in tier 4.
    """

    if not normalized_title or not normalized_description:
        return None
    return dedup_content_fingerprint(normalized_title, normalized_description)
