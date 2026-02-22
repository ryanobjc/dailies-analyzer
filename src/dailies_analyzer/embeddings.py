"""Local MLX embedding generation and semantic search."""

import json

import numpy as np

from .db import Database
from .extractor import get_conversation_text

MODEL_NAME = "bge-small"
EMBEDDING_DIM = 384
BATCH_SIZE = 128
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def _get_model():
    """Lazy-load the MLX embedding model."""
    global _model
    if _model is None:
        from mlx_embedding_models.embedding import EmbeddingModel

        _model = EmbeddingModel.from_registry(MODEL_NAME)
    return _model


def _get_embedding_text(db: Database, conversation_id: int) -> str:
    """Build text for embedding a conversation.

    Uses summary + topics when available (concise, fits in 512 token window).
    Falls back to raw conversation text otherwise.
    """
    row = db.conn.execute(
        """
        SELECT c.topic, s.summary, s.key_topics
        FROM conversations c
        LEFT JOIN conversation_summaries s ON c.id = s.conversation_id
        WHERE c.id = ?
        """,
        (conversation_id,),
    ).fetchone()

    if not row:
        return ""

    topic = row["topic"] or ""
    summary = row["summary"]
    key_topics = row["key_topics"]

    if summary:
        parts = [topic, summary]
        if key_topics:
            try:
                topics_list = json.loads(key_topics)
                parts.append("Topics: " + ", ".join(topics_list))
            except (json.JSONDecodeError, TypeError):
                pass
        return ". ".join(p for p in parts if p)

    # Fall back to raw conversation text for unsummarized conversations
    return get_conversation_text(db, conversation_id)


def embed_conversations(db: Database, batch_size: int = BATCH_SIZE) -> tuple[int, int]:
    """Embed conversations that don't have embeddings yet.

    Returns (embedded_count, skipped_count).
    """
    model = _get_model()

    already_embedded = db.get_embedded_conversation_ids()

    # Get all conversation IDs
    cursor = db.conn.execute("SELECT id FROM conversations ORDER BY id")
    all_ids = [row[0] for row in cursor]

    to_embed = [cid for cid in all_ids if cid not in already_embedded]

    if not to_embed:
        return 0, 0

    # Build texts for all conversations to embed
    texts = []
    valid_ids = []
    skipped = 0
    for cid in to_embed:
        text = _get_embedding_text(db, cid)
        if text.strip():
            texts.append(text)
            valid_ids.append(cid)
        else:
            skipped += 1

    if not texts:
        return 0, skipped

    # Process in batches
    embedded = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = valid_ids[i : i + batch_size]

        vectors = model.encode(batch_texts)

        for cid, vector in zip(batch_ids, vectors):
            embedding_bytes = np.array(vector, dtype=np.float32).tobytes()
            db.insert_embedding(cid, embedding_bytes, MODEL_NAME)

        embedded += len(batch_ids)
        yield embedded, len(texts)

    return embedded, skipped


def semantic_search(db: Database, query: str, limit: int = 20) -> list[dict]:
    """Search conversations by semantic similarity to query.

    Returns results in the same shape as search_conversations(), with added similarity score.
    """
    model = _get_model()

    # Embed the query with BGE retrieval prefix
    vectors = model.encode([QUERY_PREFIX + query])
    query_vec = np.array(vectors[0], dtype=np.float32)

    # Load all embeddings
    embeddings = db.get_all_embeddings()
    if not embeddings:
        return []

    conv_ids = [e[0] for e in embeddings]
    matrix = np.array([np.frombuffer(e[1], dtype=np.float32) for e in embeddings])

    # Cosine similarity (normalize both sides)
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarities = matrix_norms @ query_norm

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:limit]

    # Build results with conversation metadata
    results = []
    for idx in top_indices:
        cid = conv_ids[idx]
        score = float(similarities[idx])

        # Get conversation metadata + summary
        row = db.conn.execute(
            """
            SELECT
                c.id,
                c.topic,
                c.date,
                c.model,
                s.summary,
                s.key_topics,
                s.sentiment,
                s.outcome
            FROM conversations c
            LEFT JOIN conversation_summaries s ON c.id = s.conversation_id
            WHERE c.id = ?
            """,
            (cid,),
        ).fetchone()

        if row:
            r = dict(row)
            r["similarity"] = score
            results.append(r)

    return results
