# app/controllers/search_controller.py
from flask import request, jsonify, current_app
from ..models.eeiot_model import get_frames_collection
from ..models.search_model import (
    create_beit3,
    create_clip,
    create_llm,
    create_faiss_index,
    faiss_search_results,
    topk_fusion,
)
from ..utils.helpers import *  # giả sử topk_fusion/other helper nếu cần (giữ để tương thích)
import torch
import threading

# Module-level lazy init variables
_models_initialized = False
_init_lock = threading.Lock()
_beit3 = None
_beit3_tokenizer = None
_clip = None
_clip_tokenizer = None
_clip_preprocess = None
_llm = None
_index_beit = None
_index_clip = None
_index_metadata = None  # metadata kết hợp với index_beit (get_metadata=True)
DEVICE_DEFAULT = "cpu"


def _ensure_models(device=DEVICE_DEFAULT):
    """
    Lazy initialize heavy models and FAISS indexes.
    Thread-safe basic guard with lock.
    """
    global _models_initialized, _beit3, _beit3_tokenizer, _clip, _clip_tokenizer, _clip_preprocess, _llm, _index_beit, _index_clip, _index_metadata

    if _models_initialized:
        return

    with _init_lock:
        if _models_initialized:
            return
        current_app.logger.info("Initializing models and indexes (lazy)...")

        # Device selection
        device_choice = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        try:
            # create model helpers (assume these functions exist and may download weights)
            _beit3, _beit3_tokenizer = create_beit3()
            _clip, _clip_tokenizer, _clip_preprocess = create_clip()
            _llm = create_llm()
            # create_faiss_index returns (index, metadata) when get_metadata=True
            _index_beit, _index_metadata = create_faiss_index("embedding-info", "metadata", model="beit3", get_metadata=True)
            _index_clip, _ = create_faiss_index("embedding-info", "metadata", model="clip")
        except Exception as e:
            current_app.logger.exception("Failed to initialize models/indexes: %s", e)
            # Don't raise here — let request handlers report error
            _beit3 = _beit3_tokenizer = _clip = _clip_tokenizer = _clip_preprocess = _llm = None
            _index_beit = _index_clip = _index_metadata = None

        _models_initialized = True
        current_app.logger.info("Model/index initialization complete.")


# ---------- Retrieval + fuzzy functions (kept from original, small tweaks) ----------
def retrieve(query1, index, k, augment=False, query2=None, model=None, device="cpu", llm=None):
    """
    Use FAISS + optional augmentation (LLM) to return ranked list of top-k ids (indices into metadata).
    """
    sims1, ids1 = faiss_search_results(query1, index, k, augment, model, device, llm)
    topk_ids1, topk_sims1 = topk_fusion(sims1, ids1)

    if query2 is not None:
        sims2, ids2 = faiss_search_results(query2, index, k, augment, model, device, llm)
        topk_ids2, topk_sims2 = topk_fusion(sims2, ids2)

        reranking = []
        map2 = dict(zip(topk_ids2, range(len(topk_ids2))))
        set_ids2 = set(topk_ids2)
        for i, idx1 in enumerate(topk_ids1):
            fusion_rank = topk_sims1[i]
            for offset in range(6):
                if (idx1 + offset) in set_ids2:
                    fusion_rank = max(fusion_rank, topk_sims2[map2[idx1 + offset]])
            reranking.append(fusion_rank)

        map_rerank = dict(zip(topk_ids1, reranking))
        topk_ids1 = list(sorted(topk_ids1, key=lambda x: map_rerank[x], reverse=True))

    return topk_ids1


def create_fuzzy_query(detection=None, objects=None, text=None):
    ob_query = []
    if objects:
        ob_query.append(
            {
                "text": {
                    "query": objects,
                    "path": "objects",
                    "fuzzy": {"maxEdits": 1, "prefixLength": 2},
                }
            }
        )

    od_query = []
    if detection:
        for bb in detection:
            od_query.append(
                {
                    "text": {
                        "query": bb,
                        "path": "detection",
                        "fuzzy": {"maxEdits": 1, "prefixLength": 2},
                    }
                }
            )

    text_query = None
    if text:
        text_query = {
            "text": {
                "query": text,
                "path": "text",
                "fuzzy": {"maxEdits": 1, "prefixLength": 2},
            }
        }

    return ob_query, od_query, text_query


def fuzzy_search(collection, detection=None, objects=None, operator="AND", text=None, k=100):
    """
    Uses MongoDB Atlas Search via aggregation pipeline ($search).
    Returns list of idx (indices) from documents matched.
    """
    if collection is None:
        return []

    ob_query, od_query, text_query = create_fuzzy_query(detection, objects, text)
    if ob_query == [] and od_query == [] and text_query is None:
        return []

    if od_query == []:
        od_clause = []
    else:
        od_clause = (
            [{"compound": {"must": od_query}}]
            if operator == "AND"
            else [{"compound": {"should": od_query}}]
        )

    clause = (
        []
        if ob_query == [] and od_query == []
        else [{"compound": {"must": ob_query + od_clause}}]
    )

    pipeline = [
        {"$search": {"index": "default", "compound": {"must": clause}}},
        {"$limit": k},
        {"$project": {"_id": 0, "idx": 1}},
    ]

    if text:
        pipeline[0]["$search"]["compound"]["must"].append(text_query)

    try:
        results = collection.aggregate(pipeline)
        return [res["idx"] for res in results]
    except Exception as e:
        current_app.logger.exception("fuzzy_search aggregation failed: %s", e)
        return []


def search_logic(collection, metadata, model=None, query1=None, index=None, augment=False, k=100, query2=None, llm=None, detection=None, objects=None, operator="AND", text=None, device="cpu"):
    """
    Core fusion logic between FAISS results and fuzzy_search results.
    Returns tuple (topk_indices, frame_paths)
    """
    topk = None
    frame_paths = []

    if query1 is not None and index is not None and model is not None:
        topk_faiss = retrieve(query1, index, k, augment, query2, model, device, llm)
    else:
        topk_faiss = []

    topk_fuzzy = fuzzy_search(collection, detection, objects, operator, text, k)

    if topk_fuzzy == []:
        topk = topk_faiss
    elif topk_faiss == []:
        topk = topk_fuzzy
    else:
        mapping = dict(zip(topk_fuzzy, range(len(topk_fuzzy))))
        idx_set = set(topk_fuzzy)

        reranking = []
        for i, idx in enumerate(topk_faiss):
            if idx in idx_set:
                reranking.append(1 / (1000 + i) + 1 / (1000 + mapping[idx]))
            else:
                reranking.append(1 / (1000 + i))

        topk = sorted(zip(topk_faiss, reranking), key=lambda x: x[1], reverse=True)
        topk = [idx for idx, _ in topk]

    if topk:
        # metadata is expected to be a list-like where metadata[idx] yields a dict
        try:
            frame_paths = [metadata[idx] for idx in topk]
        except Exception:
            # if metadata stored differently, try lookup by 'idx' field
            frame_paths = []
            for idx in topk:
                if isinstance(metadata, dict):
                    frame_paths.append(metadata.get(idx))
                else:
                    frame_paths.append(None)

    else:
        topk = []

    return topk, frame_paths


def temporal_search(metadata, frame_idx=-1):
    frame_paths = []
    # safe bounds
    start = max(0, frame_idx - 10)
    end = min(len(metadata), frame_idx + 10)
    for idx in range(start, end):
        frame_paths.append(metadata[idx].get("idx") if isinstance(metadata[idx], dict) else metadata[idx])
    return frame_paths


# ---------- Flask-facing controller function ----------
def search_collection():
    """
    Controller to be called by route. Reads parameters from request.json and returns JSON result.

    Expected request.json keys (all optional, sensible defaults):
    {
        "query1": "text query for model 1 (beit3/clip)",
        "query2": "optional second query for fusion/ reranking",
        "model": "beit3" or "clip",          # default: "beit3"
        "k": 100,
        "augment": false,                    # use LLM augmentation
        "detection": ["a1car","e1person"],   # list or comma separated string
        "objects": "car person",             # string
        "operator": "AND" or "OR",           # default: "AND"
        "text": "some textual filter",       # text filter for mongo fuzzy search
        "device": "cpu" or "cuda"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        # parse parameters
        query1 = data.get("query1")
        query2 = data.get("query2")
        model_name = (data.get("model") or "beit3").lower()
        k = int(data.get("k", 100))
        if k <= 0:
            k = 100
        # safety cap
        MAX_K = 500
        k = min(k, MAX_K)

        augment = bool(data.get("augment", False))
        detection = data.get("detection")
        # allow "a,b,c" string
        if isinstance(detection, str):
            detection = [x.strip() for x in detection.split(",") if x.strip()]
        objects = data.get("objects")
        operator = (data.get("operator") or "AND").upper()
        if operator not in ("AND", "OR"):
            operator = "AND"
        text = data.get("text")
        device = data.get("device", DEVICE_DEFAULT)
        use_llm = data.get("use_llm", augment)  # alternative flag

        # prepare collection
        collection = get_frames_collection()
        if collection is None:
            return jsonify({"ok": False, "error": "frames collection not available"}), 500

        # ensure models/indexes initialised once
        _ensure_models(device=device)

        # choose model and index
        model = None
        index = None
        metadata = None
        if model_name == "beit3":
            model = (_beit3, _beit3_tokenizer)
            index = _index_beit
            metadata = _index_metadata
        else:
            model = (_clip, _clip_tokenizer, _clip_preprocess)
            index = _index_clip
            # if metadata left only with beit index, we still use that metadata if available
            metadata = _index_metadata if _index_metadata is not None else []

        # If models/index failed to init, still allow fuzzy-only search
        if model[0] is None and (query1 or query2):
            current_app.logger.warning("Model or index not available, falling back to fuzzy-only search.")
            query1 = None
            query2 = None
            index = None

        # perform search
        topk, frame_paths = search_logic(
            collection=collection,
            metadata=metadata,
            model=model,
            query1=query1,
            index=index,
            augment=augment,
            k=k,
            query2=query2,
            llm=_llm if use_llm else None,
            detection=detection,
            objects=objects,
            operator=operator,
            text=text,
            device=device,
        )

        return jsonify({"ok": True, "count": len(topk), "topk": topk, "paths": frame_paths}), 200

    except Exception as e:
        current_app.logger.exception("search_collection failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500
