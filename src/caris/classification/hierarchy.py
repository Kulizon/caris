"""Iconclass hierarchy access and keyword-index based search."""

import iconclass

_ICONCLASS_CACHE = None
_KEYWORD_INDEX = None   # keyword -> set of code strings
_CODE_KW_MAP = None     # code string -> list of keywords (preserves order for exact match)


def get_iconclass_hierarchy():
    global _ICONCLASS_CACHE
    if _ICONCLASS_CACHE is None:
        _ICONCLASS_CACHE = iconclass.init()
    return _ICONCLASS_CACHE


def _ensure_search_index():
    global _KEYWORD_INDEX, _CODE_KW_MAP
    if _KEYWORD_INDEX is not None:
        return
    _KEYWORD_INDEX = {}
    _CODE_KW_MAP = {}
    stack = list(get_iconclass_hierarchy()[""])
    while stack:
        child = stack.pop()
        code_str = str(child)
        kws = child.keywords()
        _CODE_KW_MAP[code_str] = kws
        for kw in kws:
            if kw not in _KEYWORD_INDEX:
                _KEYWORD_INDEX[kw] = set()
            _KEYWORD_INDEX[kw].add(code_str)
        stack.extend(child)


def search_equal(detected_objects):
    _ensure_search_index()
    if not detected_objects:
        return []
    candidates = None
    for obj in detected_objects:
        matching = _KEYWORD_INDEX.get(obj, set())
        candidates = matching if candidates is None else candidates & matching
    if not candidates:
        return []
    return [c for c in candidates if _CODE_KW_MAP.get(c) == detected_objects]


def search_subset(detected_objects):
    _ensure_search_index()
    if not detected_objects:
        return []
    candidates = None
    for obj in detected_objects:
        matching = _KEYWORD_INDEX.get(obj, set())
        candidates = matching if candidates is None else candidates & matching
    return list(candidates) if candidates else []
