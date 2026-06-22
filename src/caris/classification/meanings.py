"""Resolve Iconclass codes to their human-readable meanings."""

import os

from caris.config import DEFAULT_CLEAN_TXT_PATH


def get_meanings_from_codes(codes, filename=None):
    if filename is None:
        filename = DEFAULT_CLEAN_TXT_PATH

    if not os.path.exists(filename):
        return []

    code_to_meaning = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|', 1)
            if len(parts) == 2:
                code_to_meaning[parts[0]] = parts[1]

    meanings = []
    for code in codes:
        if code in code_to_meaning:
            meanings.append(code_to_meaning[code])
    return meanings
