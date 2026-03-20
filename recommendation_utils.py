from utils import load_data_from_json


def recommend_image_simple(filename, iconclass_codes_list):
    results = {}
    data = load_data_from_json(filename)
    for entry in data:
        results[entry] = 0
        codes = data[entry]

        for iconclass_code in iconclass_codes_list:
            iconclass_code = iconclass_code.split('(')[0].strip()
            max_score = 0

            for code in codes:
                code = code.split('(')[0].strip()

                if iconclass_code == code:
                    max_score = 1
                    break
                elif max_score < 0.5 and (
                        iconclass_code[:-1] == code[:-1] or iconclass_code[:-1] == code or iconclass_code == code[:-1]):
                    max_score = 0.5
                elif max_score < 0.25 and (
                        iconclass_code[:-2] == code[:-2] or iconclass_code[:-2] == code or iconclass_code == code[:-2]):
                    max_score = 0.25

            results[entry] += max_score
    max_key, max_value = max(results.items(), key=lambda item: item[1])

    return max_key


def calculate_idf_for_codes(data):
    import math
    N = len(data)
    code_document_counts = {}

    for codes in data.values():
        unique_codes = set(code.split('(')[0].strip() for code in codes)
        for code in unique_codes:
            code_document_counts[code] = code_document_counts.get(code, 0) + 1

    idf_scores = {}
    for code, nt in code_document_counts.items():
        idf_scores[code] = math.log((N + 1) / (nt + 1)) + 1

    return idf_scores


def recommend_image_idf_based(filename, iconclass_codes_list, idf_impact=1):
    data = load_data_from_json(filename)
    idf_scores = calculate_idf_for_codes(data)
    results = {}

    for entry, codes in data.items():
        entry_codes_clean = set(code.split('(')[0].strip() for code in codes)
        score = 0
        for iconclass_code in iconclass_codes_list:
            iconclass_code_clean = iconclass_code.split('(')[0].strip()
            if iconclass_code_clean in entry_codes_clean:
                score += idf_scores.get(iconclass_code_clean, 0) ** idf_impact
        results[entry] = score

    max_key, max_value = max(results.items(), key=lambda item: item[1])
    return max_key


def recommend_image_jaccard(filename, iconclass_codes_list):
    data = load_data_from_json(filename)
    iconclass_codes_clean = set(code.split('(')[0].strip() for code in iconclass_codes_list)
    results = {}

    for entry, codes in data.items():
        score = 0
        entry_codes_clean = set(code.split('(')[0].strip() for code in codes)
        common_codes = iconclass_codes_clean & entry_codes_clean
        all_codes = iconclass_codes_clean | entry_codes_clean
        if all_codes and common_codes:
            score = len(common_codes) / len(all_codes)
        results[entry] = score

    max_key, max_value = max(results.items(), key=lambda item: item[1])
    return max_key
