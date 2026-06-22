"""JSON persistence helpers and association lookups."""

import json
import os


def load_data_from_json(filename: str):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    return None


def save_data_to_json(filename: str, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_image_codes_to_json(filename, image_name, iconclass_codes):
    if os.path.exists(filename):
        data = load_data_from_json(filename)
    else:
        data = {}
    if not isinstance(iconclass_codes, list):
        iconclass_codes = list(iconclass_codes)
    data[image_name] = iconclass_codes
    save_data_to_json(filename, data)


def read_image_codes_from_json(filename, image_name):
    if not os.path.exists(filename):
        return None
    data = load_data_from_json(filename)
    iconclass_codes = data.get(image_name)
    return iconclass_codes


def get_codes_from_associations_list(filename, iconclass_codes):
    associated_iconclass_codes = []
    data = load_data_from_json(filename)
    for entry in data['associations_list']:
        if set(entry['input_codes']).issubset(iconclass_codes):
            associated_iconclass_codes.append(entry['output_code'])
    return list(set(associated_iconclass_codes))
