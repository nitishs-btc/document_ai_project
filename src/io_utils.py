import re
from collections import OrderedDict
from pathlib import Path


PAGE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<page>\d+)$")


def image_output_stem(path):
    return Path(path).stem


def group_image_paths(paths):
    grouped = OrderedDict()

    for path in sorted(paths):
        stem = Path(path).stem
        match = PAGE_SUFFIX_RE.match(stem)

        if match:
            base = match.group("base")
            page = int(match.group("page"))
            grouped.setdefault(base, []).append({"path": path, "page": page, "stem": stem})
            continue

        grouped.setdefault(stem, []).append({"path": path, "page": None, "stem": stem})

    documents = []

    for name, items in grouped.items():
        if len(items) > 1 and all(item["page"] is not None for item in items):
            ordered_paths = [item["path"] for item in sorted(items, key=lambda value: value["page"])]
            documents.append({"name": name, "paths": ordered_paths, "multi_page": True})
            continue

        for item in items:
            documents.append({"name": item["stem"], "paths": [item["path"]], "multi_page": False})

    return documents
