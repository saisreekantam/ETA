"""
Per-subsystem ground truth for HAI 22.04, reconstructed from two public sources that
HAI's raw CSVs do NOT combine themselves:

  1. hai_dataset_technical_details.pdf pp.33-35 ("HAI 22.04" attack table) -- gives,
     for each of the 58 conducted attacks in chronological order, which controller
     (hence which subsystem: P1 boiler / P2 turbine / P3 water-treatment) it targeted.
     26 of the 58 are "combination" attacks hitting two controllers simultaneously.
  2. summary(testN.csv).txt files -- give the authoritative minute:second start/end
     timestamp for each attack instance actually present in test1..test4.csv.

The per-file attack counts from both sources agree exactly (test1=7, test2=17, test3=10,
test4=24, summing to the documented 58), and are strictly chronological within each file,
so source (1)'s Nth attack in a file lines up with source (2)'s Nth summary entry for
that same file. No P4 (auxiliary/HIL loop) attacks exist in HAI 22.04 at all -- P4 is
legitimately all-negative in this version, same as the production model's control_room
zone having no sensor cluster and scoring ~0 by construction.

This is a manual transcription of a scanned PDF table; a handful of individual Target
Point strings were visibly OCR-garbled, but the Target Controller prefix (which is all
that determines subsystem here) read unambiguously in every row.
"""
from __future__ import annotations

import pandas as pd

# (filename, [(local_1-based_index, {subsystems})])
_TARGETS: dict[str, list[set[str]]] = {
    "test1.csv": [
        {"P1"}, {"P1"}, {"P1"}, {"P3"}, {"P1"}, {"P1"}, {"P1"},
    ],
    "test2.csv": [
        {"P1"}, {"P1"}, {"P2"}, {"P1", "P2"}, {"P3"}, {"P1", "P2"}, {"P1"},
        {"P2"}, {"P1"}, {"P2"}, {"P1"}, {"P3"}, {"P1"}, {"P3"}, {"P1"},
        {"P1"}, {"P1"},
    ],
    "test3.csv": [
        {"P1", "P3"}, {"P1"}, {"P1", "P3"}, {"P3"}, {"P1"}, {"P1"}, {"P1"},
        {"P1"}, {"P1"}, {"P3"},
    ],
    "test4.csv": [
        {"P2"}, {"P1"}, {"P2"}, {"P2", "P3"}, {"P1"}, {"P1"}, {"P1"}, {"P1"},
        {"P1"}, {"P1", "P2"}, {"P1", "P2"}, {"P1"}, {"P1", "P2"}, {"P2", "P3"},
        {"P2"}, {"P1"}, {"P1"}, {"P1", "P3"}, {"P1"}, {"P2", "P1"}, {"P1", "P2"},
        {"P1", "P2"}, {"P1"}, {"P1"},
    ],
}

# exact (start, end) datetimes transcribed from summary(testN.csv).txt, same order
_WINDOWS: dict[str, list[tuple[str, str]]] = {
    "test1.csv": [
        ("2021-07-10 05:41:22", "2021-07-10 05:44:32"),
        ("2021-07-10 07:19:12", "2021-07-10 07:20:06"),
        ("2021-07-10 11:25:13", "2021-07-10 11:27:19"),
        ("2021-07-10 15:39:12", "2021-07-10 15:40:06"),
        ("2021-07-10 16:42:17", "2021-07-10 16:47:13"),
        ("2021-07-10 19:21:16", "2021-07-10 19:22:47"),
        ("2021-07-10 22:35:20", "2021-07-10 22:36:27"),
    ],
    "test2.csv": [
        ("2021-07-13 16:38:04", "2021-07-13 16:42:21"),
        ("2021-07-13 17:21:03", "2021-07-13 17:22:08"),
        ("2021-07-13 18:13:04", "2021-07-13 18:13:49"),
        ("2021-07-13 20:28:06", "2021-07-13 20:32:14"),
        ("2021-07-13 21:10:07", "2021-07-13 21:11:02"),
        ("2021-07-13 21:58:07", "2021-07-13 22:01:03"),
        ("2021-07-13 23:40:14", "2021-07-13 23:44:58"),
        ("2021-07-14 01:15:08", "2021-07-14 01:17:40"),
        ("2021-07-14 01:40:07", "2021-07-14 01:42:49"),
        ("2021-07-14 03:23:27", "2021-07-14 03:25:04"),
        ("2021-07-14 07:21:15", "2021-07-14 07:23:46"),
        ("2021-07-14 08:11:12", "2021-07-14 08:12:07"),
        ("2021-07-14 10:35:15", "2021-07-14 10:36:35"),
        ("2021-07-14 11:23:17", "2021-07-14 11:33:30"),
        ("2021-07-14 12:17:13", "2021-07-14 12:20:01"),
        ("2021-07-14 13:52:15", "2021-07-14 13:54:53"),
        ("2021-07-14 14:31:03", "2021-07-14 14:32:41"),
    ],
    "test3.csv": [
        ("2021-07-14 18:21:09", "2021-07-14 18:26:57"),
        ("2021-07-14 20:16:03", "2021-07-14 20:22:01"),
        ("2021-07-14 23:22:01", "2021-07-14 23:24:24"),
        ("2021-07-15 01:41:07", "2021-07-15 01:42:38"),
        ("2021-07-15 02:09:10", "2021-07-15 02:10:44"),
        ("2021-07-15 03:37:10", "2021-07-15 03:43:03"),
        ("2021-07-15 05:35:10", "2021-07-15 05:37:41"),
        ("2021-07-15 06:53:07", "2021-07-15 06:56:00"),
        ("2021-07-15 07:42:10", "2021-07-15 07:43:46"),
        ("2021-07-15 09:52:12", "2021-07-15 10:25:56"),
    ],
    "test4.csv": [
        ("2021-07-15 12:42:02", "2021-07-15 12:42:40"),
        ("2021-07-15 13:20:03", "2021-07-15 13:21:31"),
        ("2021-07-15 13:57:02", "2021-07-15 13:58:38"),
        ("2021-07-15 15:08:01", "2021-07-15 15:09:38"),
        ("2021-07-15 16:07:52", "2021-07-15 16:16:17"),
        ("2021-07-15 17:22:03", "2021-07-15 17:25:09"),
        ("2021-07-15 19:45:08", "2021-07-15 19:47:10"),
        ("2021-07-15 20:29:07", "2021-07-15 20:40:20"),
        ("2021-07-15 22:41:09", "2021-07-15 22:42:12"),
        ("2021-07-16 01:07:08", "2021-07-16 01:10:07"),
        ("2021-07-16 03:35:08", "2021-07-16 03:36:47"),
        ("2021-07-16 04:02:15", "2021-07-16 04:04:51"),
        ("2021-07-16 04:59:13", "2021-07-16 05:01:46"),
        ("2021-07-16 07:20:12", "2021-07-16 07:21:29"),
        ("2021-07-16 09:17:12", "2021-07-16 09:18:29"),
        ("2021-07-16 10:39:50", "2021-07-16 10:42:04"),
        ("2021-07-16 11:22:33", "2021-07-16 11:31:37"),
        ("2021-07-16 13:23:10", "2021-07-16 13:28:52"),
        ("2021-07-16 14:59:07", "2021-07-16 15:01:50"),
        ("2021-07-16 15:57:03", "2021-07-16 15:58:32"),
        ("2021-07-16 17:34:08", "2021-07-16 17:36:40"),
        ("2021-07-16 20:08:07", "2021-07-16 20:10:52"),
        ("2021-07-16 22:17:17", "2021-07-16 22:19:12"),
        ("2021-07-16 23:05:16", "2021-07-16 23:06:42"),
    ],
}

for _f in _TARGETS:
    assert len(_TARGETS[_f]) == len(_WINDOWS[_f]), f"{_f}: target/window count mismatch"


def attack_intervals(filename: str) -> list[tuple[pd.Timestamp, pd.Timestamp, set[str]]]:
    """List of (start, end, {subsystems}) for one HAI test file, inclusive bounds."""
    return [
        (pd.Timestamp(start), pd.Timestamp(end), subsystems)
        for (start, end), subsystems in zip(_WINDOWS[filename], _TARGETS[filename])
    ]


if __name__ == "__main__":
    total = sum(len(v) for v in _TARGETS.values())
    print(f"{total} attacks transcribed across {len(_TARGETS)} files (expect 58)")
    from collections import Counter
    counts = Counter()
    for subsystems_list in _TARGETS.values():
        for s in subsystems_list:
            counts.update(s)
    print("Per-subsystem attack counts (an attack hitting 2 subsystems counts once each):", dict(counts))
