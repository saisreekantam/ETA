"""
Per-subsystem ground truth for HAI/HAIEnd 23.05, reconstructed the same way as
eval/hai_validation/attack_labels.py did for 22.04: hai_dataset_technical_details.pdf
pp.31-33 documents, for all 52 conducted attacks in chronological order, which controller
(hence subsystem) each targeted -- including 8 new AE01-AE08 "internal-point" attacks not
present in 22.04, all of which target P1 boiler controllers.

Unlike 22.04, exact attack timestamps did NOT need OCR transcription here: 23.05 ships
attack labels as a separate label-testN.csv (a clean 0/1 flag per second), so the
(start, end) windows below were extracted programmatically by finding contiguous 1-runs
in label-test1.csv/label-test2.csv -- their start times were then cross-checked against
the PDF table's HH:MM start times and match exactly for all 52 rows, confirming the
row-order alignment (both sources are strictly chronological within each file).

test1.csv's 14 attacks are ALL P1 (boiler)-only -- HAIEnd's own documentation notes its
attack campaign is "limited to the DCS of the boiler control system", so P2/P3 appear
only where a combination attack paired a P1 primitive with a P2/P3 one. P3 appears in
exactly ONE of the 52 attacks; P4 in none -- a real, structural class-imbalance property
of this dataset's attack campaign, not a labeling gap.
"""
from __future__ import annotations

import pandas as pd

_TARGETS: dict[str, list[set[str]]] = {
    "hai23-test1.csv": [{"P1"}] * 14,
    "hai23-test2.csv": (
        [{"P1"}] * 23  # rows 15-37 (23 rows, all single-P1 primitives/internal-point attacks)
        + [{"P1", "P2"}, {"P1", "P2"}, {"P1"}, {"P1"}, {"P1", "P2"}]  # rows 38-42
        + [{"P1"}, {"P1"}, {"P1", "P3"}, {"P1", "P2"}, {"P1", "P2"}]  # rows 43-47
        + [{"P1"}] * 5  # rows 48-52
    ),
}
assert len(_TARGETS["hai23-test2.csv"]) == 38

_WINDOWS: dict[str, list[tuple[str, str]]] = {
    "hai23-test1.csv": [
        ("2022-08-12 16:25:04", "2022-08-12 16:29:01"),
        ("2022-08-12 17:35:01", "2022-08-12 17:38:19"),
        ("2022-08-12 18:32:17", "2022-08-12 18:34:53"),
        ("2022-08-12 19:21:04", "2022-08-12 19:23:48"),
        ("2022-08-12 20:43:01", "2022-08-12 20:45:42"),
        ("2022-08-12 21:36:01", "2022-08-12 21:39:18"),
        ("2022-08-12 22:47:01", "2022-08-12 22:57:05"),
        ("2022-08-12 23:35:15", "2022-08-12 23:36:51"),
        ("2022-08-13 00:25:04", "2022-08-13 00:27:14"),
        ("2022-08-13 01:34:09", "2022-08-13 01:35:04"),
        ("2022-08-13 02:21:04", "2022-08-13 02:23:15"),
        ("2022-08-13 03:26:03", "2022-08-13 03:27:21"),
        ("2022-08-13 04:43:02", "2022-08-13 04:45:15"),
        ("2022-08-13 05:40:08", "2022-08-13 05:50:35"),
    ],
    "hai23-test2.csv": [
        ("2022-08-17 01:27:00", "2022-08-17 01:29:00"),
        ("2022-08-17 03:37:00", "2022-08-17 03:39:00"),
        ("2022-08-17 04:21:00", "2022-08-17 04:22:00"),
        ("2022-08-17 05:46:00", "2022-08-17 05:48:00"),
        ("2022-08-17 06:21:00", "2022-08-17 06:22:00"),
        ("2022-08-17 08:36:00", "2022-08-17 08:39:00"),
        ("2022-08-17 09:42:00", "2022-08-17 09:52:00"),
        ("2022-08-17 10:36:00", "2022-08-17 10:38:00"),
        ("2022-08-17 11:35:00", "2022-08-17 11:36:00"),
        ("2022-08-17 12:25:00", "2022-08-17 12:26:00"),
        ("2022-08-17 13:47:00", "2022-08-17 13:50:00"),
        ("2022-08-17 14:25:00", "2022-08-17 14:27:00"),
        ("2022-08-17 15:13:00", "2022-08-17 15:22:00"),
        ("2022-08-17 17:34:00", "2022-08-17 17:35:00"),
        ("2022-08-17 18:16:00", "2022-08-17 18:18:00"),
        ("2022-08-17 19:40:00", "2022-08-17 19:41:00"),
        ("2022-08-17 20:12:00", "2022-08-17 20:20:00"),
        ("2022-08-17 22:41:00", "2022-08-17 22:44:00"),
        ("2022-08-17 23:38:00", "2022-08-17 23:40:00"),
        ("2022-08-18 13:48:00", "2022-08-18 13:50:00"),
        ("2022-08-18 14:58:00", "2022-08-18 15:00:00"),
        ("2022-08-18 16:20:00", "2022-08-18 16:23:00"),
        ("2022-08-18 17:38:00", "2022-08-18 17:39:00"),
        ("2022-08-18 18:45:00", "2022-08-18 18:46:00"),
        ("2022-08-18 19:21:00", "2022-08-18 19:22:00"),
        ("2022-08-18 20:32:00", "2022-08-18 20:34:00"),
        ("2022-08-18 21:41:00", "2022-08-18 21:43:00"),
        ("2022-08-18 23:15:00", "2022-08-18 23:17:00"),
        ("2022-08-19 01:23:00", "2022-08-19 01:24:00"),
        ("2022-08-19 02:43:00", "2022-08-19 02:45:00"),
        ("2022-08-19 04:34:00", "2022-08-19 04:35:00"),
        ("2022-08-19 05:14:00", "2022-08-19 05:16:00"),
        ("2022-08-19 06:46:00", "2022-08-19 07:20:00"),
        ("2022-08-19 08:24:00", "2022-08-19 08:32:00"),
        ("2022-08-19 09:27:00", "2022-08-19 09:28:00"),
        ("2022-08-19 10:34:00", "2022-08-19 10:35:00"),
        ("2022-08-19 14:18:00", "2022-08-19 14:21:00"),
        ("2022-08-19 14:51:00", "2022-08-19 14:53:00"),
    ],
}
assert len(_WINDOWS["hai23-test2.csv"]) == 38


def attack_intervals(filename: str) -> list[tuple[pd.Timestamp, pd.Timestamp, set[str]]]:
    """test2a.csv/test2b.csv both look up under the original hai23-test2.csv key -- the
    file was split by timestamp AFTER labeling (common.py), so windows outside a given
    split's own time range simply never match any row in that file's dataframe."""
    key = "hai23-test2.csv" if filename.startswith("hai23-test2") else filename
    return [
        (pd.Timestamp(start), pd.Timestamp(end), subsystems)
        for (start, end), subsystems in zip(_WINDOWS[key], _TARGETS[key])
    ]
