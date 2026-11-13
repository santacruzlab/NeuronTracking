"""Shared helpers for the behavior analysis scripts. Keep this file in the same
folder as the step*.py scripts (e.g. NeuronTracking\\scripts)."""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import tables

HDF_FS = 60.0        # task loop rate (Hz); rows -> seconds
SET_SIZE = 8         # one trial set = 8 rewarded trials, each target once
REWARDED_SEQUENCE = ["target", "hold", "targ_transition", "target", "hold", "targ_transition", "reward"]
TIMING_DEFINITION = "First target to reward in exact two-target seven-event sequence"
NAME_RE = re.compile(r"^(airp|braz)(\d{8})_(\d+)_te(\d+)\.hdf5?$")


def list_sessions(hdf_dir, monkeys):
    """{monkey: [session dicts]} sorted chronologically (date, then session number)."""
    out = {m: [] for m in monkeys}
    for p in Path(hdf_dir).iterdir():
        m = NAME_RE.match(p.name)
        if m and m.group(1) in out:
            date, sess = m.group(2), int(m.group(3))
            out[m.group(1)].append(dict(
                path=p, monkey=m.group(1), date=datetime.strptime(date, "%Y%m%d"), sess=sess,
                label=f"{date[:4]}-{date[4:6]}-{date[6:]}_{sess:02d}"))
    for v in out.values():
        v.sort(key=lambda s: (s["date"], s["sess"]))
    return out


def get_rewarded_trials(path, block, end="hold", target_source="auto"):
    """Rewarded trials of one block, in order. Returns a dict of arrays, or None.

    duration  : peripheral target onset -> `end`
                end="hold"   cursor enters target (reach time)
                end="reward" reward delivered (adds the fixed hold time)
    target_id : which of the 8 targets. target_source = "index" (task['target_index']),
                "angle" (from task['target'] position) or "auto" (index if it looks
                like 8 clean ids, otherwise angle).
    A rewarded trial ends with the messages: target, hold, targ_transition, reward.
    """
    with tables.open_file(path, "r") as hdf:
        tbl = hdf.root.task
        block_type = tbl.read(field="block_type").ravel()
        target_index = tbl.read(field="target_index").ravel()
        target_pos = tbl.read(field="target").reshape(len(block_type), -1)
        msgs = hdf.root.task_msgs.read()

    if not np.any(block_type == block):
        return None
    names = np.array([m.decode() if isinstance(m, bytes) else m for m in msgs["msg"]])
    rows = msgs["time"].astype(int)

    onset, enter, reward = [], [], []
    for i in np.flatnonzero(names == "reward"):
        if i < 3 or list(names[i - 3:i + 1]) != ["target", "hold", "targ_transition", "reward"]:
            continue
        if block_type[rows[i - 3]] != block:
            continue
        onset.append(rows[i - 3]); enter.append(rows[i - 2]); reward.append(rows[i])
    if not onset:
        return None
    onset, enter, reward = map(np.array, (onset, enter, reward))

    duration = ((enter if end == "hold" else reward) - onset) / HDF_FS

    idx = target_index[enter]
    use_index = target_source == "index" or (
        target_source == "auto" and idx.min() >= 0 and len(np.unique(idx)) == SET_SIZE)
    if use_index:
        target_id, source = idx.astype(int), "target_index"
    else:
        pos = target_pos[enter]
        ang = np.round(np.degrees(np.arctan2(pos[:, 2], pos[:, 0]))).astype(int) % 360
        uniq = np.unique(ang)
        target_id, source = np.searchsorted(uniq, ang), f"angle ({len(uniq)} distinct)"
    return dict(duration=duration, target_id=target_id, row_onset=onset, row_enter=enter,
                row_reward=reward, target_source=source)


def make_sets(target_id, set_size=SET_SIZE):
    """Group consecutive rewarded trials into sets. Returns (n_sets, valid) where
    valid[k] is True if set k contains each of the 8 targets exactly once.
    Leftover trials that don't fill a set are ignored."""
    n_sets = len(target_id) // set_size
    sets = np.asarray(target_id[:n_sets * set_size]).reshape(n_sets, set_size)
    valid = np.array([len(np.unique(s)) == set_size for s in sets], dtype=bool)
    return n_sets, valid


def rewarded_sequence_rows(names, rows, index):
    """Return the two target rows only for the exact contiguous reward sequence."""
    if index < 6 or list(names[index - 6:index + 1]) != REWARDED_SEQUENCE:
        return -1, -1
    sequence_rows = np.asarray(rows[index - 6:index + 1], int)
    if np.any(np.diff(sequence_rows) < 0):
        return -1, -1
    return int(sequence_rows[0]), int(sequence_rows[3])


def get_two_target_rewarded_trials(path, block):
    """One entry per reward; exact first-target-to-reward timing at 60 Hz.

    Rewards without the exact seven-event sequence remain invalid placeholders,
    so later trials and sets never shift. Timing must stay within one block.
    The second target supplies the peripheral direction for set balancing.
    """
    with tables.open_file(path, "r") as hdf:
        blocks = hdf.root.task.read(field="block_type").ravel()
        msgs = hdf.root.task_msgs.read()
        target = (hdf.root.task.read(field="target").reshape(len(blocks), -1)
                  if "target" in hdf.root.task.colnames else None)
    names = [m.decode() if isinstance(m, bytes) else str(m) for m in msgs["msg"]]
    rows = np.asarray(msgs["time"], int)
    durations, starts, second_targets, rewards, angles, reasons = [], [], [], [], [], []
    pending = None
    for i, msg in enumerate(msgs):
        name = names[i]
        row = int(msg["time"])
        if name == "premove":
            pending = row if 0 <= row < len(blocks) else None
        elif name == "wait" or name.endswith("_penalty") or name == "None":
            pending = None
        elif name == "reward":
            # Attribute exact sequences at first target onset; malformed rewards
            # use attempt/reward block attribution without shifting later sets.
            first, second = rewarded_sequence_rows(names, rows, i)
            anchor = first if 0 <= first < len(blocks) else pending if pending is not None else row
            if 0 <= anchor < len(blocks) and blocks[anchor] == block:
                reason = ("sequence_mismatch_or_nonmonotonic" if first < 0 else
                          "out_of_bounds" if not 0 <= first <= second <= row < len(blocks) else
                          "block_crossing" if not np.all(blocks[first:row + 1] == block) else "")
                valid = not reason
                starts.append(first)
                second_targets.append(second)
                rewards.append(row)
                reasons.append(reason)
                durations.append((row - first) / HDF_FS if valid else np.nan)
                angle = np.nan
                if target is not None and second >= 0:
                    target_row = second
                    if 0 <= target_row < len(target):
                        p = target[target_row]
                        if len(p) >= 3 and np.all(np.isfinite(p[[0, 2]])) and np.linalg.norm(p[[0, 2]]) > 0:
                            angle = float(np.round(np.degrees(np.arctan2(p[2], p[0]))) % 360)
                angles.append(angle)
            pending = None
    return dict(duration=np.asarray(durations, float),
                row_first_target=np.asarray(starts, int), row_second_target=np.asarray(second_targets, int),
                row_reward=np.asarray(rewards, int), invalid_reason=np.asarray(reasons, str),
                target_angle=np.asarray(angles, float))


def set_stats(duration, n_sets, set_size=SET_SIZE):
    """Mean and median trial duration of each complete set."""
    d = np.asarray(duration[:n_sets * set_size]).reshape(n_sets, set_size)
    return d.mean(axis=1), np.median(d, axis=1)


def skewness(x):
    x = np.asarray(x, float)
    return float(np.mean((x - x.mean()) ** 3) / x.std() ** 3) if x.std() > 0 else 0.0
