#!/usr/bin/env python3
r"""
Generate <session>_syncHDF.mat files from Ripple .ns5 recordings.

Standalone replacement for ``tracking.Sync`` / ``preprocess_santacruz_data.Sync``
that does not need ``riglib``.  The extraction logic is the same
(``Sync._extract_rows``) but the .ns5 file is streamed in chunks and only the 17
digital-IO channels are kept, so a 10 GB recording needs a few hundred MB of RAM.
The input .ns5 files are only ever opened for reading.

    py scripts/make_syncHDF.py --ns5-dir \\server\share\storage\rawdata\ripple \
        --out-dir data/syncHDF_mat  SESSION [SESSION ...]
    py scripts/make_syncHDF.py --compare existing.mat new.mat
"""
import argparse
import os
import struct
import sys
import time

import numpy as np
import scipy.io

# Same channel layout as tracking.Sync: rows of the (channels x samples) array.
PINS_UTIL = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]) + 3
STROBE_ROW = 20 + 3
THRESH = 2500


class Ns5Header:
    """Minimal NSx 2.1/2.2/2.3 header parser (read only)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.file_type = f.read(8).decode("latin-1")
            if self.file_type == "NEURALSG":          # spec 2.1
                self.label = f.read(16)
                self.period = struct.unpack("<I", f.read(4))[0]
                self.time_res = 30000
                self.n_ch = struct.unpack("<I", f.read(4))[0]
                self.ch_ids = list(struct.unpack("<%dI" % self.n_ch, f.read(4 * self.n_ch)))
                self.data_start = f.tell()
                self.packets = [(self.data_start, None)]
                self.ts_fmt = None
            elif self.file_type == "NEURALCD":        # spec 2.2 / 2.3
                self.major, self.minor = struct.unpack("<BB", f.read(2))
                self.hdr_bytes = struct.unpack("<I", f.read(4))[0]
                self.label = f.read(16)
                f.read(256)                           # comment
                self.period = struct.unpack("<I", f.read(4))[0]
                self.time_res = struct.unpack("<I", f.read(4))[0]
                f.read(16)                            # time origin
                self.n_ch = struct.unpack("<I", f.read(4))[0]
                self.ch_ids = []
                for _ in range(self.n_ch):
                    ext = f.read(66)
                    assert ext[:2] == b"CC", "unexpected extended header"
                    self.ch_ids.append(struct.unpack("<H", ext[2:4])[0])
                self.data_start = f.tell()
                assert self.data_start == self.hdr_bytes, (self.data_start, self.hdr_bytes)
                self.ts_fmt = "<Q" if (self.major, self.minor) >= (2, 3) else "<I"
            else:
                raise ValueError("not an NSx file: %r" % self.file_type)
        self.fs = self.time_res / self.period
        self.size = os.path.getsize(path)

    def iter_packets(self):
        """Yield (byte_offset, n_samples) for every data packet."""
        if self.file_type == "NEURALSG":
            yield self.data_start, (self.size - self.data_start) // (2 * self.n_ch)
            return
        ts_len = struct.calcsize(self.ts_fmt)
        with open(self.path, "rb") as f:
            pos = self.data_start
            while pos < self.size:
                f.seek(pos)
                hdr = f.read(1 + ts_len + 4)
                if len(hdr) < 1 + ts_len + 4 or hdr[0] != 1:
                    raise ValueError("bad data packet header at byte %d" % pos)
                n = struct.unpack("<I", hdr[1 + ts_len:])[0]
                if n == 0:
                    # Point count was never finalized (recording not closed
                    # cleanly): the packet runs to the end of the file.
                    n = (self.size - (pos + 1 + ts_len + 4)) // (2 * self.n_ch)
                    yield pos + 1 + ts_len + 4, n
                    return
                yield pos + 1 + ts_len + 4, n
                pos += 1 + ts_len + 4 + n * self.n_ch * 2


def read_dio(path, chunk_samples=1_500_000):
    """Return (rows x samples uint8 of thresholded needed channels, fs).

    Rows are PINS_UTIL (16 channels) followed by the strobe row, thresholded at
    2500 exactly like Sync._extract_rows does (> 2500 on the raw int16 values).
    """
    h = Ns5Header(path)
    need = list(PINS_UTIL) + [STROBE_ROW]
    assert max(need) < h.n_ch, "file has only %d channels" % h.n_ch
    parts = []
    n_packets = 0
    with open(path, "rb") as f:
        for off, n in h.iter_packets():
            n_packets += 1
            f.seek(off)
            left = n
            while left > 0:
                m = min(chunk_samples, left)
                raw = np.fromfile(f, dtype="<i2", count=m * h.n_ch)
                full = raw.size // h.n_ch
                if full:
                    block = raw[:full * h.n_ch].reshape(full, h.n_ch)[:, need]
                    parts.append((block > THRESH).T.copy())
                if raw.size != m * h.n_ch:
                    # File shorter than its header claims: keep what exists.
                    print("WARNING: truncated file, kept %d samples" % sum(p.shape[1] for p in parts), file=sys.stderr)
                    left = 0
                    break
                left -= m
    bits = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return bits, h.fs, n_packets


def extract_rows(bits, fs):
    """Same algorithm as Sync._extract_rows, on the thresholded bit array."""
    msgtype = np.flip(bits[8:16], axis=0)
    rownum = np.flip(bits[:8], axis=0)
    strobe = bits[16].astype(int)
    weights = (1 << np.arange(7, -1, -1)).astype(np.int64)
    MSGTYPE = (msgtype.astype(np.int64) * weights[:, None]).sum(axis=0)
    ROWNUMB = (rownum.astype(np.int64) * weights[:, None]).sum(axis=0)

    find_recording_start = np.ravel(np.nonzero(strobe))[0]
    find_data_rows = np.logical_and(np.ravel(np.equal(MSGTYPE, 13)), np.ravel(np.greater(strobe, 0)))
    find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))
    rows = ROWNUMB[find_data_rows_ind].astype(float)

    prev_row = rows[0]
    counter = 0
    for ind in range(1, len(rows)):
        row = rows[ind]
        counter += row < prev_row
        rows[ind] = counter * 256 + row
        prev_row = row

    return {
        "row_number": rows,
        "ripple_samplenumber": find_data_rows_ind,
        "ripple_recording_start": find_recording_start,
        "ripple_dio_samplerate": fs,
    }


def make_sync(ns5_path, out_path):
    t0 = time.time()
    bits, fs, n_packets = read_dio(ns5_path)
    result = extract_rows(bits, fs)
    scipy.io.savemat(out_path, result)
    return result, n_packets, time.time() - t0


def compare(a, b):
    A = scipy.io.loadmat(a)
    B = scipy.io.loadmat(b)
    ok = True
    for k in ("row_number", "ripple_samplenumber", "ripple_recording_start", "ripple_dio_samplerate"):
        x, y = np.ravel(A[k]).astype(float), np.ravel(B[k]).astype(float)
        same = x.shape == y.shape and np.array_equal(x, y)
        print("%-24s %s  (%s vs %s values)" % (k, "IDENTICAL" if same else "DIFFERENT", x.size, y.size))
        ok &= same
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--ns5-dir")
    ap.add_argument("--out-dir", default="data/syncHDF_mat")
    ap.add_argument("--compare", nargs=2, metavar=("A.mat", "B.mat"))
    ap.add_argument("--header-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    os.makedirs(args.out_dir, exist_ok=True)
    for s in args.sessions:
        ns5 = os.path.join(args.ns5_dir, s + ".ns5")
        out = os.path.join(args.out_dir, s + "_syncHDF.mat")
        if args.header_only:
            h = Ns5Header(ns5)
            print(s, h.file_type, getattr(h, "major", "-"), getattr(h, "minor", "-"), "channels", h.n_ch,
                  "fs", h.fs, "packets", sum(1 for _ in h.iter_packets()))
            continue
        if os.path.exists(out) and not args.overwrite:
            print("skip (exists):", out)
            continue
        res, n_packets, secs = make_sync(ns5, out)
        print("%s: %d rows, %d packet(s), %.0fs -> %s" % (s, len(res["row_number"]), n_packets, secs, out))


if __name__ == "__main__":
    main()
