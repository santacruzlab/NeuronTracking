import csv
import os


CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "input_data",
    "sessions_analyzed_rotation.csv",
)

path_santacruz1 = r"Y:\storage\rawdata"
path_santacruz2 = r"X:\storage\rawdata"
path_santacruz3 = r"W:\storage\rawdata"



def load_sessions() -> tuple[list[str], list[str]]:
    sessions = {"airport": [], "brazos": []}

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"monkey", "hdf"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("CSV must contain 'monkey' and 'hdf' columns")

        for row_number, row in enumerate(reader, start=2):
            monkey = (row["monkey"] or "").strip().casefold()
            hdf_file = (row["hdf"] or "").strip()
            if monkey not in sessions:
                raise ValueError(
                    f"Unexpected monkey {row['monkey']!r} on CSV row {row_number}"
                )
            if not hdf_file:
                raise ValueError(f"Missing hdf filename on CSV row {row_number}")

            sessions[monkey].append(os.path.splitext(os.path.basename(hdf_file))[0])

    return sessions["airport"], sessions["brazos"]


AIRPORT_SESSIONS, BRAZOS_SESSIONS = load_sessions()

print(f"Loaded {len(AIRPORT_SESSIONS)} airport sessions and {len(BRAZOS_SESSIONS)} brazos sessions from CSV.")

for session in AIRPORT_SESSIONS:
    found_path_ns5 = False
    found_path_ns2 = False
    found_path_nev = False
    print(f"Airport session: {session} ********************************")
    for path in [path_santacruz1, path_santacruz2, path_santacruz3]:
        file_prefix_ripple = os.path.join(path, "ripple", session)
        if os.path.exists(file_prefix_ripple + '.ns5'):
            found_path_ns5 = file_prefix_ripple + '.ns5'
            print(f".ns5: found in {path}")
        if os.path.exists(file_prefix_ripple + '.ns2'):
            found_path_ns2 = file_prefix_ripple + '.ns2'
            print(f".ns2: found in {path}")
        if os.path.exists(file_prefix_ripple + '.nev'):
            found_path_nev = file_prefix_ripple + '.nev'
            print(f".nev: found in {path}")
    if not found_path_ns5:
        print(f".ns5: NOT FOUND for session {session}")
    if not found_path_ns2:
        print(f".ns2: NOT FOUND for session {session}")
    if not found_path_nev:
        print(f".nev: NOT FOUND for session {session}")

for session in BRAZOS_SESSIONS:
    found_path_ns5 = False
    found_path_ns2 = False
    found_path_nev = False
    print(f"Brazos session: {session} ********************************")
    for path in [path_santacruz1, path_santacruz2, path_santacruz3]:
        file_prefix_ripple = os.path.join(path, "ripple", session)
        if os.path.exists(file_prefix_ripple + '.ns5'):
            found_path_ns5 = file_prefix_ripple + '.ns5'
            print(f".ns5: found in {path}")
        if os.path.exists(file_prefix_ripple + '.ns2'):
            found_path_ns2 = file_prefix_ripple + '.ns2'
            print(f".ns2: found in {path}")
        if os.path.exists(file_prefix_ripple + '.nev'):
            found_path_nev = file_prefix_ripple + '.nev'
            print(f".nev: found in {path}")
    if not found_path_ns5:
        print(f".ns5: NOT FOUND for session {session}")
    if not found_path_ns2:
        print(f".ns2: NOT FOUND for session {session}")
    if not found_path_nev:
        print(f".nev: NOT FOUND for session {session}")

AIRPORT_SESSIONS_1 = 
AIRPORT_SESSIONS_2 = 