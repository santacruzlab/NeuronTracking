#%% Imports and define paths
import csv
import tables
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

#%% Load sessions from CSV

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

#%% Check for existence of HDF files and categorize sessions by location

AIRPORT_HDF_SESSIONS_1 = []
AIRPORT_HDF_SESSIONS_2 = []
AIRPORT_HDF_SESSIONS_OTHER_LOCATION = []
AIRPORT_HDF_SESSIONS_NOT_FOUND = []

for session in AIRPORT_SESSIONS:
    found_locations = []
    for path in [path_santacruz1, path_santacruz2, path_santacruz3]:
        hdf_file = os.path.join(path, "hdf", session + ".hdf")
        if os.path.exists(hdf_file):
            found_locations.append(path)

    if not found_locations:
        AIRPORT_HDF_SESSIONS_NOT_FOUND.append(session)
        print(f"HDF NOT FOUND for Airport session {session}")
        continue

    if path_santacruz2 in found_locations:
        AIRPORT_HDF_SESSIONS_1.append(session)
    if path_santacruz1 in found_locations:
        AIRPORT_HDF_SESSIONS_2.append(session)
    if path_santacruz3 in found_locations:
        AIRPORT_HDF_SESSIONS_OTHER_LOCATION.append(session)
    if len(found_locations) > 1:
        print(
            f"MIXED HDF LOCATIONS for Airport session {session}: "
            f"{', '.join(found_locations)}"
        )

BRAZOS_HDF_SESSIONS_1 = []
BRAZOS_HDF_SESSIONS_2 = []
BRAZOS_HDF_SESSIONS_3 = []
BRAZOS_HDF_SESSIONS_NOT_FOUND = []

for session in BRAZOS_SESSIONS:
    found_locations = []
    for path in [path_santacruz1, path_santacruz2, path_santacruz3]:
        hdf_file = os.path.join(path, "hdf", session + ".hdf")
        if os.path.exists(hdf_file):
            found_locations.append(path)

    if not found_locations:
        BRAZOS_HDF_SESSIONS_NOT_FOUND.append(session)
        print(f"HDF NOT FOUND for Brazos session {session}")
        continue

    if path_santacruz1 in found_locations:
        BRAZOS_HDF_SESSIONS_1.append(session)
    if path_santacruz2 in found_locations:
        BRAZOS_HDF_SESSIONS_2.append(session)
    if path_santacruz3 in found_locations:
        BRAZOS_HDF_SESSIONS_3.append(session)
    if len(found_locations) > 1:
        print(
            f"MIXED HDF LOCATIONS for Brazos session {session}: "
            f"{', '.join(found_locations)}"
        )

#%% Check for existence of ripple files and categorize sessions by location
AIRPORT_RIPPLE_SESSIONS_1 = []
AIRPORT_RIPPLE_SESSIONS_2 = []
AIRPORT_RIPPLE_SESSIONS_NOT_FOUND = []
AIRPORT_RIPPLE_SESSIONS_OTHER_LOCATION = []

for session in AIRPORT_SESSIONS:
    found_extensions = {".ns5": False, ".ns2": False, ".nev": False}
    found_locations = set()
    # print(f"Airport session: {session} ********************************")
    for path in [path_santacruz1, path_santacruz2, path_santacruz3]:
        file_prefix_ripple = os.path.join(path, "ripple", session)
        for extension in found_extensions:
            if os.path.exists(file_prefix_ripple + extension):
                found_extensions[extension] = True
                found_locations.add(path)
                # print(f"{extension}: found in {path}")

    if not found_extensions[".ns5"]:
        print(f".ns5: NOT FOUND for session {session}")
    if not found_extensions[".ns2"]:
        print(f".ns2: NOT FOUND for session {session}")
    if not found_extensions[".nev"]:
        print(f".nev: NOT FOUND for session {session}")

    if not found_locations:
        AIRPORT_RIPPLE_SESSIONS_NOT_FOUND.append(session)
        continue

    if path_santacruz2 in found_locations:
        AIRPORT_RIPPLE_SESSIONS_1.append(session)
    if path_santacruz1 in found_locations:
        AIRPORT_RIPPLE_SESSIONS_2.append(session)
    if path_santacruz3 in found_locations:
        AIRPORT_RIPPLE_SESSIONS_OTHER_LOCATION.append(session)
    if len(found_locations) > 1:
        print(f"MIXED LOCATIONS for {session}: {', '.join(sorted(found_locations))}")
    if path_santacruz3 in found_locations and not (
        path_santacruz1 in found_locations or path_santacruz2 in found_locations
    ):
        print(f"No Airport session group location found for {session}: {path_santacruz3}")

BRAZOS_RIPPLE_SESSIONS_1 = []
BRAZOS_RIPPLE_SESSIONS_2 = []
BRAZOS_RIPPLE_SESSIONS_3 = []
BRAZOS_RIPPLE_SESSIONS_NOT_FOUND = []

for session in BRAZOS_SESSIONS:
    found_extensions = {".ns5": False, ".ns2": False, ".nev": False}
    found_locations = set()
    # print(f"Brazos session: {session} ********************************")
    for path in [path_santacruz1, path_santacruz2, path_santacruz3]:
        file_prefix_ripple = os.path.join(path, "ripple", session)
        for extension in found_extensions:
            if os.path.exists(file_prefix_ripple + extension):
                found_extensions[extension] = True
                found_locations.add(path)
                # print(f"{extension}: found in {path}")

    if not found_extensions[".ns5"]:
        print(f".ns5: NOT FOUND for session {session}")
    if not found_extensions[".ns2"]:
        print(f".ns2: NOT FOUND for session {session}")
    if not found_extensions[".nev"]:
        print(f".nev: NOT FOUND for session {session}")

    if not found_locations:
        BRAZOS_RIPPLE_SESSIONS_NOT_FOUND.append(session)
        continue

    if path_santacruz1 in found_locations:
        BRAZOS_RIPPLE_SESSIONS_1.append(session)
    if path_santacruz2 in found_locations:
        BRAZOS_RIPPLE_SESSIONS_2.append(session)
    if path_santacruz3 in found_locations:
        BRAZOS_RIPPLE_SESSIONS_3.append(session)
    if len(found_locations) > 1:
        print(f"MIXED LOCATIONS for {session}: {', '.join(sorted(found_locations))}")

#%% Compare Ripple and HDF locations for each session

def build_location_map(sessions, location_groups):
    locations_by_session = {session: set() for session in sessions}
    for path, group_sessions in location_groups:
        for session in group_sessions:
            locations_by_session[session].add(path)
    return locations_by_session


AIRPORT_RIPPLE_LOCATIONS = build_location_map(
    AIRPORT_SESSIONS,
    [
        (path_santacruz2, AIRPORT_RIPPLE_SESSIONS_1),
        (path_santacruz1, AIRPORT_RIPPLE_SESSIONS_2),
        (path_santacruz3, AIRPORT_RIPPLE_SESSIONS_OTHER_LOCATION),
    ],
)
AIRPORT_HDF_LOCATIONS = build_location_map(
    AIRPORT_SESSIONS,
    [
        (path_santacruz2, AIRPORT_HDF_SESSIONS_1),
        (path_santacruz1, AIRPORT_HDF_SESSIONS_2),
        (path_santacruz3, AIRPORT_HDF_SESSIONS_OTHER_LOCATION),
    ],
)
BRAZOS_RIPPLE_LOCATIONS = build_location_map(
    BRAZOS_SESSIONS,
    [
        (path_santacruz1, BRAZOS_RIPPLE_SESSIONS_1),
        (path_santacruz2, BRAZOS_RIPPLE_SESSIONS_2),
        (path_santacruz3, BRAZOS_RIPPLE_SESSIONS_3),
    ],
)
BRAZOS_HDF_LOCATIONS = build_location_map(
    BRAZOS_SESSIONS,
    [
        (path_santacruz1, BRAZOS_HDF_SESSIONS_1),
        (path_santacruz2, BRAZOS_HDF_SESSIONS_2),
        (path_santacruz3, BRAZOS_HDF_SESSIONS_3),
    ],
)

AIRPORT_LOCATION_MISMATCHES = {}
for session in AIRPORT_SESSIONS:
    ripple_locations = AIRPORT_RIPPLE_LOCATIONS[session]
    hdf_locations = AIRPORT_HDF_LOCATIONS[session]
    if ripple_locations != hdf_locations:
        AIRPORT_LOCATION_MISMATCHES[session] = {
            "ripple": ripple_locations,
            "hdf": hdf_locations,
        }
        print(
            f"Airport location mismatch for {session}: "
            f"Ripple={sorted(ripple_locations)}, HDF={sorted(hdf_locations)}"
        )

BRAZOS_LOCATION_MISMATCHES = {}
for session in BRAZOS_SESSIONS:
    ripple_locations = BRAZOS_RIPPLE_LOCATIONS[session]
    hdf_locations = BRAZOS_HDF_LOCATIONS[session]
    if ripple_locations != hdf_locations:
        BRAZOS_LOCATION_MISMATCHES[session] = {
            "ripple": ripple_locations,
            "hdf": hdf_locations,
        }
        print(
            f"Brazos location mismatch for {session}: "
            f"Ripple={sorted(ripple_locations)}, HDF={sorted(hdf_locations)}"
        )

print(
    f"Location comparison: Airport {len(AIRPORT_LOCATION_MISMATCHES)} "
    f"mismatch(es); Brazos {len(BRAZOS_LOCATION_MISMATCHES)} mismatch(es)."
)

#%%
AIRPORT_SESSIONS_1 = AIRPORT_HDF_SESSIONS_1
AIRPORT_SESSIONS_2 = AIRPORT_HDF_SESSIONS_2

#%% test
for session in AIRPORT_SESSIONS_1:
    try:
        hdf = tables.open_file(os.path.join(path_santacruz2, 'hdf', f'{session}.hdf'))
        hdf.close()
    except Exception as e:
        print(f"Error occurred while processing {session}: {e}")

#%%
for session in AIRPORT_SESSIONS_2:
    try:
        hdf = tables.open_file(os.path.join(path_santacruz1, 'hdf', f'{session}.hdf'))
        hdf.close()
    except Exception as e:
        print(f"Error occurred while processing {session}: {e}")

#%%
for session in BRAZOS_SESSIONS:
    try:
        hdf = tables.open_file(os.path.join(path_santacruz3, 'hdf', f'{session}.hdf'))
        hdf.close()
    except Exception as e:
        print(f"Error occurred while processing {session}: {e}")
# %%
