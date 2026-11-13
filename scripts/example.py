import tables

path = "data/HDF/braz20220315_07_te90.hdf"

with tables.open_file(path, mode="r") as hdf:
    print(hdf)  # File structure
    print(hdf.root.task._v_attrs)  # Task attributes
    print("Rotation:", hdf.root.task._v_attrs.rot_angle_deg)
    print("Task columns:", hdf.root.task.colnames)
    print("First 10 messages:", hdf.root.task_msgs[:10])