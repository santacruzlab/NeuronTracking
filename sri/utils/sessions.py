#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:41:32 2024

@author: hungyunlu
"""

import tables
import os

UTILS_FOLDER = os.path.dirname(os.path.abspath(__file__))
SRI_FOLDER = os.path.dirname(UTILS_FOLDER)
PROJECT_FOLDER = os.path.dirname(SRI_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
INPUT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'input_data')
HDF_FOLDER = os.path.join(INPUT_DATA_FOLDER, 'hdf')

AIRPORT_ROTATION = []
BRAZOS_ROTATION = []

AIRPORT_SESSIONS_1 = [ #santacruz2 on storage server
    'airp20211126_05_te1576',
    'airp20211129_04_te1580',
    'airp20211130_09_te1589',
    'airp20211201_05_te1594',
    'airp20211202_04_te1598',
    'airp20211203_07_te1605',
    'airp20211205_04_te1609',
    'airp20211206_09_te1618',
    'airp20211207_04_te1622',
    'airp20211208_06_te1628',
    'airp20211209_04_te1632',
    'airp20211210_05_te1637',
    'airp20211211_04_te1641',
    'airp20211212_13_te1654',
    'airp20211213_04_te1658',
    'airp20211215_04_te1670',
    'airp20211216_04_te1674',
    'airp20211217_04_te1678',
    'airp20211220_04_te1688',
    'airp20211221_04_te1692',
    'airp20211223_04_te1700',
    'airp20211224_04_te1704',
    'airp20211229_04_te1717',
    'airp20220103_04_te1745',
    'airp20220107_04_te1761',
    'airp20220109_04_te1765',
    'airp20220110_05_te1770',
    'airp20220112_04_te1780',
    'airp20220113_04_te1784',
    'airp20220114_04_te1788',
    'airp20220116_04_te1792',
    'airp20220117_06_te1798',
    'airp20220118_04_te1802',
    'airp20220119_05_te1807'
    ]

AIRPORT_SESSIONS_2 = [ #santacruz1 on storage server
    'airp20220128_04_te1756',
    'airp20220201_04_te1780',
    'airp20220202_04_te1784',
    'airp20220205_05_te1789',
    'airp20220207_04_te1798',
    'airp20220211_05_te1812',
    'airp20220212_04_te1816',
    'airp20220213_04_te1820',
    'airp20220214_04_te1824',
    'airp20220215_06_te1830',
    'airp20220216_04_te1834',
    'airp20220218_04_te1842',
    'airp20220221_04_te1846',
    'airp20220222_05_te1851',
    'airp20220224_10_te1861',
    'airp20220225_04_te1865',
    'airp20220226_05_te1870',
    'airp20220227_04_te1874',
    'airp20220228_04_te1878',
    'airp20220303_05_te1887',
    'airp20220304_05_te1892',
    'airp20220306_04_te1907',
    'airp20220308_04_te1915',
    'airp20220312_04_te1938',
    'airp20220315_04_te1995',
    'airp20220319_14_te2079',
    'airp20220322_04_te2103',
    'airp20220323_04_te2107',
    'airp20220324_04_te2111'
    ]


AIRPORT_SESSIONS = AIRPORT_SESSIONS_1 + AIRPORT_SESSIONS_2

BRAZOS_SESSIONS = [ #santacruz3 on storage server
    'braz20220315_07_te90',
    'braz20220316_06_te96',
    'braz20220318_07_te108',
    'braz20220319_04_te112',
    'braz20220321_04_te116',
    'braz20220324_19_te176',
    'braz20220328_04_te206',
    'braz20220401_05_te220',
    'braz20220404_07_te231',
    'braz20220405_04_te235',
    'braz20220406_04_te239',
    'braz20220407_04_te243',
    'braz20220408_05_te248',
    'braz20220411_04_te257',
    'braz20220412_05_te262',
    'braz20220414_04_te286',
    'braz20220416_04_te294',
    'braz20220418_05_te299',
    'braz20220421_04_te312',
    'braz20220422_04_te316',
    'braz20220425_04_te320',
    'braz20220426_05_te325',
    'braz20220427_04_te329',
    'braz20220428_04_te333',
    'braz20220429_04_te337',
    'braz20220504_04_te376',
    'braz20220505_04_te380',
    'braz20220506_05_te387',
    'braz20220507_04_te391',
    'braz20220509_04_te395',
    'braz20220510_04_te399',
    'braz20220511_04_te408',
    'braz20220512_04_te412',
    'braz20220513_05_te417',
    'braz20220514_04_te421',
    'braz20220516_04_te425',
    'braz20220517_05_te432',
    'braz20220518_04_te436',
    'braz20220519_05_te441',
    'braz20220520_04_te445',
    'braz20220521_04_te449',
    'braz20220606_04_te458',
    'braz20220607_04_te462',
    'braz20220608_04_te466',
    'braz20220609_04_te470',
    'braz20220611_04_te478',
    'braz20220613_04_te482',
    'braz20220614_04_te486',
    'braz20220615_04_te490',
    'braz20220617_04_te498',
    'braz20220620_05_te503',
    'braz20220622_04_te511',
    'braz20220623_04_te515',
    'braz20220624_06_te521',
    'braz20220626_04_te525',
    'braz20220627_04_te529',
    'braz20220628_04_te533',
    ]

for session in AIRPORT_SESSIONS:
    try: 
        hdf = tables.open_file(os.path.join(HDF_FOLDER, f"{session}.hdf"), mode='r')
        AIRPORT_ROTATION.append(hdf.root.task._v_attrs.rot_angle_deg)
        hdf.close()
    except Exception as e:
        print(f"Error retrieving rotation for session {session}: {e}")
        AIRPORT_ROTATION.append(None)  # Append None or a default value if there's an error

for session in BRAZOS_SESSIONS:
    try: 
        hdf = tables.open_file(os.path.join(HDF_FOLDER, f"{session}.hdf"), mode='r')
        BRAZOS_ROTATION.append(hdf.root.task._v_attrs.rot_angle_deg)
        hdf.close()
    except Exception as e:
        print(f"Error retrieving rotation for session {session}: {e}")
        BRAZOS_ROTATION.append(None)  # Append None or a default value if there's an error

# for rot in set(AIRPORT_ROTATION):
#     count = AIRPORT_ROTATION.count(rot)
#     print(f"Rotation angle {rot} degrees: {count} sessions in AIRPORT_SESSIONS")

# for rot in set(BRAZOS_ROTATION):
#     count = BRAZOS_ROTATION.count(rot)
#     print(f"Rotation angle {rot} degrees: {count} sessions in BRAZOS_SESSIONS")