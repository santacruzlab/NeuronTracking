#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:41:32 2024

@author: hungyunlu
"""

AIRPORT_ROTATION = [50] * 3 + [90] * 9 + [310] * 13 + [270] * 9
BRAZOS_ROTATION = [310] * 11 + [270] * 12 + [50] * 9 + [90] * 12


AIRPORT_SESSIONS_1 = [ #santacruz2 on storage server
    
    'airp20211202_04_te1598',
    'airp20211205_04_te1609',
    'airp20211210_05_te1637',
    'airp20211216_04_te1674',
    'airp20211217_04_te1678',
    'airp20211223_04_te1700',
    'airp20211224_04_te1704',
    'airp20220107_04_te1761',
    'airp20220109_04_te1765',
    'airp20220110_05_te1770',
    'airp20220114_04_te1788',
    'airp20220116_04_te1792',
    'airp20220119_05_te1807'
    
]

AIRPORT_SESSIONS_2 = [ #santacruz1 on storage server
    
    'airp20220201_04_te1780',
    'airp20220202_04_te1784',
    'airp20220205_05_te1789',
    'airp20220207_04_te1798', 
    'airp20220211_05_te1812',
    'airp20220212_04_te1816',
    'airp20220213_04_te1820',
    'airp20220216_04_te1834',
    'airp20220218_04_te1842',
    'airp20220221_04_te1846',
    'airp20220222_05_te1851',
    'airp20220224_10_te1861',
    'airp20220303_05_te1887',
    'airp20220306_04_te1907',
    'airp20220312_04_te1938',
    'airp20220313_10_te1948',
    'airp20220315_04_te1995',
    'airp20220319_14_te2079',
    'airp20220322_04_te2103',
    'airp20220323_04_te2107',
    'airp20220324_04_te2111',
    
]


AIRPORT_SESSIONS = [ #santacruz2/santacruz1 on storage server
    
    'airp20211202_04_te1598',
    'airp20211205_04_te1609',
    'airp20211210_05_te1637',
    'airp20211216_04_te1674',
    'airp20211217_04_te1678',
    'airp20211223_04_te1700',
    'airp20211224_04_te1704',
    'airp20220107_04_te1761',
    'airp20220109_04_te1765',
    'airp20220110_05_te1770',
    'airp20220114_04_te1788',
    'airp20220116_04_te1792',
    'airp20220119_05_te1807',    
    'airp20220201_04_te1780',
    'airp20220202_04_te1784',
    'airp20220205_05_te1789',
    'airp20220207_04_te1798', 
    'airp20220211_05_te1812',
    'airp20220212_04_te1816',
    'airp20220213_04_te1820',
    'airp20220216_04_te1834',
    'airp20220218_04_te1842',
    'airp20220221_04_te1846',
    'airp20220222_05_te1851',
    'airp20220224_10_te1861',
    'airp20220303_05_te1887',
    'airp20220306_04_te1907',
    'airp20220312_04_te1938',
    'airp20220313_10_te1948',
    'airp20220315_04_te1995',
    'airp20220319_14_te2079',
    'airp20220322_04_te2103',
    'airp20220323_04_te2107',
    'airp20220324_04_te2111'
    
]

BRAZOS_SESSIONS = [ #santacruz3 on storage server
    
    'braz20220315_07_te90',
    'braz20220316_06_te96',
    'braz20220318_07_te108',
    'braz20220319_04_te112',
    'braz20220321_04_te116',
    'braz20220324_19_te176',
    'braz20220328_04_te206',
    'braz20220331_04_te215',
    'braz20220401_05_te220',
    'braz20220405_04_te235',
    'braz20220407_04_te243',
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
    'braz20220507_04_te391',
    'braz20220510_04_te399',
    'braz20220511_04_te408',
    'braz20220512_04_te412',
    'braz20220514_04_te421',
    'braz20220516_04_te425',
    'braz20220517_05_te432',
    'braz20220518_04_te436',
    'braz20220520_04_te445',
    'braz20220607_04_te462',
    'braz20220608_04_te466',
    'braz20220609_04_te470',
    'braz20220611_04_te478',
    'braz20220614_04_te486',
    'braz20220615_04_te490',
    'braz20220617_04_te498',
    'braz20220620_05_te503',
    'braz20220622_04_te511',
    'braz20220623_04_te515',
    'braz20220624_06_te521',
    'braz20220627_04_te529'
    
    ]

# AIRPORT_SESSIONS = [ #santacruz2/santacruz1 on storage server
    
#     'airp20211205_04_te1609'
    
# ]

# BRAZOS_SESSIONS = [ #santacruz3 on storage server
    
#     'braz20220315_07_te90',
#     'braz20220316_06_te96',
#     'braz20220318_07_te108',
#     'braz20220319_04_te112',
#     'braz20220321_04_te116',
#     'braz20220324_19_te176',
#     'braz20220328_04_te206'
    
#     ]