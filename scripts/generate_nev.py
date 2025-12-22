from preprocess_santacruz_data import _generate_nev_output
from sessions import BRAZOS_SESSIONS, AIRPORT_SESSIONS_1, AIRPORT_SESSIONS_2
# session = 'airp20211202_04_te1598'


# _generate_nev_output(BRAZOS_SESSIONS, count=False, 
#                      input_folder = r"J:\storage\rawdata\ripple", 
#                      output_folder = r"F:\cole\neuron_tracking_nev_outputs")

_generate_nev_output(AIRPORT_SESSIONS_1, count=False, 
                     input_folder = r"K:\storage\rawdata\ripple", 
                     output_folder = r"F:\cole\neuron_tracking_nev_outputs")

_generate_nev_output(AIRPORT_SESSIONS_2, count=False, 
                     input_folder = r"L:\storage\rawdata\ripple", 
                     output_folder = r"F:\cole\neuron_tracking_nev_outputs")


braz_last_sessions = ['braz20220623_04_te515','braz20220624_06_te521','braz20220627_04_te529']

_generate_nev_output(braz_last_sessions, count=False, 
                      input_folder = r"J:\storage\rawdata\ripple", 
                      output_folder = r"F:\cole\neuron_tracking_nev_outputs")



# airport 64 electrodes
# brazos 128 electrodes
