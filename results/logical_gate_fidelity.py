import json
import sys
import numpy as np

# Get all the data from the output files
for experiment in {"noiseless", "noisefull"}:
    for input_state in {'0','1','plus','minus','iplus','iminus'}:
        with open(f"ISA_simulator/json_data_storage/process_tomography_p{input_state}_{experiment}.json") as json_data:
            full_data = json.load(json_data)
            successful_iterations = len(full_data['P_value'])
            if successful_iterations == 0:
                sys.exit(f"Do more iterations for the experiment for input state: {input_state}. No success was recorded so far!")
        if input_state == '0':
            p_0 = np.array(full_data["P_value"])
            p_0 = (p_0.sum(axis=0))/successful_iterations
        elif input_state == '1':
            p_1 = np.array(full_data["P_value"])
            p_1 = (p_1.sum(axis=0))/successful_iterations
        elif input_state == 'plus':
            p_plus = np.array(full_data["P_value"])
            p_plus = (p_plus.sum(axis=0))/successful_iterations
        elif input_state == 'minus':
            p_minus = np.array(full_data["P_value"])
            p_minus = (p_minus.sum(axis=0))/successful_iterations
        elif input_state == 'iplus':
            p_i_plus = np.array(full_data["P_value"])
            p_i_plus = (p_i_plus.sum(axis=0))/successful_iterations
        elif input_state == 'iminus':
            p_i_minus = np.array(full_data["P_value"])
            p_i_minus = (p_i_minus.sum(axis=0))/successful_iterations
        else:
            sys.exit("Error in reading and assigning data!")


    lptm = np.identity(4)

    lptm[0, 0] = 1
    lptm[0, 1] = lptm[0, 2] = lptm[0, 3] = 0
    lptm[1, 1] = 0.5 * (p_plus[0] - p_minus[0])
    lptm[2, 2] = 0.5 * (p_i_plus[1] - p_i_minus[1])
    lptm[3, 3] = 0.5 * (p_0[2] - p_1[2])
    lptm[1, 0] = 0.5 * (p_0[0] + p_1[0])
    lptm[2, 0] = 0.5 * (p_0[1] + p_1[1])
    lptm[3, 0] = 0.5 * (p_0[2] + p_1[2])
    lptm[2, 1] = 0.5 * (p_plus[1] - p_minus[1])
    lptm[3, 1] = 0.5 * (p_plus[2] - p_minus[2])
    lptm[1, 2] = 0.5 * (p_i_plus[0] - p_i_minus[0])
    lptm[3, 2] = 0.5 * (p_i_plus[2] - p_i_minus[2])
    lptm[1, 3] = 0.5 * (p_0[0] - p_1[0])
    lptm[2, 3] = 0.5 * (p_0[1] - p_1[1])

    if experiment == "noiseless":
        noiseless_lptm = lptm
    elif experiment == "noisefull":
        noisy_lptm = lptm
    else:
        sys.exit("Experiment parameter error!")
    
gate_fidelity = (np.trace(noiseless_lptm.conj().T @ noiseless_lptm)+2)/6


print(gate_fidelity)