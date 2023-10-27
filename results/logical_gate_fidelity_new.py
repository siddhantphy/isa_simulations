import json
import sys
import numpy as np
import os
# Get all the data from the output files

def fidelity_calculator_function(carbon_value,electron_value,operation_space = ""):
    carbon_value = {"noiseless":'0',"noisefull":str(carbon_value)}
    electron_value = {"noiseless":'0',"noisefull":str(electron_value)}
    
    # print(f"the carbon value is {carbon_value}")
    # print(sys.path)
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    dir = dir.replace('results', 'json_data_storage')
    for experiment in {"noiseless", "noisefull"}:
        for input_state in {'0','1','plus','minus','iplus','iminus'}:
            carbon_read = carbon_value[experiment]
            electron_read = electron_value[experiment]

            # with open(f"ISA_simulator/json_data_storage/process_tomography_p{input_state}_X_{experiment}_{carbon_read}_newest.json") as json_data:
            with open(f"{dir}/process_tomography_p{input_state}_X_{experiment}_{carbon_read}_newest_with_electron_dec{electron_read}{str(operation_space)}_supercomputer.json") as json_data:

                full_data = json.load(json_data)
                successful_iterations = len(full_data['P_values'])
                if successful_iterations == 0:
                    sys.exit(f"Do more iterations for the experiment for input state: {input_state} with coherence values {carbon_read} carbon and {electron_read} electron. No success was recorded so far!")
            if input_state == '0':
                p_0 = np.array(full_data["P_values"])
                p_0 = (p_0.sum(axis=0))/successful_iterations
            elif input_state == '1':
                p_1 = np.array(full_data["P_values"])
                p_1 = (p_1.sum(axis=0))/successful_iterations
            elif input_state == 'plus':
                p_plus = np.array(full_data["P_values"])
                p_plus = (p_plus.sum(axis=0))/successful_iterations
            elif input_state == 'minus':
                p_minus = np.array(full_data["P_values"])
                p_minus = (p_minus.sum(axis=0))/successful_iterations
            elif input_state == 'iplus':
                p_i_plus = np.array(full_data["P_values"])
                p_i_plus = (p_i_plus.sum(axis=0))/successful_iterations
            elif input_state == 'iminus':
                p_i_minus = np.array(full_data["P_values"])
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
        
    gate_fidelity = (np.trace(noisy_lptm.conj().T @ noiseless_lptm)+2)/6


    print(gate_fidelity)
    return gate_fidelity
if __name__ == "__main__":
    
    fidelity_calculator_function(5e2,5e1,"physical")