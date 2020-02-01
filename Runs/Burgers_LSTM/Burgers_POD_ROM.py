import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(5)
tf.random.set_random_seed(5)

from POD import generate_pod_bases, plot_pod_modes
from GP import galerkin_projection
from LSTM_NETS import lstm_for_dynamics,evaluate_rom_deployment_lstm

Rnum = 1000.0
x = np.linspace(0.0,1.0,num=128)
dx = 1.0/np.shape(x)[0]

tsteps = np.linspace(0.0,2.0,num=400)
dt = 2.0/np.shape(tsteps)[0]
num_modes = 3 
num_epochs = 500
reg_param = 0.001
reg_param_fnl = 0.001
# Deployment mode
deployment_mode = 'test' # or train

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the Burgers problem definition
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def exact_solution(t):
    t0 = np.exp(Rnum/8.0)

    return (x/(t+1))/(1.0+np.sqrt((t+1)/t0)*np.exp(Rnum*(x*x)/(4.0*t+4)))

def collect_snapshots():
    snapshot_matrix = np.zeros(shape=(np.shape(x)[0],np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix[:,t] = exact_solution(tsteps[t])[:]

    snapshot_matrix_mean = np.mean(snapshot_matrix,axis=1)
    snapshot_matrix = (snapshot_matrix.transpose()-snapshot_matrix_mean).transpose()

    return snapshot_matrix, snapshot_matrix_mean

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the ROM assessment
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Snapshot collection
    sm, sm_mean = collect_snapshots() # Note that columns of a snapshot/state are time always and a state vector is a column vector

    # Truth
    phi_trunc, cf_trunc = generate_pod_bases(sm,num_modes,tsteps)
    perfect_output = cf_trunc[:,-1]

    np.save('POD_Modes.npy',phi_trunc)
    np.save('Snapshot_Mean.npy',sm_mean)
    np.save('Burgers_Coefficients_400.npy',cf_trunc)

    # # POD Galerkin - for comparison
    output_state_gp, state_tracker_gp = galerkin_projection(phi_trunc,cf_trunc,sm_mean,tsteps,Rnum,dt,dx,num_modes)

    np.save('Burgers_GP_Coefficients_400.npy',state_tracker_gp)

    # LSTM network
    model = lstm_for_dynamics(cf_trunc,num_epochs,10,deployment_mode)
    output_state_lstm, state_tracker_lstm = evaluate_rom_deployment_lstm(model,cf_trunc,tsteps,num_modes,10)

    np.save('LSTM_Coefficients_400.npy',state_tracker_lstm)

    #Visualization
    plt.figure()
    plt.plot(x[:],sm_mean+(np.matmul(phi_trunc,perfect_output))[:],label='Truth')
    plt.plot(x[:],sm_mean+(np.matmul(phi_trunc,output_state_gp))[:],label='POD-GP')
    plt.plot(x[:],sm_mean+(np.matmul(phi_trunc,output_state_lstm[:,0]))[:],label='POD-LSTM')
    plt.legend(fontsize=18)
    plt.show()

    #Visualization of state stabilization - modal coefficient 3
    mode_num = 2
    plt.figure()
    plt.plot(cf_trunc[mode_num,:-1],label='Truth')
    plt.plot(state_tracker_gp[mode_num,:-1],label='POD-GP')
    plt.plot(state_tracker_lstm[mode_num,:-1],label='POD-LSTM')
    plt.legend()
    plt.show()