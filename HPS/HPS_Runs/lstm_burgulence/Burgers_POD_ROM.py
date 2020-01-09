import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(5)
tf.random.set_random_seed(5)

from POD import generate_pod_bases, plot_pod_modes
from GP import galerkin_projection
from LSTM_NETS import lstm_for_dynamics,evaluate_rom_deployment_lstm

# Some parameters
nx = 2048
alpha = 2.0e-3
Rnum = 1.0/alpha
x = np.linspace(0.0,2.0*np.pi,num=2048)
dx = 2.0*np.pi/np.shape(x)[0]
dt = 0.2/400
tsteps = np.linspace(0.0,0.2,num=int(0.2/dt))
num_modes = 3

# Deployment mode
deployment_mode = 'test' # or 'train'

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Calculate energy spectra
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def spectra_calculation(u):
    # Transform to Fourier space
    array_hat = np.real(np.fft.fft(u))

    # Normalizing data
    array_new = np.copy(array_hat / float(nx))
    # Energy Spectrum
    espec = 0.5 * np.absolute(array_new)**2
    # Angle Averaging
    eplot = np.zeros(nx // 2, dtype='double')
    for i in range(1, nx // 2):
        eplot[i] = 0.5 * (espec[i] + espec[nx - i])

    return eplot

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Collect data from DNS
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def collect_snapshots():
    snapshot_matrix = np.load('Dataset.npy')[:,:-1]
    trange = np.arange(np.shape(tsteps)[0])

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

    # Truth - each column of phi spans the global domain
    phi_trunc, cf_trunc = generate_pod_bases(sm,num_modes,tsteps)
    perfect_output = cf_trunc[:,-1]

    # # POD Galerkin - for comparison
    output_state_gp, state_tracker_gp = galerkin_projection(phi_trunc,cf_trunc,sm_mean,tsteps,Rnum,dt,dx,num_modes)

    # # LSTM network - note this will only give good predictions till the last three timesteps
    model = lstm_for_dynamics(cf_trunc,deployment_mode)
    output_state_lstm, state_tracker_lstm = evaluate_rom_deployment_lstm(model,cf_trunc,tsteps)
    np.save('Burgulence_LSTM_Coefficients.npy',state_tracker_lstm)

    #Visualization - Spectra
    u_true = sm_mean+(np.matmul(phi_trunc,perfect_output))[:]
    u_gp = sm_mean+(np.matmul(phi_trunc,output_state_gp))[:]
    u_lstm = sm_mean+(np.matmul(phi_trunc,output_state_lstm[:,0]))[:]

    plt.figure()
    kx_plot = np.array([float(i) for i in list(range(0, nx // 2))])
    espec1=spectra_calculation(u_true)
    espec2=spectra_calculation(u_gp)
    espec3=spectra_calculation(u_lstm)
    plt.loglog(kx_plot, espec1,label='Truth')
    plt.loglog(kx_plot, espec2,label='GP')
    plt.loglog(kx_plot, espec3,label='LSTM')
    plt.legend()
    plt.show()

    # Spectra residuals
    plt.figure()
    kx_plot = np.array([float(i) for i in list(range(0, nx // 2))])
    plt.loglog(kx_plot, np.abs(espec2-espec1),label='GP-Residual')
    plt.loglog(kx_plot, np.abs(espec3-espec1),label='LSTM-Residual')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x[:],u_true[:],label='Truth')
    plt.plot(x[:],u_gp[:],label='POD-GP')
    plt.plot(x[:],u_lstm[:],label='POD-LSTM')
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