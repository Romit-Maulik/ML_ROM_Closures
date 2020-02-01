import numpy as np
import matplotlib.pyplot as plt

mse_train = np.load('Train_Loss_NODE.npy')
mse_val = np.load('Val_Loss_NODE.npy')

plt.figure()
plt.semilogy(mse_train,label='Training')
plt.semilogy(mse_val,label='Validation')
plt.title('Error convergence - NODE')
plt.legend(fontsize=12)
plt.show()