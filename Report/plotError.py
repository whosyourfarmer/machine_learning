import numpy as np
import matplotlib.pyplot as plt

validation_error = [4.66615137321,4.19427355463,4.11060899445,4.09090626218,4.116596404,4.11188930272]
train_error = [1.84917357477,1.61870618277,1.58248015607,1.57205547512,1.57721614483,1.58131308385]
ind = range(8)
plt.xticks(ind,('0.1','0.3','0.5','0.7','0.9','1.0'))
plt.plot(train_error,label='train_error')
plt.plot(validation_error,label='validation_error')
plt.legend()
plt.xlabel("parameter value")
plt.ylabel("M.A.E.")
plt.show()

