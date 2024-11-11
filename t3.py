import cytnx

print(cytnx.Device.Ngpus)

import matplotlib.pyplot as plt

dcut = 2

plt.figure(figsize = (10,4),dpi = 300)
plt.suptitle(f'dcut={dcut}',fontsize = 16)
plt.subplot(121)
plt.subplot(122)
plt.show()
plt.savefig(f'{dcut}.png')