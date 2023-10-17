import numpy as np

n = 1000 # size of the networks
gamma = 2.7 
mean_degree = 10

kappa_0 = (
    (1 - 1 / n)
    / (1 - n ** ((2 - gamma) / (gamma - 1)))
    * (gamma - 2)
    / (gamma - 1)
    * mean_degree
)
kappa_c = kappa_0 * n ** (1 / (gamma - 1))

kappas = []
for _ in range(n):
    kappas.append(
        kappa_0
        * (1 - np.random.uniform(0, 1) * (1 - (kappa_c / kappa_0) ** (1 - gamma)))
        ** (1 / (1 - gamma))
    )


# with open('test.txt', 'w') as f:
#     for k in kappas:
#         f.write(f'{k}\n')
    
print(kappas)
