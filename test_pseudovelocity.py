import endaq.calc.pseudovelocity
import scipy.signal
import numpy as np

import tqdm
import time

fs = 1e3

freqs = np.linspace(1, 1e2, 1000)
omegas = 2 * np.pi * freqs
damp = 1/20

sos = np.ones((len(omegas), 6))

for i, omega in enumerate(omegas):
    tf = scipy.signal.TransferFunction([-1], [1, 2*damp*1000, 1000**2]).to_discrete(dt=1/fs)
    sos[i, :2] = tf.num
    sos[i, 2] = 0
    sos[i, 3:] = tf.den


# sos = scipy.signal.butter(2, .1, output='sos')
z = np.zeros((sos.shape[0], 2))
x = 100000*np.random.standard_normal((int(1e6),))
x[0] = 1

t1 = time.time()
endaq.calc.pseudovelocity._ps_bank(sos, x, z)
tC = time.time() - t1
print(tC)

time.sleep(1)

t0 = time.time()
expected = np.zeros_like(z)
for i, s in tqdm.tqdm(enumerate(sos), total=len(sos)):
    _x = scipy.signal.sosfilt([s], x)
    expected[i, 0] = _x.max()
    expected[i, 1] = _x.min()
tPy = time.time() - t0
print(tPy)
print(f'{100*(1-tC/tPy):3.3f}')

# print(z)
# print(expected)

print(np.abs(z - expected).max())
