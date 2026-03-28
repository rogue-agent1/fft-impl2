#!/usr/bin/env python3
"""FFT implementation with inverse and spectral analysis."""
import math
def fft(x):
    N=len(x)
    if N<=1: return x
    even=fft(x[0::2]);odd=fft(x[1::2])
    T=[complex(math.cos(-2*math.pi*k/N),math.sin(-2*math.pi*k/N))*odd[k] for k in range(N//2)]
    return [even[k]+T[k] for k in range(N//2)]+[even[k]-T[k] for k in range(N//2)]
def ifft(X):
    N=len(X);conj=[x.conjugate() for x in X];result=fft(conj)
    return [x.conjugate()/N for x in result]
def magnitude(X): return [abs(x) for x in X]
def phase(X): return [math.atan2(x.imag,x.real) for x in X]
def power_spectrum(x):
    X=fft(x);N=len(X);return [abs(X[k])**2/N for k in range(N//2)]
if __name__=="__main__":
    N=256;sr=256;signal=[math.sin(2*math.pi*10*i/sr)+0.5*math.sin(2*math.pi*25*i/sr) for i in range(N)]
    X=fft([complex(x) for x in signal]);ps=power_spectrum([complex(x) for x in signal])
    peak=max(range(len(ps)),key=lambda i:ps[i])
    print(f"FFT: dominant freq={peak}Hz (expected 10Hz)")
    reconstructed=ifft(X)
    err=max(abs(reconstructed[i].real-signal[i]) for i in range(N))
    assert err<1e-10;print(f"IFFT reconstruction error: {err:.2e}")
    print("FFT OK")
