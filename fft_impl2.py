#!/usr/bin/env python3
"""fft_impl2 - FFT, inverse FFT, convolution, spectral analysis."""
import sys, argparse, math, json

def fft(x):
    n = len(x)
    if n <= 1: return x
    even = fft(x[0::2]); odd = fft(x[1::2])
    T = [complex(math.cos(-2*math.pi*k/n), math.sin(-2*math.pi*k/n)) * odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + [even[k] - T[k] for k in range(n//2)]

def ifft(X):
    n = len(X)
    conj = [x.conjugate() for x in X]
    result = fft(conj)
    return [x.conjugate() / n for x in result]

def pad_pow2(x):
    n = 1
    while n < len(x): n <<= 1
    return x + [0] * (n - len(x))

def convolve(a, b):
    n = 1
    while n < len(a) + len(b) - 1: n <<= 1
    fa = fft(pad_pow2([complex(x) for x in a] + [0]*(n-len(a))))
    fb = fft(pad_pow2([complex(x) for x in b] + [0]*(n-len(b))))
    fc = [a*b for a,b in zip(fa, fb)]
    result = ifft(fc)
    return [round(x.real, 10) for x in result[:len(a)+len(b)-1]]

def magnitude_spectrum(x, sample_rate=1.0):
    X = fft(pad_pow2([complex(v) for v in x]))
    n = len(X)
    freqs = [i * sample_rate / n for i in range(n//2)]
    mags = [abs(X[i]) * 2 / n for i in range(n//2)]
    return list(zip(freqs, mags))

def main():
    p = argparse.ArgumentParser(description="FFT toolkit")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        print("=== FFT of sine wave ===")
        sr = 128; dur = 1; freq = 5
        signal = [math.sin(2*math.pi*freq*t/sr) for t in range(sr*dur)]
        spectrum = magnitude_spectrum(signal, sr)
        top = sorted(spectrum, key=lambda x: -x[1])[:3]
        print(f"Top frequencies: {[(round(f,1), round(m,3)) for f,m in top]}")

        print("\n=== Convolution ===")
        a = [1, 2, 3, 4]; b = [1, 0, -1]
        print(f"{a} * {b} = {convolve(a, b)}")

        print("\n=== Round-trip FFT->IFFT ===")
        orig = [1, 2, 3, 4, 5, 6, 7, 8]
        recovered = [round(x.real, 6) for x in ifft(fft([complex(v) for v in orig]))]
        print(f"Original:  {orig}")
        print(f"Recovered: {recovered}")
    else: p.print_help()

if __name__ == "__main__":
    main()
