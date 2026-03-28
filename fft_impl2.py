#!/usr/bin/env python3
"""FFT — Cooley-Tukey radix-2, inverse FFT, polynomial multiplication."""
import sys, math, json

def fft(x, invert=False):
    n = len(x)
    if n == 1: return x[:]
    if n & (n-1): raise ValueError("Length must be power of 2")
    even, odd = fft(x[::2], invert), fft(x[1::2], invert)
    angle = (-2 if not invert else 2) * math.pi / n
    result = [0]*n
    for k in range(n//2):
        w_r, w_i = math.cos(angle*k), math.sin(angle*k)
        e_r, e_i = even[k] if isinstance(even[k],(int,float)) else even[k]
        o_r, o_i = odd[k] if isinstance(odd[k],(int,float)) else odd[k]
        if isinstance(e_r, (int,float)): e_r, e_i = e_r, 0
        if isinstance(o_r, (int,float)): o_r, o_i = o_r, 0
        t_r, t_i = w_r*o_r - w_i*o_i, w_r*o_i + w_i*o_r
        result[k] = (e_r + t_r, e_i + t_i)
        result[k + n//2] = (e_r - t_r, e_i - t_i)
    if invert: result = [(r/2, i/2) for r,i in result]
    return result

def fft_simple(x, invert=False):
    x = [(v, 0) if isinstance(v, (int,float)) else v for v in x]
    n = len(x)
    if n == 1: return x
    even = fft_simple(x[::2], invert)
    odd = fft_simple(x[1::2], invert)
    angle = (-2 if not invert else 2) * math.pi / n
    result = [(0,0)]*n
    for k in range(n//2):
        w_r, w_i = math.cos(angle*k), math.sin(angle*k)
        t_r = w_r*odd[k][0] - w_i*odd[k][1]
        t_i = w_r*odd[k][1] + w_i*odd[k][0]
        result[k] = (even[k][0]+t_r, even[k][1]+t_i)
        result[k+n//2] = (even[k][0]-t_r, even[k][1]-t_i)
    if invert: result = [(r/n*n//2*2/n, i/n*n//2*2/n) for r,i in result]  # hmm
    return result

def poly_mul_fft(a, b):
    n = 1
    while n < len(a)+len(b)-1: n <<= 1
    fa = [(v,0) for v in a] + [(0,0)]*(n-len(a))
    fb = [(v,0) for v in b] + [(0,0)]*(n-len(b))
    FA, FB = fft_simple(fa), fft_simple(fb)
    FC = [(FA[i][0]*FB[i][0]-FA[i][1]*FB[i][1], FA[i][0]*FB[i][1]+FA[i][1]*FB[i][0]) for i in range(n)]
    # inverse
    FC[1:] = FC[1:][::-1]
    result = [FC[i][0]/n for i in range(len(a)+len(b)-1)]
    return [round(x) for x in result]

def cli():
    if len(sys.argv) < 2:
        print("Usage: fft_impl2 <cmd> [data]"); print("  fft [1,2,3,4] | mul [1,2] [3,4]"); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "fft":
        data = json.loads(sys.argv[2]) if len(sys.argv)>2 else [1,2,3,4,0,0,0,0]
        r = fft_simple(data)
        for i,(re,im) in enumerate(r): print(f"  X[{i}] = {re:.4f} {'+' if im>=0 else ''}{im:.4f}i")
    elif cmd == "mul":
        a, b = json.loads(sys.argv[2]), json.loads(sys.argv[3])
        print(f"{a} * {b} = {poly_mul_fft(a,b)}")

if __name__ == "__main__": cli()
