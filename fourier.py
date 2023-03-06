import math
import numpy
pi = math.pi

def Exp(Val):
    a = complex(numpy.cos(Val), numpy.sin(Val))
    return a

#pads dataset to the next power of 2
def PadDataset(Arr):

    Arr = list(Arr)
    N = len(Arr)
    P = 2**(int(numpy.floor(numpy.log2(N)) + 1))
    for i in range(P-N):
        Arr.append(0)
    return Arr

#naive DFT
def Naive(Arr):
    N = len(Arr)
    Transformed = [0] * N
    for iterate in range(N):
        a = complex(0,0)
        for iterate2 in range(N):
            a += (Exp(-1 * tau * iterate * iterate2 / N) * Arr[iterate2])
        Transformed[iterate] = a
    return Transformed

#naive inverse DFT
def InvNaive(Arr):
    N = len(Arr)
    Transformed = [0] * N
    for iterate in range(N):
        a = complex(0,0)
        for iterate2 in range(N):
            a += (Exp(tau * iterate * iterate2 / N) * Arr[iterate2])
        Transformed[iterate] = a / N
    return Transformed

#Cooley-Tukey algorithm!
def CooleyTukey(Arr):
    N = len(Arr)
    
    if N == 1:
        return[Arr[0]]
    
    Transformed = [0] * N
    even = CooleyTukey(Arr[0::2])
    odd = CooleyTukey(Arr[1::2])
    
    for iterate in range(N//2):
        X = Exp(-1*tau*iterate/N)
        
        Transformed[iterate] = even[iterate] + X * odd[iterate]
        Transformed[iterate + N//2] = even[iterate] - X * odd[iterate]
        
    return Transformed

def InverseCooleyTukey(Arr):
    a = CooleyTukey(Arr)
    for i in range(len(a)):
        a[i] /= len(a)
    return a

#finds all prime factors of a positive integer
    #uses trial division, other (faster) implementations are available
def FindPrimeFactors(N):
    bound = int(numpy.floor(numpy.sqrt(N)))
    factors = []
    trial = 2
    while trial <= bound:
        if N % trial == 0:
            factors.append(trial)
            while N % trial == 0:
                N = N / trial
        trial += 1
            
    return factors

#finds primitive root of a positive integer
def FindPrimitiveRoot(N):
    a = FindPrimeFactors(N-1)
    for i in range(2,N-1):
        if (i ** (N-1)) % N == 1:
            for j in a:
                check = True
                n = i
                for p in range(int((N-1)/j-1)):
                    n = n * i % N
                if n == 1:
                #if (i**((N-1)/j)) % N == 1:
                    check = False
            if check:
                return i

#Computes sequence of gen ** n % M for some modulus M and generator element gen
def Sequence(N, gen):
    a = [0] * (N-1)
    a[0] = 1
    for i in range(N-2):
        a[i+1] = a[i] * gen % N
    return a

#rader's algorithm!
def rader(Arr):
    N = len(Arr)

    #Finds generator element
    Generator = FindPrimitiveRoot(N)
    GeneratorSeq = Sequence(N,Generator)

    #Finds inverse element
    Inverse = (Generator ** (N-2))% N
    InverseSeq = Sequence(N,Inverse)

    #It's convolution time
    a = Exp(tau / N)
    U = [a ** (InverseSeq[i]) for i in range(len(InverseSeq))]
    V = [Arr[GeneratorSeq[i]] for i in range(len(GeneratorSeq))]

    #Pads datasets so Cooley-Tukey algorithm can be performed
    for i in range(2):
        U = PadDataset(U)
        V = PadDataset(V)

    #Calculates Fourier Transform of each part of convolution
    U = CooleyTukey(U)
    V = CooleyTukey(V)

    #calculates convolution
    Conv = [U[i] * V[i] for i in range(len(U))]

    #takes inverse fourier transform of convolution
    Conv = InverseCooleyTukey(Conv)
    Trans = [Conv[i] + Conv[i+N-1] for i in range(N-1)]

    Transformed = [0] * N
    for i in Arr:
        Transformed[0] += i
    for i in range(N-1):
        Transformed[InverseSeq[i]] = Arr[0] + Trans[i]
    return Transformed

#bluestein's algorithm!
def bluestein(Arr):

    N = len(Arr)
    U = [0] * N
    V = [0] * N
    Chirp = [0] * N

    #computes chirp sequence
    for i in range(N):
        Chirp[i] = Exp(-1 * (i**2) * pi / 2 / N)

    #computes sequences for convolution
    for i in range(N):
        a = (i**2) * pi / N
        U[i] = Arr[i] * Exp(-a)
        V[i] = Exp(a)

    #pads each sequence as required
    for i in range(2):
        U = PadDataset(U)
        V = PadDataset(V)

    for i in range(N):
        V[-(i+1)] = V[i]

    #calculates fourier transforms of each sequence
    U = CooleyTukey(U)
    V = CooleyTukey(V)

    #calculates convolution
    Conv = [U[i] * V[i] for i in range(len(U))]
    #takes inverse fourier transform
    Conv = InverseCooleyTukey(Conv)
    #shrinks transformed sequence to its original size
    Conv = Conv[:N]

    #calculates DFT
    Transformed = [Conv[i] * Chirp[i] for i in range(N)]
        
    return Transformed

