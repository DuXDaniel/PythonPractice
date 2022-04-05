import numpy as np
import scipy as sp
import math as math
import statsmodels as sm
import sklearn as skl
import matplotlib as mpl
import matplotlib.pyplot as plt

a = np.arange(15).reshape(3,5);
print(a)

print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))

b = np.array([6,7,8])
print(b)
print(type(b))

a = np.array([2,3,4])
print(a)
print(a.dtype)
b = np.array([1.2, 3.5, 5.1])
print(b.dtype)

b = np.array([(1.5,2,3),(4,5,6)])
print(b)
print(b.dtype)

c = np.array([[1,2],[3,4]], dtype = complex)
print(c)
print(c[0,1])

A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])

print(A*B)
print(A@B)
print(A.dot(B))

rg = np.random.default_rng(1)
a = np.ones((2,3), dtype=int)
b = rg.random((2,3))
print(a)
a *= 3
print(a)
print(b)
b+=a
print(b)
try:
    a+=b
except Exception as e:
    print("type miscasting for a+=b float into int")
print(a)

a = rg.random((2,3))
print(a.sum())
print(a.min())
print(a.max())

b = np.arange(12).reshape(3,4)
print(b)
print(b.sum(axis=0))
print(b.min(axis=1))
print(b.cumsum(axis=1))

B = np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))
C = np.array([2.,-1.,4.])
print(np.add(B,C))

a = np.arange(10)**3
a[:6:2] = 1000
print(a)
print(a[::-1])

def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype = int)
print(b)
print(b[2,3])
print(b[0:5,1])
print(b[:,1])
print(b[1:3,:])

print(b[-1])
print(b[-1,:])

for row in b:
    print(row)

c = np.array([[[0,1,2],[10,12,13]],[[100,101,102],[110,112,113]]])
print(c.shape)
print(c[1,...])
print(c[1,:,:])
print(c[1])
print(c[...,2])
print(c[:,:,2])

for element in b.flat:
    print(element)

a = np.floor(10*rg.random((3,4)))
print(a)
print(a.shape)
print(a.ravel())
print(a.reshape(6,2))
print(a.T)
print(a.T.shape)
print(a.shape)

print(a.resize((2,6)))
print(a)

print(a.reshape(3,-1))

a = np.floor(10*rg.random((2,2)))
print(a)
b = np.floor(10*rg.random((2,2)))
print(b)
print(np.vstack((a,b)))
print(np.hstack((a,b)))

print(np.column_stack((a,b)))
a = np.array([4.,2.])
b = np.array([3.,8.])
print(np.column_stack((a,b)))
print(np.hstack((a,b)))
print(a[:,np.newaxis])
print(np.column_stack((a[:,np.newaxis],b[:,np.newaxis])))
print(np.hstack((a[:,np.newaxis],b[:,np.newaxis])))

print(np.column_stack is np.hstack)
print(np.row_stack is np.vstack)

print(np.r_[1:4,0,4])

a = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
b = a
print(b is a)
def f(x):
    print(id(x))

print(id(a))
print(f(a))

c = a.view()
print(c is a)
print(c.base is a)
print(c.flags.owndata)
c = c.reshape((2,6))
print(a.shape)
c[0,4] = 1234
print(a)

s = a[:,1:3]
s[:] = 10
print(a)

d = a.copy()
print(d is a)
print(d.base is a)
d[0,0] = 9999
print(d)
print(a)

del a

a = np.arange(12)**2
i = np.array([1,1,3,8,5])
print(a[i])
j = np.array([[3,4],[9,7]])
print(a[j])

palette = np.array([[0,0,0],
                    [255,0,0],
                    [0,255,0],
                    [0,0,255],
                    [255,255,255]])
image = np.array([[0,1,2,0],
                  [0,3,4,0]])
print(palette[image])

a = np.arange(12).reshape(3,4)
print(a)
i = np.array([[0,1],
              [1,2]])
j = np.array([[2,1],[3,3]])

print(a[i,j])
print(a[i,2])
print(a[:,j])
l = (i,j)
print(a[l])

s = np.array([i,j])
print(a[tuple(s)])

time = np.linspace(20,145,5)
data = np.sin(np.arange(20)).reshape(5,4)
print(time)
print(data)
ind = data.argmax(axis = 0)
print(ind)
time_max = time[ind]
data_max = data[ind,range(data.shape[1])]
print(time_max)
print(data_max)
print(np.all(data_max == data.max(axis=0)))

a = np.arange(5)
print(a)
a[[1,3,4]] = 0
print(a)

a = np.arange(5)
a[[0,0,2]] = [1,2,3]
print(a)
a = np.arange(5)
a[[0,0,2]] += 1
print(a)

a = np.arange(12).reshape(3,4)
b = a > 4
print(b)
print(a[b])

a[b] = 0
print(a)

def mandelbrot(h,w,maxit=20,r=2):
    x = np.linspace(-2.5,1.5,4*h+1)
    y = np.linspace(-1.5,1.5,3*w+1)
    A,B = np.meshgrid(x,y)
    C = A+B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i
        z[diverge] = r

    return divtime

mpl.interactive(True)
plt.imshow(mandelbrot(400,400))

input("press to continue")

a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])
b2 = np.array([True,False,True,False])
print(a[b1,:])
print(a[b1])
print(a[:,b2])
print(a[b1,b2])

a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)
print(ax)
print(bx)
print(cx)
print(ax.shape, bx.shape, cx.shape)
result = ax + bx * cx
print(result)
print(result[3,2,4])
print(a[3] + b[2]*c[4])

def ufunc_reduce(ufct,*vectors):
    vs = np.ix_(*vectors)
    r = ufct.identity
    for v in vs:
        r = ufct(r,v)
    return r

print(ufunc_reduce(np.add,a,b,c))

a = np.arange(30)
b = a.reshape((2,-1,3))
print(b.shape)
print(b)

x = np.arange(0,10,2)
y = np.arange(5)
m = np.vstack([x,y])
print(m)
xy = np.hstack([x,y])
print(xy)

rg = np.random.default_rng(1)
mu,sigma = 2,0.5
v = rg.normal(mu,sigma,10000)
plt.figure()
plt.hist(v,bins=50,density = True) # mpl version
(n,bins) = np.histogram(v,bins=50,density = True) # numpy version
plt.plot(0.5*(bins[1:]+bins[:-1]),n)

input("press to continue")
