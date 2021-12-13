#
# Copyright (c) 2021 Stefano Dal Forno.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def bands(k,a1,a2):
	band = np.sqrt(3 + 2*np.cos(np.dot(k,a1)) + 2*np.cos(np.dot(k,a2)) + 2*np.cos(np.dot(k,(a2-a1))))
	return band
	
def opt_mat_elems(k,a1,a2,n,m):
	N = n**2 + n*m + m**2
	elem = ((n-m)*np.cos(np.dot(k,(a2-a1))) - (2*n+m)*np.cos(np.dot(k,a1)) + (n+2*m)*np.cos(np.dot(k,a2)))/2/np.sqrt(N)/bands(k,a1,a2)
	return elem

def writefile(path,xdata,ydata,header=''):
	f = open(path,'w+')
	f.write(header)
	for i,x in enumerate(xdata):
		#y = ' '.join(map(str, ydata[:,i]))
		f.write('%s %s\n'%(x,ydata[i]))
	f.close()

def u_hel(n,m,N,M,R):
	qs = np.arange(-10,10)
	um = ((2*n+m)*M + 2*N*qs)//R
	us = [ i//m for i in um if i%m==0 and (i//m)%2!=0] # even numbers? mystery
	res = us[np.argmin(np.abs(us))]
	return res

text = '''n, m = %(n)s, %(m)s
Diameter = %(diam).2f %(unit)s.
T = %(Tnorm).2f %(unit)s.
C = %(Cnorm).2f %(unit)s.
Linear representation:
 * BZ width (-pi/T, pi/T) = (-%(piT).2f, %(piT).2f) 1/%(unit)s.
 * Sub bands %(NU)s.
Helical Representation:
 * BZ width (-%(NU)spi/T, %(NU)spi/T) = (-%(NUpiT).2f, %(NUpiT).2f) 1/%(unit)s.
 * Sub bands %(M)s.
'''

gamma, en_unit = 2.9, 'eV'

# graphene lattice vectors
#a0, unit = 2.461, 'Angstrom'
a0, unit = 0.2461, 'nm'
#a0, unit = 4.6511, 'bohr'
a1 = a0*np.array([np.sqrt(3)/2,1/2])
a2 = a0*np.array([np.sqrt(3)/2,-1/2])
b0 = 4*np.pi/np.sqrt(3)/a0
b1 = b0*np.array([1/2,np.sqrt(3)/2])
b2 = b0*np.array([1/2,-np.sqrt(3)/2])

# carbon nanotube parameters
n, m = 4, 2
R = np.gcd(2*m+n,2*n+m)
M = np.gcd(m,n)
N = n**2 + n*m + m**2
NU = 2*N//R
u = u_hel(n,m,N,M,R)
linmu = range(-NU//2,NU//2)
helmu = range(0,M)

# CNT lattice vectors
C     = n*a1 + m*a2
T     = (2*m+n)/R*a1 - (2*n+m)/R*a2
Cnorm = np.linalg.norm(C)
Tnorm = np.linalg.norm(T)
diam  = Cnorm/np.pi
Kpar  = R/2/N * (m*b1 - n*b2)
Kort  = ((2*n+m)*b1 + (2*m+n)*b2)/2/N
BZlin = np.linalg.norm(Kpar)

Khel1 = M*Kort - u*Kpar
Khel2 = NU/M*Kpar
BZhel = np.linalg.norm(Khel2)

# dictionary with all parameters
dic={'n':n,'m':m,'unit':unit,'diam':diam,'Tnorm':Tnorm,'Cnorm':Cnorm,'piT':np.pi/Tnorm,'NUpiT':NU*np.pi/Tnorm,'NU':NU,'M':M}

# position of carbon atoms
A = a0*np.array([np.sqrt(3)/3,0])
B = a0*np.array([2*np.sqrt(3)/3,0])

# lattice points for drawing
R1 = A
R2 = B - a2
R3 = A - a2
R4 = B - a1 - a2

# lattice points for drawing
K1 = b0*np.array([1/2,np.sqrt(3)/6])  # K1 and K2 are the two inequivalent K point in the BZ
K2 = b0*np.array([1/2,-np.sqrt(3)/6])
K3 = K2 - b2
K4 = K1 - b1

# kgrid from -1/2, 1/2, for both lin and hel BZs
kgrid = 1000
Kz = np.linspace(-0.5,0.5,kgrid)

# settings for drawings
gs = gridspec.GridSpec(2, 3)
fig = plt.figure()
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 1])
ax4 = plt.subplot(gs[0, 2])
ax5 = plt.subplot(gs[1, 2])
plt.tight_layout(pad=1,h_pad=None, w_pad=None,rect=(0,0,1,0.95))

fig.suptitle('CNT (%i,%i)'%(n,m), fontsize=16)
fig.text(0.05,0.97,text%dic, horizontalalignment='left', verticalalignment='top')

minx = min(0,C[0],T[0],C[0]+T[0]) - 2*a0
maxx = max(0,C[0],T[0],C[0]+T[0]) + 2*a0
miny = min(0,C[1],T[1],C[1]+T[1]) - 2*a0
maxy = max(0,C[1],T[1],C[1]+T[1]) + 2*a0
#mind = min(minx,miny)
#maxd = max(maxx,maxy)
ax1.set_title('Direct space')
ax1.set_xlim(minx + a0,maxx - a0)
ax1.set_ylim(miny + a0,maxy - a0)
ax1.set_xlabel(unit)
ax1.set_ylabel(unit)
ax1.set_aspect('equal')
minkxe = min(0,Kpar[0],-Kort[0]*NU/2,Kpar[0]+Kort[0]*NU/2) - 4*np.pi/a0
maxkxe = max(0,Kpar[0],Kort[0]*NU/2,Kpar[0]+Kort[0]*NU/2) + 4*np.pi/a0
minkye = min(0,Kpar[1],-Kort[1]*NU/2,Kpar[1]+Kort[1]*NU/2) - 4*np.pi/a0
maxkye = max(0,Kpar[1],Kort[1]*NU/2,Kpar[1]+Kort[1]*NU/2) + 4*np.pi/a0
minke  = min(minkxe,minkye)
maxke  = max(maxkxe,maxkye)
ax2.set_title('Reciprocal space (linear)')
ax2.set_xlim(minke + 3*np.pi/a0,maxke - 3*np.pi/a0)
ax2.set_ylim(minke + 3*np.pi/a0,maxke - 3*np.pi/a0)
ax2.set_xlabel('1/'+unit)
ax2.set_ylabel('1/'+unit)
ax2.set_aspect('equal')
kor = -b1-b2-Khel2/2
kor1= Khel1-b1-b2-Khel2/2
kor2= Khel2-b1-b2-Khel2/2
minkxf = min(0,Khel2[0]/2,-Khel2[0]/2,kor[0],-kor[0]) - 4*np.pi/a0
maxkxf = max(0,Khel2[0]/2,-Khel2[0]/2,kor[0],-kor[0]) + 4*np.pi/a0
minkyf = min(0,Khel2[1]/2,-Khel2[1]/2,-Khel2[1]/2+Khel1[1]) - 4*np.pi/a0
maxkyf = max(0,Khel2[1]/2,-Khel2[1]/2,Khel2[1]/2) + 4*np.pi/a0
minkf  = min(minkxf,minkyf)
maxkf  = max(maxkxf,maxkyf)
ax3.set_title('Reciprocal space (helical)')
ax3.set_xlim(minkf + 3*np.pi/a0,maxkf - 3*np.pi/a0)
ax3.set_ylim(minkf + 3*np.pi/a0,maxkf - 3*np.pi/a0)
ax3.set_xlabel('1/'+unit)
ax3.set_ylabel('1/'+unit)
ax3.set_aspect('equal')
ax4.set_title('Band structure (linear)')
#ax4.set_xlim()
ax4.set_xlabel('1/'+unit)
ax4.set_ylabel('Energy (%s)'%(en_unit))
ax5.set_title('Band structure (helical)')
#ax5.set_xlim()
ax5.set_xlabel('1/'+unit)
ax5.set_ylabel('Energy (%s)'%(en_unit))

# plot basis vectors of CNT
ax1.annotate('',xytext=(0,0),xy=C,arrowprops=dict(color='b',width=1,headwidth=5,headlength=5),annotation_clip=False).arrow_patch.set_clip_box(ax1.bbox)
ax1.annotate('',xytext=(0,0),xy=T,arrowprops=dict(color='b',width=1,headwidth=5,headlength=5),annotation_clip=False).arrow_patch.set_clip_box(ax1.bbox)
ax2.annotate('',xytext=(0,0),xy=Kpar,arrowprops=dict(color='b',width=1,headwidth=5,headlength=5),annotation_clip=False).arrow_patch.set_clip_box(ax2.bbox)
ax2.annotate('',xytext=(0,0),xy=Kort,arrowprops=dict(color='b',width=1,headwidth=5,headlength=5),annotation_clip=False).arrow_patch.set_clip_box(ax2.bbox)
ax3.annotate('',xytext=kor,xy=kor1,arrowprops=dict(color='b',width=1,headwidth=5,headlength=5),annotation_clip=False).arrow_patch.set_clip_box(ax3.bbox)
ax3.annotate('',xytext=kor,xy=kor2,arrowprops=dict(color='b',width=1,headwidth=5,headlength=5),annotation_clip=False).arrow_patch.set_clip_box(ax3.bbox)

dirlat = []
reclate = []
reclatf = []

for i in range(-30,30):
	for j in range(-30,30):
		Rr = i*a1 + j*a2
		G = i*b1 + j*b2
		if (minx<Rr[0]<maxx and miny<Rr[1]<maxy):
			dirlat.append(Rr)
		if (minke<G[0]<maxke and minke<G[1]<maxke):
			reclate.append(G)
		if (minkf<G[0]<maxkf and minkf<G[1]<maxkf):
			reclatf.append(G)

dirlat = np.array(dirlat)
reclate = np.array(reclate)
reclatf = np.array(reclatf)
ax1.scatter(dirlat[:,0],dirlat[:,1],s=1,c='r')
ax2.scatter(reclate[:,0],reclate[:,1],s=1,c='r')
ax3.scatter(reclatf[:,0],reclatf[:,1],s=1,c='r')

R1s = dirlat + R1
R2s = dirlat + R2
R3s = dirlat + R3
R4s = dirlat + R4

for p1, p2 in zip(R1s, R2s):
	ax1.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
for p1, p2 in zip(R2s, R3s):
	ax1.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
for p1, p2 in zip(R3s, R4s):
	ax1.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')

K1s = reclate + K1
K2s = reclate + K2
K3s = reclate + K3
K4s = reclate + K4

for p1, p2 in zip(K1s, K2s):
	ax2.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
for p1, p2 in zip(K1s, K3s):
	ax2.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
for p1, p2 in zip(K2s, K4s):
	ax2.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')

K1s = reclatf + K1
K2s = reclatf + K2
K3s = reclatf + K3
K4s = reclatf + K4

for p1, p2 in zip(K1s, K2s):
	ax3.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
for p1, p2 in zip(K1s, K3s):
	ax3.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
for p1, p2 in zip(K2s, K4s):
	ax3.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')

linear_k = []
linear_band = []
for i in linmu:
	K = np.outer(Kz,Kpar) + i*Kort
	ax2.plot(K[:,0],K[:,1],'r')
	ax2.annotate(i,xy=(i*Kort[0],i*Kort[1]))
	band = gamma*bands(K,a1,a2)
	ax4.plot(Kz*BZlin,band,'r')
	ax4.plot(Kz*BZlin,-band,'r')
	linear_k += list(Kz*BZlin)
	linear_band += list(band)
	ax4.annotate(i,xy=(0,band[kgrid//2]))
	
writefile('./el_band_linear.dat',linear_k,linear_band)
# save to file name

for i in helmu:
	K = np.outer(Kz,Khel2) + i*Khel1/M
	ax3.plot(K[:,0],K[:,1],'r')
	#ax3.annotate(i,xy=(i*Kort[0],i*Kort[1]))
	band = gamma*bands(K,a1,a2)
	mat = np.sqrt(4.375)*opt_mat_elems(K,a1,a2,n,m)
	ax5.plot(Kz*BZhel,band,'r')
	ax5.plot(Kz*BZhel,-band,'r')
	#ax5.plot(Kz*BZhel,mat**2,'k--')
	#ax5.annotate(i,xy=(0,band[kgrid//2]))

plt.show()
