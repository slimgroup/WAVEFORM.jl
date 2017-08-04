# A simple 2D transmission full waveform inversion experiment
# to showcase how this framework operates. Much of the 
# 

using JOLI   # Linear operators
using PyPlot # For plotting
using Waveform

# Turn off annoying JOLI type checks
JOLI.jo_type_mismatch_error_set(false)
JOLI.jo_type_mismatch_warn_set(false)

# number of model points in each direction (z,x)
n = 101*[1;1];

# grid spacing in metres
d = 10.0*[1;1];

# location of the coordinate axis origin
o = 0.0*[1;1];

# Frequencies
freqs = [5.0;10.0;15.0];

# Source wavelet time shift
t0 = 0.0;

# Peak frequency of Ricker Wavelet
f0 = 10.0;

# Units of velocity parameter vector 
unit = "m/s";

# Background velocity (m/s)
vel_background = 2000;

# Model domain range
L = n.*d;

# Source grid definition
xsrc = 0.0:100.0:L[2];
ysrc = [0.0];
zsrc = [10.0];

# Receiver grid definition
xrec = 0:10.0:L[2];
yrec = [0.0];
zrec = [950.0];

# Model type containing the domain geometry
model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec);
nsrc = length(model.xsrc); nfreq = length(model.freq);

comp_n = n;
comp_d = d;
comp_o = o;
npml = 20*ones(Int64,2,2);

# Stencil used to discretize the PDE
pde_scheme = helm2d_chen9p;

# If true, 
cut_pml = true;

implicit_matrix = false;

# Objective function
misfit = least_squares;

# Binary mask to indicate which sources/frequencies to use (default: all of them)
srcfreqmask = trues(nsrc,nfreq);

# Use a direct solver for 2D problems
lsopts = LinSolveOpts(solver=:lufact);


# Options for discretizing + solving the Helmholtz PDEs
opts = PDEopts{Int64,Float64}(pde_scheme,comp_n,comp_d,comp_o,cut_pml,implicit_matrix,npml,misfit,srcfreqmask,lsopts);


v0 = vel_background*ones(n...); 
v = copy(v0);
v[div(n[1],3):2*div(n[1],3),div(n[2],3):2*div(n[2],3)] = 1.25*vel_background;
v = vec(v);
v0 = vec(v0);

# True velocity
imshow(v);

Q = eye(nsrc);
D = forw_model(v,Q,model,opts);


# Plot a frequency slice
imshow(real(D[:,1:nsrc]),aspect="auto");

obj! = construct_pde_misfit(v,Q,D,model,opts,batch_mode=false)

using OptimPackNextGen
proj! = (xproj,x)->project_bounds!(x,minimum(v),maximum(v),xproj);
vest = spg(obj!,proj!,v0,3,maxfc=50,verb=true)
imshow(reshape(vest,model.n...),vmin=minimum(v),vmax=maximum(v))
