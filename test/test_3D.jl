using JOLI
using Revise
using Waveform



n = 100*[1;1;1];
d = 10.0*[1;1;1];
o = 0.0*[1;1;1];
v0 = 2000;
freq = 4.0;
λ = v0/freq;

t0 = 0.0;
f0 = 10.0;
unit = "m/s";
xsrc = [(n.*d)[1]/2];
ysrc = [(n.*d)[2]/2];
zsrc = [100.0];
xrec = linspace(0.0,(n.*d)[1]::Float64,n[1]);
yrec = linspace(0.0,(n.*d)[2]::Float64,n[2]);
zrec = [100.0];
freqs = [freq];
nsrc = length(xsrc)*length(ysrc)*length(zsrc);
nfreq = length(freqs)
model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec);

comp_n = n;
comp_d = d;
comp_o = o;
npml = convert(Int,λ/minimum(comp_d))*ones(Int64,(2,3));
npml = 30*ones(Int64,(2,3));
scheme = Waveform.helm3d_operto27;
cut_pml = true;
implicit_matrix = true;
srcfreqmask = trues(nsrc,nfreq);
misfit = Waveform.least_squares;
lsopts = LinSolveOpts(solver=:fgmres,maxinnerit=5,tol=1e-6);
lsopts.precond = :mlgmres;
opts = PDEopts{Int64,Float64}(scheme,comp_n,comp_d,comp_o,cut_pml,implicit_matrix,npml,misfit,srcfreqmask,lsopts);

nλ = maximum(n.*d/λ);
ppλ = λ/maximum(d);
v = v0*ones(Float64,n...);
v = vec(v);
opts.implicit_matrix = true;

(H,comp_grid,T,DT_adj,P) = helmholtz_system(v,model,freq,opts);

# ensure linear system is actually solved
nt = comp_grid.comp_n
q = zeros(eltype(H),tuple(nt...));
q[div.(nt,2)...] = 1.0
u = H\vec(q)
@test norm(H*u-vec(q)) < 1e-6*norm(q)
ut = H'\vec(q);
@test norm(H'*ut-vec(q)) < 1e-6*norm(q)

# test computed green's function ≈ analytical green's function in the non-pml region
(xt,yt,zt) = odn_to_grid(comp_grid.comp_o,comp_grid.comp_d,comp_grid.comp_n)
idx = indmax(abs(q))
(ix,iy,iz) = ind2sub(q,idx)
ω = (2*pi*freq)/v0;
R = [sqrt((x-xt[ix])^2 + (y-yt[ix])^2 + (z-zt[ix])^2) for x in xt, y in yt, z in zt];
G = [prod(model.d)*exp(im*ω*r)/(4*π*r) for r in R];
r = reshape(u,size(G))-G;
r[ix,iy,iz] = 0;
@test norm(comp_grid.comp_to_phys_grid*vec(r))/norm(comp_grid.comp_to_phys_grid*vec(u)) < 0.01

# test derivative behaviour
n = 50*[1;1;1];

xsrc = [(n.*d)[1]/2];
ysrc = [(n.*d)[2]/2];
zsrc = [100.0];
xrec = linspace(0.0,(n.*d)[1]::Float64,n[1]);
yrec = linspace(0.0,(n.*d)[2]::Float64,n[2]);
zrec = [100.0];
freqs = [freq];
nsrc = length(xsrc)*length(ysrc)*length(zsrc);
nfreq = length(freqs)
model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec);

comp_n = n;
comp_d = d;
comp_o = o;
npml = convert(Int,λ/minimum(comp_d))*ones(Int64,(2,3));
npml = 10*ones(Int64,(2,3));
scheme = Waveform.helm3d_operto27;
cut_pml = true;
implicit_matrix = true;
srcfreqmask = trues(nsrc,nfreq);
misfit = Waveform.least_squares;
lsopts = LinSolveOpts(solver=:fgmres,maxinnerit=5,tol=1e-10);
lsopts.precond = :mlgmres;
opts = PDEopts{Int64,Float64}(scheme,comp_n,comp_d,comp_o,cut_pml,implicit_matrix,npml,misfit,srcfreqmask,lsopts);

nλ = maximum(n.*d/λ);
ppλ = λ/maximum(d);
v = v0*ones(Float64,n...);
v = vec(v);
opts.implicit_matrix = true;

(H,comp_grid,T,DT_adj,P) = helmholtz_system(v,model,freq,opts);
nt = comp_grid.comp_n
q = zeros(eltype(H),tuple(nt...));
q[div.(nt,2)...] = 1.0;
q = vec(q);
u0 = H\q;
δv = randn(size(v));
δvx = comp_grid.phys_to_comp_grid*δv;
δu = H\(-T(u)*δvx);
step = 10.^(-5.0:1.0);
err0 = zeros(length(step))
err1 = zeros(length(step))
for i=1:length(step)
    h = step[i]
    H1 = helmholtz_system(v+δv*h,model,freq,opts)[1];
    u1 = H1\q;
    err0[i] = norm(u1-u0);
    err1[i] = norm(u1-u0-h*δu)
end

@test abs(median(diff(log10.(err0)))-1) < 0.2
@test abs(median(diff(log10.(err1)))-2) < 0.2)

