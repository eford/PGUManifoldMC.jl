using Distributions
using Lora
using PGUManifoldMC

function C(n::Int, c::Float64)
  X = eye(n)
  [(j <= n-i) ? X[i+j, i] = X[i, i+j] = c^j : nothing for i = 1:(n-1), j = 1:(n-1)]
  X
end

n = 15
μ = zeros(n)
Σ = C(n, 0.5)
ν = 30.

Σt = 28*Σ/30
Σtinv = inv(Σt)

function plogtarget(p::Vector, v::Vector)
  hdf = 0.5*ν
  hdim = 0.5*n
  shdfhdim = hdf+hdim
  v = lgamma(shdfhdim)-lgamma(hdf)-hdim*log(ν)-hdim*log(pi)-0.5*logdet(Σt)
  z = p-μ
  v-shdfhdim*log1p(dot(z, Σtinv*z)/ν)
end

v0 = Dict(:p=>[-4., 2., 3., 1., 2.4, -4., 2., 3., 1., 2.4, -4., 2., 3., 1., 2.4])

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:reverse,
  init=Any[(:p, v0[:p]), (:v, Any[v0[:p]])],
  order=1
)

model = likelihood_model([p], isindexed=false)

# Simulation 01

sampler = MALA(1.)

mcrange = BasicMCRange(nsteps=11000, burnin=1000)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(
  model,
  sampler,
  mcrange,
  v0,
  tuner=AcceptanceRateMCTuner(0.6, score=x -> logistic_rate_score(x, 3.), verbose=false),
  outopts=outopts
)

tic()
run(job)
runtime = toc()

chain = output(job)

ppostmean = mean(chain)

ess(chain, vtype=:bm)

ess(chain, vtype=:bm)/runtime

acceptance(chain)
