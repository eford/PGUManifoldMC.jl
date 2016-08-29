if !isdefined(:Distributions) using Distributions end
if !isdefined(:Lora) using Lora end
if !isdefined(:PGUManifoldMC) using PGUManifoldMC end

DATADIR = "../../data"
SUBDATADIR = "mala"

nchains = 10
nmcmc = 11000
nburnin = 1000

dataset = readdlm(joinpath(DATADIR, "example2.csv"), ',', header=false);

obs_times = dataset[:,1]
obs_rv = dataset[:,2]
sigma_obs = dataset[:,3]

ndata = length(obs_times)
# npars = RvModelKeplerian.num_param_per_planet + 1

include("rv_model.jl")    # provides plogtarget
#if !isdefined(:RvModelKeplerian)  using RvModelKeplerian end  # Why doesn't this work?
using RvModelKeplerian 

set_times(obs_times);     # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);


include("utils_ex.jl")
param_true = make_param_true_ex2()
param_perturb_scale = make_param_perturb_scale(param_true)
param_init = 0
param_init = param_true.+0.001*param_perturb_scale.*randn(length(param_true))
println("param_init= ",param_init)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  autodiff=:forward
)

#= Uncommenting this gives an error
p = BasicContMuvParameter(
    :p,
    logtarget=plogtarget,
    #nkeys=1,
    autodiff=:reverse,
    init=Any[(:p, param_init)]
    #order=1
  )
=#

model = likelihood_model(p, false)

sampler = MALA(0.02)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

v0 = Dict(:p=>param_init)

job = BasicMCJob(
  model,
  sampler,
  mcrange,
  v0,
  # tuner=VanillaMCTuner(verbose=true),
  tuner=AcceptanceRateMCTuner(0.574, verbose=true),
  outopts=outopts
)

tic()
run(job)
runtime = toc()

chain = output(job)

acceptance(chain)

mean(chain)
