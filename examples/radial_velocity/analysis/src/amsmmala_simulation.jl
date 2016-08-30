using Distributions
using Lora
using PGUManifoldMC

DATADIR = "../../data"
SUBDATADIR = "amsmmala"

nmcmc = 11000
nburnin = 1000

dataset = readdlm(joinpath(DATADIR, "example2.csv"), ',', header=false);

obs_times = dataset[:,1]
obs_rv = dataset[:,2]
sigma_obs = dataset[:,3]

ndata = length(obs_times)

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
param_init = param_true.+0.01*param_perturb_scale.*randn(length(param_true))
println("param_init= ",param_init)

sampler = AMSMMALA(
  0.06,
  update=(sstate, pstate, i, tot) -> mod_update!(sstate, pstate, i, tot, 7),
  transform=H -> softabs(H, 1000.)
)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

v0 = Dict(:p=>param_init)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  #  nkeys=1,
  autodiff=:forward,
  #autodiff=:reverse,
  #init=Any[(:p, v0[:p]), (:v, Any[v0[:p]])],
  order=2
)

model = likelihood_model(p, false)
#model = likelihood_model([p], isindexed=false)

times = Array(Float64, length(target_accept_rates))
stepsizes = Array(Float64, length(target_accept_rates))
amsmmala_esses = Array(Float64, length(target_accept_rates))

for i in 1:length(target_accept_rates)
  target_accept_rate = target_accept_rate[i]
  mctuner = PSMMALAMCTuner(
    VanillaMCTuner(verbose=true),
    VanillaMCTuner(verbose=true),
    AcceptanceRateMCTuner(target_accept_rate, verbose=true)
  )

  job = BasicMCJob(
    model,
    sampler,
    mcrange,
    v0,
    tuner=mctuner,
    outopts=outopts
  )

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  println("# mean[",i,"] = ",mean(chain))
  println("# ess[",i,"] = ",ess(chain))

  times[i] = runtime
  stepsizes[i] = job.sstate.tune.step
  alsmmala_esses[i] = minimum(ess(chain))

end

