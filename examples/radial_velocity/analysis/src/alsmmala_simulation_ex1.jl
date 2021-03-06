using Distributions
using Klara
using PGUManifoldMC

include("rv_model.jl")

using RvModelKeplerian

DATADIR = "../../data"
SUBDATADIR = "alsmmala"

dataset = readdlm(joinpath(DATADIR, "example1.csv"), ',', header=false); # read observational data
obs_times = dataset[:,1]
obs_rv = dataset[:,2]
sigma_obs = dataset[:,3]
set_times(obs_times);     # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);

include("utils_ex.jl")
param_true = make_param_true_ex1()
param_perturb_scale = make_param_perturb_scale(param_true)
param_init = 0
param_init = param_true.+0.01*param_perturb_scale.*randn(length(param_true))
println("param_init= ",param_init)

sampler = ALSMMALA(
  # 0.05,
  0.03,
  identitymala=false,
  #update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10., 0.15),
  # update=(sstate, pstate, i, tot) -> mod_update!(sstate, pstate, i, tot, 50),
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 7.),
  transform=H -> softabs(H, 1000.),
  initupdatetensor=(true, false)
)

nmcmc = 50000
nburnin = 10000
mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

v0 = Dict(:p=>param_init)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  diffopts=DiffOptions(mode=:forward, order=2)
)

model = likelihood_model(p, false)
#model = likelihood_model([p], isindexed=false)

#target_accept_rates = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
#target_accept_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #, 0.45 0.5] #, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
target_accept_rates = [0.7]
alsmmala_times = Array(Float64, length(target_accept_rates))
alsmmala_stepsizes = Array(Float64, length(target_accept_rates))
alsmmala_acceptance = Array(Float64, length(target_accept_rates))
alsmmala_esses = Array(Float64, length(target_accept_rates))
alsmmala_iacts = Array(Float64, length(target_accept_rates))
alsmmala_chains = Array(Any, length(target_accept_rates))

for i in 1:length(target_accept_rates)
  target_accept_rate = target_accept_rates[i]
  println("# i= ", i, ": ",target_accept_rate)
  mctuner = PSMMALAMCTuner(
    VanillaMCTuner(verbose=true),
    VanillaMCTuner(verbose=true),
    VanillaMCTuner(verbose=true)
    # AcceptanceRateMCTuner(target_accept_rate, verbose=true)
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
  println("# std[",i,"] =",vec(std(chain.value,2)))
  println("# ess[",i,"] = ",ess(chain))

  alsmmala_times[i] = runtime
  alsmmala_stepsizes[i] = job.sstate.tune.totaltune.step
  alsmmala_acceptance[i] = ratio
  alsmmala_esses[i] = minimum(ess(chain))
  alsmmala_iacts[i] = maximum(iact(chain))
  alsmmala_chains[i] = chain
end
