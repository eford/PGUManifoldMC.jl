using Distributions
using Klara
using PGUManifoldMC

include("rv_model.jl")

using RvModelKeplerian

DATADIR = "../../data"
SUBDATADIR = "smmala"

dataset = readdlm(joinpath(DATADIR, "example2.csv"), ',', header=false); # read observational data
obs_times = dataset[:,1]
obs_rv = dataset[:,2]
sigma_obs = dataset[:,3]
set_times(obs_times);     # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);

include("utils_ex.jl")
param_true = make_param_true_ex2()
param_perturb_scale = make_param_perturb_scale(param_true)
param_init = 0
param_init = param_true.+0.01*param_perturb_scale.*randn(length(param_true))
println("param_init= ",param_init)

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

# target_accept_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 0.9]
target_accept_rates = [0.7]
#target_accept_rates = [0.25, 0.5, 0.75, 0.9]  # 0.574 for MALA
#target_accept_rates = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
#target_accept_rates = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
#target_accept_rates = [0.574]
jobs = Array(Any,length(target_accept_rates))
smmala_chains = Array(Any,length(target_accept_rates))
smmala_times = Array(Float64, length(target_accept_rates))
smmala_stepsizes = Array(Float64, length(target_accept_rates))
smmala_acceptance = Array(Float64, length(target_accept_rates))
smmala_esses = Array(Float64, length(target_accept_rates))
smmala_iacts = Array(Float64, length(target_accept_rates))

for i in 1:length(target_accept_rates)
  target_accept_rate = target_accept_rates[i]
  println("# i= ", i, ": ",target_accept_rate)
  println("# p= ",param_init)
  sampler = SMMALA(0.2, H -> softabs(H, 1000.))
  mctuner = AcceptanceRateMCTuner(target_accept_rate, score=x -> logistic_rate_score(x, 0.3), verbose=true)
  #mctuner = VanillaMCTuner(verbose=true)
  v0 = Dict(:p=>param_init)
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
  println("# iact[",i,"] = ",iact(chain))

  jobs[i] = job
  smmala_times[i] = runtime
  smmala_stepsizes[i] = job.sstate.tune.step
  smmala_acceptance[i] = ratio
  smmala_esses[i] = minimum(ess(chain))
  smmala_iacts[i] = maximum(iact(chain))
  smmala_chains[i] = chain
end
