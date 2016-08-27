using Distributions
using Lora
using PGUManifoldMC

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
using RvModelKeplerian
set_times(obs_times);     # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  #  nkeys=1,
    autodiff=:forward,
  #  init=Any[(:p, v0[:p]), (:v, Any[v0[:p]])]
  #order=1
)

model = likelihood_model(p, false)

sampler = MALA(0.02)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
i = 1

include("utils_ex.jl")
param_true = make_param_true_ex2()
param_perturb_scale = make_param_perturb_scale(param_true)
param_init = 0

while i <= nchains
  param_init = param_true.+0.001*param_perturb_scale.*randn(length(param_true))
  v0 = Dict(:p=>param_init) 
  println("param_init= ",param_init)
  job = BasicMCJob(
    model,
    sampler,
    mcrange,
    v0,
    tuner=VanillaMCTuner(verbose=true),
    outopts=outopts
  )

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  if 0.5 < ratio < 0.65
    writedlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(DATADIR, SUBDATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.step

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  end
end

writedlm(joinpath(DATADIR, SUBDATADIR, "times.csv"), times, ',')
writedlm(joinpath(DATADIR, SUBDATADIR, "stepsizes.csv"), stepsizes, ',')

