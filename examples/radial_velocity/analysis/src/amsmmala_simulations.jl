using Distributions
using Lora
using PGUManifoldMC

DATADIR = "../../data"
SUBDATADIR = "amsmmala"

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

sampler = AMSMMALA(
  0.06,
  update=(sstate, pstate, i, tot) -> mod_update!(sstate, pstate, i, tot, 7),
  transform=H -> softabs(H, 1000.)
)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

mctuner = PSMMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  AcceptanceRateMCTuner(0.15, verbose=false)
)

sampler = SMMALA(0.25, H -> softabs(H, 1000.))

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])


times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
nupdates = Array(Int64, nchains)
i = 1

while i <= nchains
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

  if 0.145 < ratio < 0.165
    writedlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(DATADIR, SUBDATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.totaltune.step
    nupdates[i] = job.sstate.updatetensorcount

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  else
    println("Iteration ", i, " of ", nchains, " ignored with acceptance ratio ", ratio)
  end
end

writedlm(joinpath(DATADIR, SUBDATADIR, "times.csv"), times, ',')
writedlm(joinpath(DATADIR, SUBDATADIR, "stepsizes.csv"), stepsizes, ',')
