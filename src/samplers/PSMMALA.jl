### Abstract PSMMALA state

abstract PSMMALAState <: MCSamplerState

### PSMMALA state subtypes

## MuvPSMMALAState holds the internal state ("local variables") of the PSMMALA sampler for multivariate parameters

type MuvPSMMALAState <: PSMMALAState
  pstate::ParameterState{Continuous, Multivariate}
  tune::MCTunerState
  sqrttunestep::Real
  ratio::Real
  μ::RealVector
  runningmean::RealVector
  newinvtensor::RealMatrix
  oldinvtensor::RealMatrix
  cholinvtensor::RealLowerTriangular
  newfirstterm::RealVector
  oldfirstterm::RealVector
  presentupdatetensor::Bool
  pastupdatetensor::Bool
  count::Integer
  updatetensorcount::Integer

  function MuvPSMMALAState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    sqrttunestep::Real,
    ratio::Real,
    μ::RealVector,
    runningmean::RealVector,
    newinvtensor::RealMatrix,
    oldinvtensor::RealMatrix,
    cholinvtensor::RealLowerTriangular,
    newfirstterm::RealVector,
    oldfirstterm::RealVector,
    presentupdatetensor::Bool,
    pastupdatetensor::Bool,
    count::Integer,
    updatetensorcount::Integer
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end
    new(
      pstate,
      tune,
      sqrttunestep,
      ratio,
      μ,
      runningmean,
      newinvtensor,
      oldinvtensor,
      cholinvtensor,
      newfirstterm,
      oldfirstterm,
      presentupdatetensor,
      pastupdatetensor,
      count,
      updatetensorcount
    )
  end
end

MuvPSMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=PSMMALAMCTune()) =
  MuvPSMMALAState(
  pstate,
  tune,
  NaN,
  NaN,
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size+2),
  Array(eltype(pstate), pstate.size, pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size)),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size),
  false,
  false,
  0,
  0
)

exp_decay(i::Integer, tot::Integer, a::Real=10., b::Real=0.) = (1-b)*exp(-a*i/tot)+b

pow_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(a^(i/tot))+b

lin_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(1/(1+a*i))+b

quad_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(1/(1+a*abs2(i)))+b

mala_only_update!(sstate::MuvPSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = false

smmala_only_update!(sstate::MuvPSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = true

mod_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  n::Integer=10
) =
  sstate.presentupdatetensor = mod(i, n) == 0 ? true : false

cos_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.2,
  c::Real=0.2
) =
  sstate.presentupdatetensor = rand(Bernoulli(b*cos(a*i*pi/tot)+c))

rand_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  p::Real=0.5
) =
  sstate.presentupdatetensor = rand(Bernoulli(p))

rand_exp_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(exp_decay(i, tot, a, b)))

rand_pow_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-3,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(pow_decay(i, tot, a, b)))

rand_lin_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-2,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(lin_decay(i, tot, a, b)))

rand_quad_decay_update!(
  sstate::MuvPSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-5,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(quad_decay(i, tot, a, b)))

### Metropolis-adjusted Langevin Algorithm (PSMMALA)

immutable PSMMALA <: LMCSampler
  driftstep::Real
  update!::Function
  transform::Union{Function, Void}

  function PSMMALA(driftstep::Real, update!::Function, transform::Union{Function, Void})
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep, update!, transform)
  end
end

PSMMALA(driftstep::Real=1.; update::Function=rand_update!, transform::Union{Function, Void}=nothing) =
  PSMMALA(driftstep, identitymala, update, transform, initupdatetensor)

### Initialize PSMMALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::PSMMALA
)
  parameter.uptotensorlogtarget!(pstate)
  if sampler.transform != nothing
    pstate.tensorlogtarget = sampler.transform(pstate.tensorlogtarget)
  end
  pstate.diagnosticvalues[1] = false
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"
end

## Initialize PSMMALA state

tuner_state(sampler::PSMMALA, tuner::PSMMALAMCTuner) =
  PSMMALAMCTune(
    BasicMCTune(NaN, 0, 0, tuner.smmalatuner.period),
    BasicMCTune(NaN, 0, 0, tuner.malatuner.period),
    BasicMCTune(sampler.driftstep, 0, 0, tuner.totaltuner.period)
  )

function sampler_state(
  sampler::PSMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvPSMMALAState(generate_empty(pstate), tuner_state(sampler, tuner))
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.runningmean[1] = copy(pstate.value)
  sstate.runningmean[2] = copy(pstate.value)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
  sstate
end

### Reset PSMMALA sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::PSMMALA
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
  pstate.diagnosticvalues[1] = false
end

## Reset sampler state

function reset!(
  sstate::PSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::PSMMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.runningmean[1] = copy(pstate.value)
  sstate.runningmean[2] = copy(pstate.value)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
end

Base.show(io::IO, sampler::PSMMALA) = print(io, "PSMMALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::PSMMALA) = show(io, sampler)
