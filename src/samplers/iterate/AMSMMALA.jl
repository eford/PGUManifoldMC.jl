function codegen(::Type{Val{:iterate}}, ::Type{AMSMMALA}, job::BasicMCJob)
  local result::Expr
  burninbody = []
  ambody = []
  smmalapasttensorbody = []
  smmalabody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Multivariate
    error("Only multivariate parameter states allowed in AMSMMALA code generation")
  end

  push!(body, :(_job.sstate.count += 1))

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.malatuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    push!(body, :(_job.sstate.tune.totaltune.proposed += 1))
  end

  push!(
    body,
    :(
      if _job.sstate.count > _job.sampler.t0
        _job.sampler.update!(_job.sstate, _job.pstate, _job.sstate.count, _job.range.nsteps)
      end
    )
  )

  push!(smmalabody, :(_job.sstate.updatetensorcount += 1))

  if job.tuner.smmalatuner.verbose
    push!(smmalabody, :(_job.sstate.tune.smmalatune.proposed += 1))
  end

  push!(smmalapasttensorbody, :(_job.parameter.uptotensorlogtarget!(_job.pstate)))

  if job.sampler.transform != nothing
    push!(smmalapasttensorbody, :(_job.pstate.tensorlogtarget = _job.sampler.transform(_job.pstate.tensorlogtarget)))
  end

  push!(smmalapasttensorbody, :(_job.sstate.oldinvtensor = inv(_job.pstate.tensorlogtarget)))

  push!(smmalapasttensorbody, :(_job.sstate.oldfirstterm = _job.sstate.oldinvtensor*_job.pstate.gradlogtarget))

  push!(smmalapasttensorbody, :(_job.sstate.cholinvtensor = ctranspose(chol(Hermitian(_job.sstate.oldinvtensor)))))

  push!(smmalabody, Expr(:if, :(!_job.sstate.pastupdatetensor), Expr(:block, smmalapasttensorbody...)))

  push!(smmalabody, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sstate.tune.totaltune.step*_job.sstate.oldfirstterm))

  push!(
    smmalabody,
    :(
      _job.sstate.pstate.value =
        _job.sstate.μ+_job.sstate.sqrttunestep*_job.sstate.cholinvtensor*randn(_job.pstate.size)
    )
  )

  push!(smmalabody, :(_job.parameter.uptotensorlogtarget!(_job.sstate.pstate)))

  if job.sampler.transform != nothing
    push!(smmalabody, :(_job.sstate.pstate.tensorlogtarget = _job.sampler.transform(_job.sstate.pstate.tensorlogtarget)))
  end

  push!(smmalabody, :(_job.sstate.newinvtensor = inv(_job.sstate.pstate.tensorlogtarget)))

  push!(smmalabody, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(
    smmalabody,
    :(
      _job.sstate.ratio += (
        0.5*(
          logdet(_job.sstate.tune.totaltune.step*_job.sstate.oldinvtensor)
          +dot(
            _job.sstate.pstate.value-_job.sstate.μ,
            _job.pstate.tensorlogtarget*(_job.sstate.pstate.value-_job.sstate.μ)
          )/_job.sstate.tune.totaltune.step
        )
      )
    )
  )

  push!(smmalabody, :(_job.sstate.newfirstterm = _job.sstate.newinvtensor*_job.sstate.pstate.gradlogtarget))

  push!(
    smmalabody,
    :(_job.sstate.μ = _job.sstate.pstate.value+0.5*_job.sstate.tune.totaltune.step*_job.sstate.newfirstterm)
  )

  push!(smmalabody, :(
      _job.sstate.ratio -= (
        0.5*(
          logdet(_job.sstate.tune.totaltune.step*_job.sstate.newinvtensor)
          +dot(
            _job.pstate.value-_job.sstate.μ,
            _job.sstate.pstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
          )/_job.sstate.tune.totaltune.step
        )
      )
    )
  )

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstate.gradlogtarget)))

  if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
    push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstate.gradloglikelihood)))
  end

  if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
    push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstate.gradlogprior)))
  end

  push!(update, :(_job.pstate.tensorlogtarget = copy(_job.sstate.pstate.tensorlogtarget)))

  if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
    push!(update, :(_job.pstate.tensorloglikelihood = copy(_job.sstate.pstate.tensorloglikelihood)))
  end

  if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
    push!(update, :(_job.pstate.tensorlogprior = copy(_job.sstate.pstate.tensorlogprior)))
  end

  push!(update, :(_job.sstate.oldinvtensor = copy(_job.sstate.newinvtensor)))

  push!(update, :(_job.sstate.cholinvtensor = ctranspose(chol(Hermitian(_job.sstate.newinvtensor)))))

  push!(update, :(_job.sstate.oldfirstterm = copy(_job.sstate.newfirstterm)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  if in(:accept, job.outopts[:diagnostics])
    push!(update, :(_job.pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[1] = false))
  end

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(update, :(_job.sstate.tune.totaltune.accepted += 1))
  end

  if job.tuner.smmalatuner.verbose
    push!(update, :(_job.sstate.tune.smmalatune.accepted += 1))
  end

  push!(
    smmalabody,
    Expr(
      :if,
      :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))),
      Expr(:block, update...),
      Expr(:block, noupdate...)
    )
  )

  if job.tuner.malatuner.verbose
    push!(ambody, :(_job.sstate.tune.malatune.proposed += 1))
  end

  push!(
    ambody,
    :(
      if _job.sstate.pastupdatetensor
        # Once fully migrated to Julia 0.5 or higher, use LinAlg.lowrankupdate instead of chol in order to reduce complexity
        _job.sstate.cholinvtensor = ctranspose(chol(Hermitian(_job.sstate.oldinvtensor)))
      else
        covariance!(
          _job.sstate.newinvtensor,
          _job.sstate.oldinvtensor,
          _job.sstate.count-2,
          _job.pstate.value,
          _job.sstate.lastmean,
          _job.sstate.secondlastmean
        )

        _job.sstate.cholinvtensor = ctranspose(chol(Hermitian(_job.sstate.newinvtensor)))
      end
    )
  )

  push!(
    ambody,
    :(
      _job.sstate.pstate.value =
        _job.pstate.value+_job.sstate.sqrttunestep*_job.sstate.cholinvtensor*randn(_job.pstate.size)
    )
  )

  push!(ambody, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(ambody, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  push!(update, :(_job.sstate.oldinvtensor = copy(_job.sstate.newinvtensor)))

  if in(:accept, job.outopts[:diagnostics])
    push!(update, :(_job.pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[1] = false))
  end

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(update, :(_job.sstate.tune.totaltune.accepted += 1))
  end

  if job.tuner.malatuner.verbose
    push!(update, :(_job.sstate.tune.malatune.accepted += 1))
  end

  push!(
    ambody,
    Expr(
      :if,
      :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))),
      Expr(:block, update...),
      Expr(:block, noupdate...)
    )
  )

  push!(body, Expr(:if, :(_job.sstate.presentupdatetensor), Expr(:block, smmalabody...), Expr(:block, ambody...)))

  push!(body, :(_job.sstate.secondlastmean = copy(_job.sstate.lastmean)))

  push!(body, :(mean!(_job.sstate.lastmean, _job.sstate.count, _job.pstate.value)))

  push!(body, :(_job.sstate.pastupdatetensor = _job.sstate.presentupdatetensor))

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune.totaltune)))
  end

  if job.tuner.smmalatuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.smmalafrequency = _job.sstate.tune.smmalatune.proposed/_job.sstate.tune.totaltune.proposed)
    )
    push!(burninbody, :(rate!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.malatuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.malafrequency = _job.sstate.tune.malatune.proposed/_job.sstate.tune.totaltune.proposed)
    )
    push!(burninbody, :(rate!(_job.sstate.tune.malatune)))
  end

  if job.tuner.totaltuner.verbose || job.tuner.smmalatuner.verbose || job.tuner.malatuner.verbose
    fmt_tot_iter = format_iteration(ndigits(job.range.burnin))
    fmt_burnin_iter = format_iteration(ndigits(job.tuner.totaltuner.period))
    fmt_perc = format_percentage()

    push!(burninbody, :(println(
      "Burnin iteration ",
      $(fmt_tot_iter)(_job.sstate.count),
      " out of ",
      _job.range.burnin,
      "..."
    )))

    if job.tuner.totaltuner.verbose
      push!(burninbody, :(println(
        "  Total : ",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.totaltune.rate),
        "%) acceptance rate"
      )))
    end

    if job.tuner.smmalatuner.verbose
      push!(burninbody, :(println(
        "  SMMALA: ",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.smmalatune.rate),
        "%) acceptance rate, ",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.proposed),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.smmalafrequency),
        "%) running frequency"
      )))
    end

    if job.tuner.malatuner.verbose
      push!(burninbody, :(println(
        "  MALA  : ",
        $(fmt_burnin_iter)(_job.sstate.tune.malatune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.malatune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.malatune.rate),
        "%) acceptance rate, ",
        $(fmt_burnin_iter)(_job.sstate.tune.malatune.proposed),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.malafrequency),
        "%) running frequency"
      )))
    end
  end

  if isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(burninbody, :(tune!(_job.sstate.tune.totaltune, _job.tuner.totaltuner)))
    push!(burninbody, :(_job.sstate.sqrttunestep = sqrt(_job.sstate.tune.totaltune.step)))
  end

  if job.tuner.smmalatuner.verbose
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.malatuner.verbose
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.malatune)))
  end

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.malatuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    push!(burninbody, :(_job.sstate.tune.totaltune.totproposed += _job.sstate.tune.totaltune.proposed))

    push!(burninbody, :(_job.sstate.tune.totaltune.proposed = 0))

    if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
      push!(burninbody, :(_job.sstate.tune.totaltune.accepted = 0))
      push!(burninbody, :(_job.sstate.tune.totaltune.rate = NaN))
    end

    push!(
      body,
      Expr(
        :if,
        :(
          _job.sstate.tune.totaltune.totproposed <= _job.range.burnin &&
          mod(_job.sstate.tune.totaltune.proposed, _job.tuner.totaltuner.period) == 0
        ),
        Expr(:block, burninbody...)
      )
    )
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
