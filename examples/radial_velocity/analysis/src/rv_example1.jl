if(!isdefined(:RvModelKeplerian)) 
  include("rv_model.jl")    # provides plogtarget
  using RvModelKeplerian
end

include("utils_ex.jl")
param_true = make_param_true_ex1()

# Make true values and one set of simulated data with noise
num_obs = 50
observation_timespan = 2*365.25                                
times = observation_timespan*sort(rand(num_obs));             # For observations w/ uneven spacing
sigma_obs = 2.0*ones(num_obs);    
model_true = map(t->calc_model_rv(param_true, t),times);

set_times(times);                  # For observations w/ uneven spacing
set_sigma_obs(sigma_obs);
set_obs( model_true .+ sigma_obs .* randn(length(times)) );

if(!isdefined(:ForwardDiff))        using ForwardDiff       end

function test_rv_example1()
  plogtarget(param_true)
  ForwardDiff.gradient(plogtarget,param_true)
  ForwardDiff.hessian(plogtarget,param_true)
end

#=
=#

