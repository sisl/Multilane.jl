struct GaussianCopula
    cov::Array{Float64, 2}
    _chol::Array{Float64, 2}
    # GaussianCopula(cov::Array{Float64,2}) = new(cov, chol(cov, Val{:L}))
    # two argument constructor is allowed for easy copy-paste for debugging
end
GaussianCopula(cov::Array{Float64,2}) = GaussianCopula(cov, ctranspose(chol(cov)))
GaussianCopula(ndim::Int, rho::Float64) = GaussianCopula(rho*ones(ndim, ndim) + (1.0-rho)*diagm(ones(ndim)))

ndim(c::GaussianCopula) = size(c.cov,1)

normal_pdf(x) = exp.(-(x.^2)/2.0)/sqrt(2*pi) # assumes zero mean, unit variance
normal_cdf(x) = 0.5*(1.0 + erf.(x/sqrt(2.0))) # assumes zero mean, unit variance

function sample_mvn(rng::AbstractRNG, cov_chol::Array{Float64})
    return cov_chol*randn(rng, size(cov_chol,1))
end

function rand(rng::AbstractRNG, c::GaussianCopula, sample::Vector{Float64}=Vector{Float64}(ndim(c)))
    return normal_cdf(sample_mvn(rng, c._chol))
end
rand(c::GaussianCopula) = rand(Base.GLOBAL_RNG, c)
