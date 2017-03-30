export getzero, getone

# Currently support scalars, arrays and Tuples. Should also support Sets and Dicts.
typealias BasicAGL Union{AbstractFloat, AbstractArray, Tuple}

# Function to return appropriate multiplicative zero data for each supported type.
@inline getzero(val::AbstractFloat) = 0.0
@inline getzero(val::AbstractArray) = zeros(val)
@inline getzero(val::Tuple) = map(getzero, val)

# Function to return appropriate multiplicative one data for each supported type.
@inline getone(val::AbstractFloat) = 1.0
@inline getone(val::AbstractArray) = ones(val)
@inline getone(val::Tuple) = map(getone, val)

# Function to return random uniformly distributed numbers of the correct shape for
# each supported type.
@inline getrand(val::AbstractFloat, lb, ub) = rand() * (ub - lb) + lb
@inline getrand(val::AbstractArray, lb, ub) = rand(size(val)) * (ub - lb) + lb
@inline getrand(val::Tuple, lb, ub) = map(getrand, val)
