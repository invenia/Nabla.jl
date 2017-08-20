# Nabla

[![Build Status](https://travis-ci.org/invenia/Nabla.jl.svg?branch=master)](https://travis-ci.org/invenia/Nabla.jl)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/g0gun5dxbkt631am/branch/master?svg=true)](https://ci.appveyor.com/project/iamed2/nabla-jl/branch/master)
[![codecov.io](http://codecov.io/github/invenia/Nabla.jl/coverage.svg?branch=master)](http://codecov.io/github/invenia/Nabla.jl?branch=master)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/Nabla.jl/stable)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.github.io/Nabla.jl/latest)

Nabla.jl is a reverse-mode automatic differentiation package. Although it is a general purpose tool, the decisions made regarding our priorities for development are motivated from the perspective of its usefulness in Machine Learning. For example, we have prioritised having excellent support for linear algebra optimisations and higher-order functions, rather than making it possible to take higher-order derivatives (i.e. Nabla *currently* only supports first-order derivatives).

It is currently under active development, and improved documentation will be made available soon. Until then, take a look at the code in the examples folder for an indication of how to use the package. Given the early stage of development, we anticipate the presence of a number of bugs and performance issues. If you think you may have run into any of these, or have any particular feature requests, please raise an issue and let us know.
