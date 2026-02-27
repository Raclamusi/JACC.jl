
JACC.zeros(::VectorEngineBackend, T, dims...) = VectorEngine.zeros(T, dims...)
JACC.ones(::VectorEngineBackend, T, dims...) = VectorEngine.ones(T, dims...)
JACC.fill(::VectorEngineBackend, value, dims...) = VectorEngine.fill(value, dims...)
