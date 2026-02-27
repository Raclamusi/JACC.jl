module VectorEngineImpl

import JACC
import JACC: LaunchSpec
using VectorEngine
using VectorEngine.VEDA

struct VectorEngineBackend end

@inline JACC.get_backend(::Val{:vectorengine}) = VectorEngineBackend()

include("array.jl")
# include("multi.jl")
# include("async.jl")
# include("experimental/experimental.jl")

JACC.synchronize(::VectorEngineBackend) = synchronize()

JACC.default_stream(::VectorEngineBackend) = nothing

JACC.create_stream(::VectorEngineBackend) = nothing

@inline function nstreams()
    count_ref = Ref{Cint}()
    VEDA.@check VEDA.vedaCtxStreamCnt(count_ref)
    return count_ref[]
end

@inline function JACC.parallel_for(f, ::VectorEngineBackend, N::Integer, x...)
    function kernel(offset_i, N, x...)
        @inbounds @vectorize for delta_i in 1:N
            i = offset_i + delta_i
            @inline f(i, x...)
        end
        return
    end
    ns = min(N, nstreams())
    for s in 0:ns-1
        offset_i = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_i
        @veda stream=s kernel(offset_i, partial_n, x...)
    end
    synchronize()
end

@inline function JACC.parallel_for(
        f, spec::LaunchSpec{VectorEngineBackend}, N::Integer, x...)
    # TODO
end

@inline function JACC.parallel_for(
        f, ::VectorEngineBackend, (M, N)::NTuple{2, Integer}, x...)
    function kernel(offset_j, (M, N), x...)
        @inbounds for delta_j in 1:N
            j = offset_j + delta_j
            @vectorize for i in 1:M
                @inline f(i, j, x...)
            end
        end
        return
    end
    ns = min(N, nstreams())
    for s in 0:ns-1
        offset_j = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_j
        @veda stream=s kernel(offset_j, (M, partial_n), x...)
    end
    synchronize()
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{VectorEngineBackend},
        (M, N)::NTuple{2, Integer}, x...)
    # TODO
end

@inline function JACC.parallel_for(
        f, ::VectorEngineBackend, (L, M, N)::NTuple{3, Integer}, x...)
    function kernel(offset_k, (L, M, N), x...)
        @inbounds for delta_k in 1:N
            k = offset_k + delta_k
            for j in 1:M
                @vectorize for i in 1:L
                    @inline f(i, j, k, x...)
                end
            end
        end
        return
    end
    ns = min(N, nstreams())
    for s in 0:ns-1
        offset_k = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_k
        @veda stream=s kernel(offset_k, (L, M, partial_n), x...)
    end
    synchronize()
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{VectorEngineBackend},
        (L, M, N)::NTuple{3, Integer}, x...)
    # TODO
end

mutable struct VectorEngineReduceWorkspace{T} <: JACC.ReduceWorkspace
    tmp::VEArray{T, 1}
    ret::VEArray{T, 1}
end

@inline function JACC.reduce_workspace(::VectorEngineBackend, init::T) where {T}
    VectorEngineReduceWorkspace{T}(VEArray{T, 1}(undef, 0), VEArray([init]))
end

@inline JACC.get_result(wk::VectorEngineReduceWorkspace{T}) where {T} = collect(wk.ret)[]

@inline function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{VectorEngineBackend}, N::Integer, f, x...)
    # TODO
end

@inline function JACC.parallel_reduce(
        f, ::VectorEngineBackend, N::Integer, x...; op, init)
    function kernel(offset_i, N, id, init, ret, x...)
        tmp = init
        @inbounds @vectorize for delta_i in 1:N
            i = offset_i + delta_i
            tmp = @inline op(tmp, f(i, x...))
        end
        @inbounds ret[id] = tmp
        return
    end
    ns = min(N, nstreams())
    ret = VEArray{typeof(init)}(undef, ns)
    for s in 0:ns-1
        offset_i = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_i
        @veda stream=s kernel(offset_i, partial_n, s + 1, init, ret, x...)
    end
    synchronize()
    return reduce(op, collect(ret))
end

@inline function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{VectorEngineBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    # TODO
end

@inline function JACC.parallel_reduce(f, ::VectorEngineBackend,
        (M, N)::NTuple{2, Integer}, x...; op, init)
    function kernel(offset_j, (M, N), id, init, ret, x...)
        tmp = init
        @inbounds for delta_j in 1:N
            j = offset_j + delta_j
            @vectorize for i in 1:M
                tmp = @inline op(tmp, f(i, j, x...))
            end
        end
        @inbounds ret[id] = tmp
        return
    end
    ns = min(N, nstreams())
    ret = VEArray{typeof(init)}(undef, ns)
    for s in 0:ns-1
        offset_j = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_j
        @veda stream=s kernel(offset_j, (M, partial_n), s + 1, init, ret, x...)
    end
    synchronize()
    return reduce(op, collect(ret))
end

@inline function JACC.parallel_reduce(
        f, ::VectorEngineBackend, dims::NTuple{N, Integer},
        x...; op, init)::typeof(init) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(
        JACC.ReduceKernel1DND{typeof(init)}(), prod(dims), ids, f,
        x...; op = op, init = init)
end

JACC.sync_workgroup(::VectorEngineBackend) = nothing

JACC.array_type(::VectorEngineBackend) = VEArray

JACC.array(::VectorEngineBackend, x::AbstractArray) = VEArray(x)

# JACC.shared(::VectorEngineBackend, x::AbstractArray) = x

end
