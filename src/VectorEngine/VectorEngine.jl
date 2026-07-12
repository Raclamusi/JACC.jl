module VectorEngineImpl

import JACC
import JACC: LaunchSpec
using VectorEngine
using VectorEngine.VEDA
using Base.Cartesian

struct VectorEngineBackend end

@inline JACC.get_backend(::Val{:vectorengine}) = VectorEngineBackend()

include("array.jl")
# include("multi.jl")
# include("async.jl")
# include("experimental/experimental.jl")

JACC.synchronize(::VectorEngineBackend) = VectorEngine.synchronize()

JACC.default_stream(::VectorEngineBackend) = nothing

JACC.create_stream(::VectorEngineBackend) = nothing

@inline function _nstreams()
    count_ref = Ref{Cint}()
    VEDA.@check VEDA.vedaCtxStreamCnt(count_ref)
    return count_ref[]
end

const _use_packed_vector_ref = Ref(false)
@inline function _use_packed_vector()
    return _use_packed_vector_ref[]
end

function __init__()
    if get(ENV, "JACC_VE_USE_PACKED", "0") in ("1", "true", "True", "TRUE")
        _use_packed_vector_ref[] = true
    end
end

@inline function JACC.parallel_for(f, ::VectorEngineBackend, N::Integer, x...)
    JACC.parallel_for(f, VectorEngineBackend(), (N,), x...)
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{VectorEngineBackend}, N::Integer, x...)
    JACC.parallel_for(f, spec, (N,), x...)
end

@generated function _parallel_for_kernel(f, offset, dims::NTuple{Rank, Integer}, x...) where {Rank}
    quote
        @nloops $(Rank-1) j (d -> (d == $(Rank-1)) ? (offset .+ (1:dims[end])) : (1:dims[d+1])) begin
            @vectorize for i in 1:dims[1]
                $(Rank == 1 ? :(i += offset) : nothing)
                @inline f(i, $(map(d -> Symbol("j_", d), 1:Rank-1)...), x...)
            end
        end
        return
    end
end

@generated function _parallel_for_packed_kernel(f, offset, dims::NTuple{Rank, Integer}, x...) where {Rank}
    quote
        @nloops $(Rank-1) j (d -> (d == $(Rank-1)) ? (offset .+ (1:dims[end])) : (1:dims[d+1])) begin
            @vectorize length=512 for i in 1:dims[1]
                $(Rank == 1 ? :(i += offset) : nothing)
                @inline f(i, $(map(d -> Symbol("j_", d), 1:Rank-1)...), x...)
            end
        end
        return
    end
end

@inline function JACC.parallel_for(f, ::VectorEngineBackend, dims::NTuple{Rank, Integer}, x...) where {Rank}
    JACC.parallel_for(f, LaunchSpec{VectorEngineBackend}(), dims, x...)
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{VectorEngineBackend}, dims::NTuple{Rank, Integer}, x...) where {Rank}
    nkernels = min(dims[end], _nstreams())
    args = map(vedaconvert, x)
    veargs = VEDA.VEArgs()
    # veargs[[0,1]] are set for each kernel launch
    for i in eachindex(args)
        veargs[i+1] = args[i]
    end
    kernel_tt = Tuple{typeof(f), Int, typeof(dims), map(typeof, args)...}
    if _use_packed_vector()
        func = vefunction(_parallel_for_packed_kernel, kernel_tt)
    else
        func = vefunction(_parallel_for_kernel, kernel_tt)
    end
    for kernel_id in 0:nkernels-1
        offset = kernel_id * dims[end] ÷ nkernels
        partial_n = (kernel_id + 1) * dims[end] ÷ nkernels - offset
        veargs[0] = convert(Int, offset)
        veargs[1] = convert(typeof(dims), (dims[1:end-1]..., partial_n))
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, kernel_id, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
    if spec.sync
        VectorEngine.synchronize()
    end
end

mutable struct VectorEngineReduceWorkspace{T} <: JACC.ReduceWorkspace
    ret::T
end

@inline function JACC.reduce_workspace(::VectorEngineBackend, init::T) where {T}
    VectorEngineReduceWorkspace{T}(init)
end

@inline JACC.get_result(wk::VectorEngineReduceWorkspace{T}) where {T} = wk.ret

const _reduce_buffers = Dict{DataType, VectorEngine.VEVector}()
@inline function _get_reduce_buffer(::Type{T}, n::Integer) where {T}
    buf = get!(_reduce_buffers, T) do
        VEArray{T}(undef, n)
    end::VEArray{T, 1}
    if length(buf) < n
        buf = VEArray{T}(undef, n)
        _reduce_buffers[T] = buf
    end
    return @inbounds view(buf, Base.OneTo(n))
end

@inline function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{VectorEngineBackend}, N::Integer, f, x...)
    JACC._parallel_reduce!(reducer, (N,), f, x...)
end

@inline function JACC.parallel_reduce(f, ::VectorEngineBackend, N::Integer, x...; op, init)
    JACC.parallel_reduce(f, VectorEngineBackend(), (N,), x...; op, init)
end

@inline function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{VectorEngineBackend}, dims::NTuple{Rank, Integer}, f, x...) where {Rank}
    # Asynchronous parallel_reduce is not supported on VectorEngine
    reducer.ret = JACC.parallel_reduce(f, VectorEngineBackend(), dims, x...; op = reducer.op, init = reducer.init)
    return
end

@generated function _parallel_reduce_kernel(f, op, offset, dims::NTuple{Rank, Integer}, ret, init, x...) where {Rank}
    quote
        tmp = init
        @vectorize for i in 1:dims[1]
            $(Rank == 1 ? :(i += offset) : nothing)
            @nloops $(Rank-1) j (d -> (d == $(Rank-1)) ? (offset .+ (1:dims[end])) : (1:dims[d+1])) begin
                tmp = @inline op(tmp, f(i, $(map(d -> Symbol("j_", d), 1:Rank-1)...), x...))
            end
        end
        @inbounds ret[] = tmp
        return
    end
end

@generated function _parallel_reduce_packed_kernel(f, op, offset, dims::NTuple{Rank, Integer}, ret, init, x...) where {Rank}
    quote
        tmp = init
        @vectorize length=512 for i in 1:dims[1]
            $(Rank == 1 ? :(i += offset) : nothing)
            @nloops $(Rank-1) j (d -> (d == $(Rank-1)) ? (offset .+ (1:dims[end])) : (1:dims[d+1])) begin
                tmp = @inline op(tmp, f(i, $(map(d -> Symbol("j_", d), 1:Rank-1)...), x...))
            end
        end
        @inbounds ret[] = tmp
        return
    end
end

@inline function JACC.parallel_reduce(f, ::VectorEngineBackend, dims::NTuple{Rank, Integer}, x...; op, init) where {Rank}
    nkernels = min(dims[end], _nstreams())
    ret = _get_reduce_buffer(typeof(init), nkernels)
    args = map(vedaconvert, (init, x...))
    veargs = VEDA.VEArgs()
    # veargs[[0,1,2]] are set for each kernel launch
    for i in eachindex(args)
        veargs[i+2] = args[i]
    end
    kernel_tt = Tuple{typeof(f), typeof(op), Int, typeof(dims), VectorEngine.VEDeviceArray{typeof(init), 0, AS.Global}, map(typeof, args)...}
    if _use_packed_vector()
        func = vefunction(_parallel_reduce_packed_kernel, kernel_tt)
    else
        func = vefunction(_parallel_reduce_kernel, kernel_tt)
    end
    for kernel_id in 0:nkernels-1
        offset = kernel_id * dims[end] ÷ nkernels
        partial_n = (kernel_id + 1) * dims[end] ÷ nkernels - offset
        veargs[0] = convert(Int, offset)
        veargs[1] = convert(typeof(dims), (dims[1:end-1]..., partial_n))
        veargs[2] = vedaconvert(@inbounds view(ret, kernel_id + 1))
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, kernel_id, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
    VectorEngine.synchronize()
    return reduce(op, collect(ret); init)
end

JACC.sync_workgroup(::VectorEngineBackend) = nothing

JACC.array_type(::VectorEngineBackend) = VEArray

JACC.array(::VectorEngineBackend, x::AbstractArray) = VEArray(x)

# JACC.shared(::VectorEngineBackend, x::AbstractArray) = x

end
