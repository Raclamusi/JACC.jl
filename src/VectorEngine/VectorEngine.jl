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

const use_packed_vector_ref = Ref(false)
@inline function use_packed_vector()
    return use_packed_vector_ref[]
end

function __init__()
    if get(ENV, "JACC_VE_USE_PACKED", "0") in ("1", "true", "True", "TRUE")
        use_packed_vector_ref[] = true
    end
end

@inline function JACC.parallel_for(f, ::VectorEngineBackend, N::Integer, x...)
    function kernel(offset_i, N, x...)
        @vectorize for delta_i in 1:N
            i = offset_i + delta_i
            @inline f(i, x...)
        end
        return
    end
    function packed_kernel(offset_i, N, x...)
        @vectorize length=512 for delta_i in 1:N
            i = offset_i + delta_i
            @inline f(i, x...)
        end
        return
    end
    ns = min(N, nstreams())
    args = map(vedaconvert, x)
    veargs = VEDA.VEArgs()
    for i in eachindex(args)
        veargs[i+1] = args[i]
    end
    kernel_tt = Tuple{Int, Int, map(typeof, args)...}
    if use_packed_vector()
        func = vefunction(packed_kernel, kernel_tt)
    else
        func = vefunction(kernel, kernel_tt)
    end
    for s in 0:ns-1
        offset_i = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_i
        veargs[0] = offset_i
        veargs[1] = partial_n
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, s, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
    synchronize()
end

@inline function JACC.parallel_for(
        f, spec::LaunchSpec{VectorEngineBackend}, N::Integer, x...)
    # TODO
end

@inline function JACC.parallel_for(
        f, ::VectorEngineBackend, (M, N)::NTuple{2, Integer}, x...)
    function kernel(offset_j, (M, N), x...)
        for delta_j in 1:N
            j = offset_j + delta_j
            @vectorize for i in 1:M
                @inline f(i, j, x...)
            end
        end
        return
    end
    function packed_kernel(offset_j, (M, N), x...)
        for delta_j in 1:N
            j = offset_j + delta_j
            @vectorize length=512 for i in 1:M
                @inline f(i, j, x...)
            end
        end
        return
    end
    ns = min(N, nstreams())
    args = map(vedaconvert, x)
    veargs = VEDA.VEArgs()
    for i in eachindex(args)
        veargs[i+1] = args[i]
    end
    kernel_tt = Tuple{Int, Tuple{typeof(M), Int}, map(typeof, args)...}
    if use_packed_vector()
        func = vefunction(packed_kernel, kernel_tt)
    else
        func = vefunction(kernel, kernel_tt)
    end
    for s in 0:ns-1
        offset_j = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_j
        veargs[0] = offset_j
        veargs[1] = (M, partial_n)
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, s, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
    synchronize()
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{VectorEngineBackend},
        (M, N)::NTuple{2, Integer}, x...)
    # TODO
end

@inline function JACC.parallel_for(
        f, ::VectorEngineBackend, (L, M, N)::NTuple{3, Integer}, x...)
    function kernel(offset_k, (L, M, N), x...)
        for delta_k in 1:N
            k = offset_k + delta_k
            for j in 1:M
                @vectorize for i in 1:L
                    @inline f(i, j, k, x...)
                end
            end
        end
        return
    end
    function packed_kernel(offset_k, (L, M, N), x...)
        for delta_k in 1:N
            k = offset_k + delta_k
            for j in 1:M
                @vectorize length=512 for i in 1:L
                    @inline f(i, j, k, x...)
                end
            end
        end
        return
    end
    ns = min(N, nstreams())
    args = map(vedaconvert, x)
    veargs = VEDA.VEArgs()
    for i in eachindex(args)
        veargs[i+1] = args[i]
    end
    kernel_tt = Tuple{Int, Tuple{typeof(L), typeof(M), Int}, map(typeof, args)...}
    if use_packed_vector()
        func = vefunction(packed_kernel, kernel_tt)
    else
        func = vefunction(kernel, kernel_tt)
    end
    for s in 0:ns-1
        offset_k = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_k
        veargs[0] = offset_k
        veargs[1] = (L, M, partial_n)
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, s, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
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

const reduce_buffers = Dict{DataType, VectorEngine.VEVector}()
@inline function get_reduce_buffer(::Type{T}, n::Integer) where {T}
    buf = get!(reduce_buffers, T) do
        VEArray{T}(undef, n)
    end::VEArray{T, 1}
    if length(buf) < n
        buf = VEArray{T}(undef, n)
        reduce_buffers[T] = buf
    end
    return view(buf, 1:n)
end

@inline function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{VectorEngineBackend}, N::Integer, f, x...)
    # TODO
end

@inline function JACC.parallel_reduce(
        f, ::VectorEngineBackend, N::Integer, x...; op, init)
    function kernel(offset_i, N, ret, init, x...)
        tmp = init
        @vectorize for delta_i in 1:N
            i = offset_i + delta_i
            tmp = @inline op(tmp, f(i, x...))
        end
        @inbounds ret[] = tmp
        return
    end
    function packed_kernel(offset_i, N, ret, init, x...)
        tmp = init
        @vectorize length=512 for delta_i in 1:N
            i = offset_i + delta_i
            tmp = @inline op(tmp, f(i, x...))
        end
        @inbounds ret[] = tmp
        return
    end
    ns = min(N, nstreams())
    ret = get_reduce_buffer(typeof(init), ns)
    args = map(vedaconvert, (init, x...))
    veargs = VEDA.VEArgs()
    for i in eachindex(args)
        veargs[i+2] = args[i]
    end
    kernel_tt = Tuple{Int, Int, VectorEngine.VEDeviceArray{typeof(init), 0, AS.Global}, map(typeof, args)...}
    if use_packed_vector()
        func = vefunction(packed_kernel, kernel_tt)
    else
        func = vefunction(kernel, kernel_tt)
    end
    for s in 0:ns-1
        offset_i = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_i
        veargs[0] = offset_i
        veargs[1] = partial_n
        veargs[2] = vedaconvert(view(ret, s + 1))
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, s, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
    synchronize()
    return reduce(op, collect(ret); init)
end

@inline function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{VectorEngineBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    # TODO
end

@inline function JACC.parallel_reduce(f, ::VectorEngineBackend,
        (M, N)::NTuple{2, Integer}, x...; op, init)
    function kernel(offset_j, (M, N), ret, init, x...)
        tmp = init
        @vectorize for i in 1:M
            for delta_j in 1:N
                j = offset_j + delta_j
                tmp = @inline op(tmp, f(i, j, x...))
            end
        end
        @inbounds ret[] = tmp
        return
    end
    function packed_kernel(offset_j, (M, N), ret, init, x...)
        tmp = init
        @vectorize length=512 for i in 1:M
            for delta_j in 1:N
                j = offset_j + delta_j
                tmp = @inline op(tmp, f(i, j, x...))
            end
        end
        @inbounds ret[] = tmp
        return
    end
    ns = min(N, nstreams())
    ret = get_reduce_buffer(typeof(init), ns)
    args = map(vedaconvert, (init, x...))
    veargs = VEDA.VEArgs()
    for i in eachindex(args)
        veargs[i+2] = args[i]
    end
    kernel_tt = Tuple{Int, Tuple{typeof(M), Int}, VectorEngine.VEDeviceArray{typeof(init), 0, AS.Global}, map(typeof, args)...}
    if use_packed_vector()
        func = vefunction(packed_kernel, kernel_tt)
    else
        func = vefunction(kernel, kernel_tt)
    end
    for s in 0:ns-1
        offset_j = s * N ÷ ns
        partial_n = (s + 1) * N ÷ ns - offset_j
        veargs[0] = offset_j
        veargs[1] = (M, partial_n)
        veargs[2] = vedaconvert(view(ret, s + 1))
        VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, s, veargs.handle)
    end
    VEDA.vedaArgsDestroy(veargs.handle)
    synchronize()
    return reduce(op, collect(ret); init)
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
