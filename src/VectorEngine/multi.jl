module Multi

import Base: Callable
using JACC
import JACC.VectorEngineImpl: VectorEngineBackend, _nstreams, _use_packed_vector, _parallel_for_kernel, _parallel_for_packed_kernel, _parallel_reduce_kernel, _parallel_reduce_packed_kernel
using VectorEngine
using VectorEngine.VEDA

@inline ndevices() = VectorEngine.ndevices()

function JACC.Multi.ndev(::VectorEngineBackend)
    return ndevices()
end

struct ArrayPart{T, N}
    a::VEDeviceArray{T, N, AS.Global}
    dev_id::Int
    ndev::Int
    ghost_dims::Int
end

@inline Base.size(p::ArrayPart) = size(p.a)
@inline Base.length(p::ArrayPart) = length(p.a)
@inline Base.@propagate_inbounds Base.getindex(p::ArrayPart, i...) = getindex(p.a, i...)
@inline Base.@propagate_inbounds Base.setindex!(p::ArrayPart, v, i...) = setindex!(p.a, v, i...)
@inline device_id(p::ArrayPart) = p.dev_id
@inline ghost_dims(p::ArrayPart) = p.ghost_dims

@inline JACC.Multi.device_id(::VectorEngineBackend, p::ArrayPart) = p.dev_id

struct MultiArray{T, N, NG}
    a1::Vector{ArrayPart{T, N}}
    a2::Vector{VEArray{T, N}}
    orig_size::NTuple{N, Int}
end

JACC.to_host(x::MultiArray) = convert(Base.Array, x)

@inline ghost_dims(x::MultiArray{T, N, NG}) where {T, N, NG} = NG
@inline JACC.Multi.part_length(::VectorEngineBackend, x::MultiArray) = size(x.a2[1])[end]

@inline process_param(x, dev_id) = x
@inline process_param(x::MultiArray, dev_id) = x.a1[dev_id]

JACC.Multi.multi_array_type(::VectorEngineBackend) = MultiArray

# FIXME:
#   - what about ghost elements
function Base.convert(::Type{Base.Array}, x::MultiArray{T, Rank}) where {T, Rank}
    ndev = x.a1[1].ndev
    total_length = x.orig_size[end]
    base_size = prod(x.orig_size[1:end-1])
    total_size = total_length * base_size
    partlen = cld(total_length, ndev)
    partsize = partlen * base_size
    ng = ghost_dims(x)
    ngsize = ng * base_size
    ret = Base.Array{T, Rank}(undef, x.orig_size)
    for i in 1:ndev
        device!(i - 1)
        dst_start = (i - 1) * partsize + 1
        dst_stop = min(i * partsize, total_size)
        copy_length = dst_stop - dst_start + 1
        src_start = (i == 1) ? 1 : 1 + ngsize
        copyto!(ret, dst_start, x.a2[i], src_start, copy_length)
    end
    device!(0)
    return ret
end

function make_multi_array(x::Base.Array{T, Rank}) where {T, Rank}
    ndev = ndevices()
    total_length = size(x, Rank)
    partlen = cld(total_length, ndev)
    ndev = cld(total_length, partlen)
    parts = Vector{VEArray{T, Rank}}(undef, ndev)
    devparts = Vector{ArrayPart{T, Rank}}(undef, ndev)

    for i in 1:ndev
        device!(i - 1)
        start = (i - 1) * partlen + 1
        stop = min(i * partlen, total_length)
        parts[i] = VEArray(selectdim(x, Rank, start:stop))
        devparts[i] = ArrayPart(vedaconvert(parts[i]), i, ndev, 0)
    end

    device!(0)
    return MultiArray{T, Rank, 0}(devparts, parts, size(x))
end

function make_multi_array(x::Base.Array{T, Rank}, ghost_dims) where {T, Rank}
    ndev = ndevices()
    total_length = size(x, Rank)
    partlen = cld(total_length, ndev)
    ndev = cld(total_length, partlen)
    parts = Vector{VEArray{T, Rank}}(undef, ndev)
    devparts = Vector{ArrayPart{T, Rank}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        device!(i - 1)
        start = (i - 1) * partlen + 1
        stop = min(i * partlen, total_length)
        if i != 1
            start -= ng
        end
        if i != ndev
            stop += ng
        end
        parts[i] = VEArray(selectdim(x, Rank, start:stop))
        devparts[i] = ArrayPart(vedaconvert(parts[i]), i, ndev, ng)
    end

    device!(0)
    return MultiArray{T, Rank, ng}(devparts, parts, size(x))
end

function JACC.Multi.array(::VectorEngineBackend, x::Base.Array; ghost_dims)
    if ghost_dims == 0 || ndevices() == 1
        return make_multi_array(x)
    else
        return make_multi_array(x, ghost_dims)
    end
end

function JACC.Multi.ghost_shift(::VectorEngineBackend, i::Integer, arr::ArrayPart)
    dev_id = device_id(arr)
    if dev_id == 1
        ind = i
    else
        ind = i + ghost_dims(arr)
    end
    return ind
end

function JACC.Multi.ghost_shift(::VectorEngineBackend, i::NTuple{Rank, Integer}, arr::ArrayPart) where {Rank}
    dev_id = device_id(arr)
    if dev_id == 1
        ind = i
    else
        ind = (i[1:end-1]..., i[end] + ghost_dims(arr))
    end
    return ind
end

function JACC.Multi.sync_ghost_elems!(::VectorEngineBackend, arr::MultiArray{T, Rank}) where {T, Rank}
    ndev = arr.a1[1].ndev
    ng = ghost_dims(arr)
    ngsize = ng * prod(arr.orig_size[1:end-1])
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        device!(i - 1)
        partsize = length(arr.a2[i])
        ghost_lr = Base.Array(selectdim(arr.a2[i], Rank, (partsize + 1 - 2*ngsize):(partsize - ngsize)))
        device!(i)
        copyto!(arr.a1[i + 1], 1 + ngsize, ghost_lr, 1, length(ghost_lr))
    end

    #Right to left swapping
    for i in 2:ndev
        device!(i - 1)
        ghost_rl = Base.Array(selectdim(arr.a2[i], Rank, (1 + ngsize):(2*ngsize)))
        device!(i - 2)
        partsize = length(arr.a2[i - 1])
        copyto!(arr.a1[i - 1], partsize + 1 - ngsize, ghost_rl, 1, length(ghost_rl))
    end

    device!(0)
    return nothing
end

function JACC.Multi.copy!(::VectorEngineBackend, x::MultiArray, y::MultiArray)
    @boundscheck x.orig_size == y.orig_size

    ndev = x.a1[1].ndev

    if ghost_dims(x) == ghost_dims(y)
        for i in 1:ndev
            device!(i - 1)
            copyto!(x.a2[i], y.a2[i])
        end
    else
        x_ngsize = ghost_dims(x) * prod(x.orig_size[1:end-1])
        y_ngsize = ghost_dims(y) * prod(y.orig_size[1:end-1])
        for i in 1:ndev
            device!(i - 1)
            x_start = (i == 1) ? 1 : 1 + x_ngsize
            y_start = (i == 1) ? 1 : 1 + y_ngsize
            copysize = length(x.a2[i]) - ((i == 1 || i == ndev) ? x_ngsize : 2 * x_ngsize)
            copyto!(x.a2[i], x_start, y.a2[i], y_start, copysize)
        end
        JACC.Multi.sync_ghost_elems!(VectorEngineBackend(), x)
    end

    device!(0)
    return x
end


@inline function JACC.Multi.parallel_for(::VectorEngineBackend, N::Integer, f::Callable, x...)
    JACC.Multi.parallel_for(VectorEngineBackend(), (N,), f, x...)
end

@inline function JACC.Multi.parallel_for(::VectorEngineBackend, dims::NTuple{Rank, Integer}, f::Callable, x...) where {Rank}
    ndev = ndevices()
    partlen = cld(dims[end], ndev)
    ndev = cld(dims[end], partlen)
    lastlen = dims[end] - (ndev - 1) * partlen
    for dev in 1:ndev
        device_n = (dev == ndev) ? lastlen : partlen
        device!(dev - 1)
        nkernels = min(device_n, _nstreams())
        args = map(vedaconvert, process_param.(x, dev))
        veargs = VEDA.VEArgs()
        # veargs[[0,1]] are set for each kernel launch
        let i = 2
            for arg in args
                if sizeof(arg) > 0
                    veargs[i] = arg
                    i += 1
                end
            end
        end
        kernel_tt = Tuple{typeof(f), Int, typeof(dims), map(typeof, args)...}
        if _use_packed_vector()
            func = vefunction(_parallel_for_packed_kernel, kernel_tt)
        else
            func = vefunction(_parallel_for_kernel, kernel_tt)
        end
        for kernel_id in 0:nkernels-1
            offset = kernel_id * device_n ÷ nkernels
            partial_n = (kernel_id + 1) * device_n ÷ nkernels - offset
            veargs[0] = convert(Int, offset)
            veargs[1] = convert(typeof(dims), (dims[1:end-1]..., partial_n))
            VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, kernel_id, veargs.handle)
        end
        VEDA.vedaArgsDestroy(veargs.handle)
    end
    for dev in 1:ndev
        device!(dev - 1)
        VectorEngine.synchronize()
    end
    device!(0)
    return nothing
end

const _reduce_buffers = Dict{DataType, VectorEngine.VEVector}[]
@inline function _get_reduce_buffer(::Type{T}, dev::Integer, n::Integer) where {T}
    if isempty(_reduce_buffers)
        resize!(_reduce_buffers, ndevices())
        fill!(_reduce_buffers, Dict{DataType, VectorEngine.VEVector}())
    end
    buf = get!(_reduce_buffers[dev], T) do
        VEArray{T}(undef, n)
    end::VEArray{T, 1}
    if length(buf) < n
        buf = VEArray{T}(undef, n)
        _reduce_buffers[dev][T] = buf
    end
    return @inbounds view(buf, Base.OneTo(n))
end

@inline function JACC.Multi.parallel_reduce(::VectorEngineBackend, N::Integer, f::Callable, x...; op = +, init = 0.0)
    JACC.Multi.parallel_reduce(VectorEngineBackend(), (N,), f, x...; op, init)
end

@inline function JACC.Multi.parallel_reduce(::VectorEngineBackend, dims::NTuple{Rank, Integer}, f::Callable, x...; op = +, init = 0.0) where {Rank}
    ndev = ndevices()
    partlen = cld(dims[end], ndev)
    ndev = cld(dims[end], partlen)
    lastlen = dims[end] - (ndev - 1) * partlen
    type = typeof(init)
    # device_ret = Vector{SubArray{type, 1, VEArray{type, 1}, Tuple{Base.OneTo{Int64}}, true}}(undef, ndev)
    device_ret = Vector{VEArray{type, 1}}(undef, ndev)
    for dev in 1:ndev
        device_n = (dev == ndev) ? lastlen : partlen
        device!(dev - 1)
        nkernels = min(device_n, _nstreams())
        # device_ret[dev] = _get_reduce_buffer(type, dev, nkernels)
        device_ret[dev] = VEArray{type, 1}(undef, nkernels)
        args = map(vedaconvert, (init, process_param.(x, dev)...))
        veargs = VEDA.VEArgs()
        # veargs[[0,1,2]] are set for each kernel launch
        let i = 3
            for arg in args
                if sizeof(arg) > 0
                    veargs[i] = arg
                    i += 1
                end
            end
        end
        kernel_tt = Tuple{typeof(f), typeof(op), Int, typeof(dims), VectorEngine.VEDeviceArray{type, 0, AS.Global}, map(typeof, args)...}
        if _use_packed_vector()
            func = vefunction(_parallel_reduce_packed_kernel, kernel_tt)
        else
            func = vefunction(_parallel_reduce_kernel, kernel_tt)
        end
        for kernel_id in 0:nkernels-1
            offset = kernel_id * device_n ÷ nkernels
            partial_n = (kernel_id + 1) * device_n ÷ nkernels - offset
            veargs[0] = convert(Int, offset)
            veargs[1] = convert(typeof(dims), (dims[1:end-1]..., partial_n))
            veargs[2] = vedaconvert(@inbounds view(device_ret[dev], kernel_id + 1))
            VEDA.@check VEDA.vedaLaunchKernel(func.fun.handle, kernel_id, veargs.handle)
        end
        VEDA.vedaArgsDestroy(veargs.handle)
    end
    ret = init
    for dev in 1:ndev
        device!(dev - 1)
        VectorEngine.synchronize()
        ret = reduce(op, collect(device_ret[dev]); init = ret)
    end
    device!(0)
    return ret
end

end # module Multi
