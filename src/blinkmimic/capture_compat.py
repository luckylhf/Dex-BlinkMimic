"""Per-Data Warp scratch arena for vGPUs without CUDA memory pools.

MuJoCo-Warp 3.11 allocates temporary collision, constraint and solver arrays
inside each physics call. On a vGPU without async allocation those arrays must
be allocated before capture. Reuse them by original allocation site and call
ordinal, with separate arenas per Data/device. Kernels, array initialization,
physics parameters and source packages remain unchanged.
"""
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
import sys
import os
import warp as wp

_installed=False
_active=ContextVar('dex_warp_scratch_scope',default=None)
_bypass=ContextVar('dex_warp_allocator_bypass',default=False)
_contexts={}
_reuse_enabled=True


def _allocator(name,original):
    @wraps(original)
    def allocate(*args,**kwargs):
        scope=_active.get()
        caller=sys._getframe(1)
        if (scope is None or _bypass.get() or not _reuse_enabled or
                '/mujoco_warp/' not in caller.f_code.co_filename):
            return original(*args,**kwargs)
        site=(caller.f_code.co_filename,caller.f_lineno,name)
        ordinal=scope['counts'][site];scope['counts'][site]+=1
        source=args[0] if args else kwargs.get('src',kwargs.get('shape'))
        if hasattr(source,'shape'):
            shape=(tuple(source.shape),str(source.dtype),tuple(source.strides))
        elif isinstance(source,(tuple,list)):
            shape=tuple(source)
        else:
            shape=source
        signature=(repr(shape),repr(args[1:]),repr({k:str(v) for k,v in kwargs.items() if k not in ('src','shape')}))
        key=(scope['data'],str(wp.get_device()),site,ordinal,signature)
        if key not in _contexts:
            if wp.get_device().is_capturing:
                raise RuntimeError('Warp scratch was not warmed before capture: '+repr((site,ordinal,signature)))
            token=_bypass.set(True)
            try:
                _contexts[key]=original(*args,**kwargs)
            finally:
                _bypass.reset(token)
        else:
            result=_contexts[key]
            if name in ('zeros','zeros_like'):result.zero_()
            elif name=='clone':wp.copy(result,source)
            elif name in ('full','full_like'):result.fill_(kwargs['value'] if 'value' in kwargs else args[1])
            elif name in ('ones','ones_like'):result.fill_(1)
        return _contexts[key]
    return allocate


def install_capture_preallocation():
    global _installed
    if _installed or os.environ.get('DEX_CAPTURE_ARENA')!='1' or not wp.is_cuda_available() or wp.is_mempool_supported('cuda:0'):
        return
    import mujoco_warp as mjwarp
    for name in ('empty','zeros','empty_like','zeros_like','clone','full','full_like','ones','ones_like'):
        if hasattr(wp,name):setattr(wp,name,_allocator(name,getattr(wp,name)))
    for name in ('step','forward','reset_data'):
        original=getattr(mjwarp,name)
        def scoped(m,d,*args,_fn=original,**kwargs):
            token=_active.set(dict(data=id(d),counts=defaultdict(int)))
            try:return _fn(m,d,*args,**kwargs)
            finally:_active.reset(token)
        setattr(mjwarp,name,scoped)
    _installed=True
    print('Dex capture compatibility: per-Data Warp scratch arena (vGPU lacks memory pools)',flush=True)
