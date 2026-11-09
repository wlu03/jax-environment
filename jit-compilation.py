from time import perf_counter

import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x * x + 2 * x + 1)

fast_f = jax.jit(f) # create a callable version of f that uses JIT compilation
# now this can be used as fast_f(x)

# Prepare the input before timing.
x = jnp.linspace(0, 1, 1_000_000).block_until_ready() # create 1000000 points between (0-1)

def time_ms(fn, runs=100):
    start = perf_counter()
    for _ in range(runs):
        fn(x).block_until_ready()  # Wait for the actual result.
    return (perf_counter() - start) * 1000 / runs

# Warm up ordinary JAX operations for a fair comparison.
f(x).block_until_ready()

first = time_ms(fast_f, runs=1)  # Compilation + execution.
normal = time_ms(f)
compiled = time_ms(fast_f)      # Reuse compiled code.


print(f"First JIT call:    {first:.3f} ms")
print(f"Without @jit:     {normal:.3f} ms per call")
print(f"After JIT warmup: {compiled:.3f} ms per call")
print(f"Speedup:          {normal / compiled:.2f}x")


# Notes:
    # JAX allows us to transform python functions. JAX does this by reducing each function into a sequenc
    # of primitive operations. For example
def f(x):
    return x * x + jnp.sin(x) 

# this computation can be represented as three primitive ops

# a = mul(x * x)
# b = sin(x)
# result = add(a,b) 

# this can be viewed using make_jaxpr(<function_name>)(<input>)
print(jax.make_jaxpr(f)(2.0)) # run this to see the primiitives and types

# JAX tracing records supports operations on tracers like multiplication, addition, logs
# but array functions like arr.append() is an ordinary Python operation that modifies a list. it isn't recorded as part of the mathematical computation
# thus, you should use python functions like print() inside jax functions

# dont use wle and if loops in jax functions
# use jax control flow operatoers
def control_flow_ex(x): 
    return jax.lax.cond(
        x > 0,  
        lambda x: x,      # if true
        lambda x: 2 * x,  # if false
        x
    )
print(control_flow_ex(10)) # 10
print(control_flow_ex(-5)) # -10

def loop_flow_ex(x): 
    i = jax.lax.fori_loop(
        0, 
        n, 
        lambda index, value: value + 1, 
        0
    )
    return x + i
print(loop_flow_ex(10, 20)) # 30 

