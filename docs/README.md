# `ipc-module`

`ipc-module` is a Python library for measuring and analyzing [information processing capacity (IPC)](https://doi.org/10.1038/srep00514), an indicator proposed in reservoir computing.
Using `pytorch` and `cupy`, it can quickly measure the IPC of various systems on GPUs.
Currently, it supports the measurement of IPC for one-dimensional inputs.
It is also possible to measure the IPC for arbitrary input time series distributions using Gram-Schmidt orthogonalization.

## Installation

```bash
pip install ipc-module
```

Alternatively, you can use various project management tools (such as `uv`, `poetry`, `pipenv`, etc.) to install it.

## Usage

In short, IPC is a metric that evaluates how much a system can process input time-series information.
It generates various linear and nonlinear transformations of input time series using orthogonal polynomials without duplication.
It evaluates how well each transformation can be reproduced from the system's current state.
For a detailed explanation of IPC and more usage examples of `ipc-module`, see **Chapters 6 and 7** and their sample code at [**RC bootcamp (URL: https://rc-bootcamp.github.io/)**](https://rc-bootcamp.github.io/), a hands-on resource for reservoir computing beginners.

### Measuring IPC of echo state network

The following example shows how to measure the IPC of an echo state network (ESN), a type of recurrent neural network.
The following cell generates a 1-input, 100-dimensional ESN and stores its response to a 100,000-step input time series generated from i.i.d. uniform random numbers in [-1, 1] as `xs`.
When you use it, replace this `xs` with the time series of the system to be measured.


```python
import numpy as np

seed = 1234
input_dim = 1
esn_dim = 100
t_washout = 1000
t_sample = 100000
esn_sr = np.linspace(0, 2.0, 21)[1:]  # Spectral radii to test.

rngs = np.random.default_rng(seed).spawn(4)
x0 = rngs[0].uniform(-1, 1, (len(esn_sr), esn_dim))
us = rngs[1].uniform(-1, 1, (t_washout + t_sample, input_dim))  # Input signal.
w_esn = rngs[2].normal(0, 1 / np.sqrt(esn_dim), (esn_dim, esn_dim))
w_esn /= np.abs(np.linalg.eigvals(w_esn)).max()  # Normalize to spectral radius 1.
w_in = rngs[3].normal(0, 1, (esn_dim, input_dim))

x = x0
xs = np.zeros((t_washout + t_sample, *x0.shape), dtype=x0.dtype)
for idx in range(t_washout + t_sample):
    xw = esn_sr[..., None] * (x[..., None, :] @ w_esn.T)[..., 0, :]
    uw = us[idx : idx + 1] @ w_in.T
    x = np.tanh(xw + (0.5 * uw + 0.5))  # Asymmetric input.
    # x = np.tanh(xw + uw)  # Symmetric input.
    xs[idx] = x

print(f"shape of us: {us.shape}")
print(f"shape of xs: {xs.shape}")
```

    shape of us: (101000, 1)
    shape of xs: (101000, 20, 100)


The response of the ESN, with spectral radius varying from 0.1 to 2.0 in 0.1 steps, is stored, so the shape of `xs` is `(100000, 20, 100)`.
In the next cell, we measure the IPC for this response `xs` all at once.
Besides `numpy`, `pytorch` and `cupy` can also be used for fast computation on a GPU.
This calculation is done using the [`profiler.UnivariateProfiler`](profiler.md#ipc_module.profiler.UnivariateProfiler) class as follows:


```python
from ipc_module.config import set_progress_bar
from ipc_module.profiler import UnivariateProfiler

set_progress_bar(False)  # Global setting to hide progress bars.

backend = "torch"
assert backend in ["numpy", "torch", "cupy"], "Invalid backend."

if backend == "torch":
    import torch

    xs_c, us_c = torch.from_numpy(xs).cuda(), torch.from_numpy(us).cuda()
elif backend == "cupy":
    import cupy

    xs_c, us_c = cupy.asarray(xs), cupy.asarray(us)
else:
    xs_c, us_c = xs, us


profiler = UnivariateProfiler(
    us_c,
    xs_c,
    poly_name="GramSchmidt",
    surrogate_num=1000,
    surrogate_seed=0,
    offset=t_washout,
    axis1=0,
    axis2=-1,
)
```

In IPC, transformations of the input time series are calculated by varying the degree, which represents the strength of nonlinearity, and the delay of the input.
The [`calc`](profiler.md#ipc_module.profiler.UnivariateProfiler.calc) method calculates the IPC by specifying the degree and the corresponding maximum delay.
The results can be saved in `npz` format using the [`save`](profiler.md#ipc_module.profiler.UnivariateViewer.save) method.


```python
degrees = [1, 2, 3, 4, 5, 6]
taus = [200, 100, 50, 20, 10, 5]
for deg, tau in zip(degrees, taus, strict=True):
    profiler.calc(deg, tau + 1, zero_offset=True)
profiler.save("./ipc_esn.npz", esn_sr=esn_sr, esn_dim=esn_dim)
```

You can load saved results with [`profiler.UnivariateViewer`](profiler.md#ipc_module.profiler.UnivariateViewer), the parent class of [`profiler.UnivariateProfiler`](profiler.md#ipc_module.profiler.UnivariateProfiler).
But since [`profiler.UnivariateViewer`](profiler.md#ipc_module.profiler.UnivariateViewer) doesn't possess `us` or `xs`, it can't calculate IPC.
Use [`profiler.UnivariateViewer.to_dataframe`](profiler.md#ipc_module.profiler.UnivariateViewer.to_dataframe) and [`helper.visualize_dataframe`](helper.md#ipc_module.helper.visualize_dataframe) to visualize IPC results like this:


```python
import matplotlib.pyplot as plt

plt.rcParams["figure.facecolor"] = (1, 1, 1, 0.5)
plt.rcParams["axes.facecolor"] = (1, 1, 1, 0.5)

from ipc_module.helper import visualize_dataframe
from ipc_module.profiler import UnivariateViewer

viewer = UnivariateViewer("./ipc_esn.npz")
esn_sr, esn_dim = viewer.info["esn_sr"], viewer.info["esn_dim"]
df, rank = viewer.to_dataframe(max_scale=1.0, truncate_by_rank=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
visualize_dataframe(
    ax,
    df,
    xticks=esn_sr,
    threshold=0.5,
    cmap="tab10",
    group_by="component",  # NOTE: Either "degree", "component", or "detail" is available.
    fontsize=12,
)
ax.plot(esn_sr, rank, lw=1.0, color="#333333", marker="o", label="Rank")
ax.legend(fontsize=12, loc="upper left", frameon=False, bbox_to_anchor=(1.0, 1.0), ncol=2)
ax.set_xlabel("ESN Spectral Radius", fontsize=14)
ax.set_title(f"IPC of {esn_dim}-dim ESN", fontsize=16)

None
```



![png](assets/README_14_0.png)



### Measuring IPC of nonlinear autoregressive moving average task

The example below shows how to measure the IPC for the NARMA10 (nonlinear autoregressive moving average of order 10) task.
The NARMA10 task is a nonlinear time-series prediction problem widely used as a benchmark in reservoir computing.


```python
def narma_func(us, y_init, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1, mu=0.25, kappa=0.25):
    assert us.shape[0] > 10
    vs = mu * us + kappa
    ys = np.zeros_like(vs)
    ys[:10] = y_init
    for idx in range(10, ys.shape[0]):
        ys[idx] += alpha * ys[idx - 1]
        ys[idx] += beta * ys[idx - 1] * np.sum(ys[idx - 10 : idx], axis=0)
        ys[idx] += gamma * vs[idx - 10] * vs[idx - 1]
        ys[idx] += delta
    return ys


seed = 1234
input_dim = 1
esn_dim = 100
t_washout = 1000
t_sample = 100000
amps = np.linspace(0, 0.25, 11)[1:]

rng = np.random.default_rng(seed)
us = rng.uniform(-1.0, 1.0, (t_sample + t_washout, 1))
ys = narma_func(
    us[..., None, :], y_init=np.zeros((amps.shape[0], 1)), mu=amps[:, None], kappa=amps[:, None]
)

print(f"shape of us: {us.shape}")
print(f"shape of ys: {ys.shape}")
```

    shape of us: (101000, 1)
    shape of ys: (101000, 10, 1)


The input time series is generated from a uniform distribution in [-1, 1].
The output time series is generated using the `narma_func` function.
The input distribution is varied by `mu` and `kappa` (`vs = mu * us + kappa`).
Typically, `mu=0.25` and `kappa=0.25` are used, but here we vary them continuously to examine their effect on IPC.


```python
backend = "torch"
assert backend in ["numpy", "torch", "cupy"], "Invalid backend."

if backend == "torch":
    import torch

    ys_c, us_c = torch.from_numpy(ys).cuda(), torch.from_numpy(us).cuda()
elif backend == "cupy":
    import cupy

    ys_c, us_c = cupy.asarray(ys), cupy.asarray(us)
else:
    ys_c, us_c = ys, us


profiler = UnivariateProfiler(
    us_c,
    ys_c,
    poly_name="GramSchmidt",
    surrogate_num=1000,
    surrogate_seed=0,
    offset=t_washout,
    axis1=0,
    axis2=-1,
)

degrees = [1, 2, 3, 4, 5, 6]
taus = [200, 100, 50, 20, 10, 5]
for deg, tau in zip(degrees, taus, strict=True):
    profiler.calc(deg, tau + 1, zero_offset=True)
profiler.save("./ipc_narma.npz", amps=amps)
```

The following cell plots the results using the same code as before.
This time, we set `group_by="detail"` to display the IPC for each degree and each delay component, which correspond to the components necessary to solve the NARMA10 task.


```python
viewer = UnivariateViewer("./ipc_narma.npz")
amps = viewer.info["amps"]
df, rank = viewer.to_dataframe(max_scale=1.0, truncate_by_rank=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
visualize_dataframe(
    ax,
    df,
    xticks=amps,
    threshold=0.01,
    cmap="tab10",
    group_by="detail",  # NOTE: Either "degree", "component", or "detail" is available.
    fontsize=12,
)
ax.plot(amps, rank, lw=1.0, color="#333333", marker="o", label="Rank")
ax.legend(fontsize=12, loc="upper left", frameon=False, bbox_to_anchor=(1.0, 1.0), ncol=1)
ax.set_xlabel(
    r"$\sigma$ (Input amplitude, i.e., $v[t] := \sigma u[t] + \sigma \sim \mathcal{U}\left([0, \sigma]\right)$)",
    fontsize=14,
)
ax.set_title("IPC of NARMA10 task", fontsize=16)

None
```



![png](assets/README_21_0.png)



## License

`ipc-module` is provided under the [MIT License](https://opensource.org/licenses/MIT).
You are free to use, modify, and redistribute it.
However, you must include the copyright notice and this permission notice in all copies or substantial portions of the software.

## Contributing

We welcome contributions to the development of `ipc-module`.

### Reporting issues

If you find any bugs or have suggestions for improvement, please report them on the [GitHub Issues page](https://github.com/rc-bootcamp/ipc-module/issues).

### Citation

If you use `ipc-module` in your research or development, please cite the following details about [RC bootcamp](https://rc-bootcamp.github.io/) and its paper.
`ipc-module` is part of RC bootcamp and was released on PyPI at the same time as its publication.

```bibtex
@article{inoue2026reservoir,
  title   = {Reservoir Computing Bootcamp---{{From Python}}/{{NumPy}} Tutorial for the Complete Beginners to Cutting-Edge Research Topics of Reservoir Computing},
  author  = {Inoue, Katsuma and Kubota, Tomoyuki and Hoan Tran, Quoc and Akashi, Nozomi and Terajima, Ryo and Kabayama, Tempei and Guan, JingChuan and Nakajima, Kohei},
  year    = 2026,
  month   = feb,
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume  = {36},
  number  = {2},
  pages   = {023109},
  issn    = {1054-1500},
  doi     = {10.1063/5.0283386}
}
```

## Contact

For questions or feedback about `ipc-module`, contact us at `k-inoue[at]isi.imi.i.u-tokyo.ac.jp`.
