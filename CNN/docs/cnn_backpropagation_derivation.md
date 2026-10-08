# General Derivation of Backpropagation in a Convolutional Neural Network

> This document summarizes my current understanding of backpropagation through a **linear CNN architecture**, such as:
>
> `Conv -> Activation -> Pool -> Conv -> Activation -> Pool -> Flatten -> Dense`
>
> Architectures with branches or skip connections, such as ResNet, require additional gradient accumulation across multiple computational paths and are left for future work.

---

## 1. Notation

For convolution layer `l`, let:

- `l` denote the current layer.
- `f` denote an output filter / output channel.
- `c` denote an input channel.
- `i, j` denote kernel row and column indices.
- `r, s` denote output row and column indices.
- `x, y` denote input row and column indices.
- `S` denote convolution stride.
- `P` denote padding.
- `K` denote kernel side length.

Let:

$$
X_l[c,x,y]
$$

be the input to convolution layer `l`,

$$
W_l[f,c,i,j]
$$

be the kernel weight at kernel coordinate `(i,j)` for input channel `c` of filter `f`, and

$$
b_l[f]
$$

be the bias of filter `f`.

The input coordinate used by kernel position `(i,j)` while producing output position `(r,s)` is

$$
x = rS + i - P
$$

$$
y = sS + j - P
$$

If `(x,y)` lies outside the valid input region, that term is treated according to the padding rule. For zero padding, it contributes `0`.

---

## 2. Forward Convolution

The contribution of input channel `c` to filter `f` at output position `(r,s)` is

$$
Z_l[f,c,r,s]
=
\sum_i\sum_j
W_l[f,c,i,j]
X_l[c,rS+i-P,\;sS+j-P]
$$

The complete pre-activation output of filter `f` is

$$
Z_l[f,r,s]
=
b_l[f]
+
\sum_c Z_l[f,c,r,s]
$$

Equivalently,

$$
Z_l[f,r,s]
=
b_l[f]
+
\sum_c\sum_i\sum_j
W_l[f,c,i,j]
X_l[c,rS+i-P,\;sS+j-P]
$$

If an activation function is applied afterward,

$$
A_l[f,r,s]
=
\phi(Z_l[f,r,s])
$$

For ReLU,

$$
\phi(z)=\max(0,z)
$$

and

$$
\phi'(z)=
\begin{cases}
1, & z>0 \\
0, & z\le 0
\end{cases}
$$

---

## 3. Convolution Kernel Gradient

Consider one kernel weight

$$
W_l[f,c,i,j]
$$

Because convolution reuses the same kernel weight at every spatial output position, the total gradient must sum the contribution from all output positions `(r,s)` where that weight participates.

By the chain rule,

$$
\frac{\partial L}{\partial W_l[f,c,i,j]}
=
\sum_{r,s}
\frac{\partial L}{\partial Z_l[f,c,r,s]}
\frac{\partial Z_l[f,c,r,s]}
{\partial W_l[f,c,i,j]}
$$

Since

$$
Z_l[f,r,s]
=
b_l[f]+
\sum_c Z_l[f,c,r,s]
$$

we have

$$
\frac{\partial Z_l[f,r,s]}
{\partial Z_l[f,c,r,s]}
=1
$$

therefore

$$
\frac{\partial L}{\partial Z_l[f,c,r,s]}
=
\frac{\partial L}{\partial Z_l[f,r,s]}
$$

Define

$$
\delta_l[f,r,s]
=
\frac{\partial L}{\partial Z_l[f,r,s]}
$$

Then

$$
\boxed{
\frac{\partial L}{\partial W_l[f,c,i,j]}
=
\sum_{r,s}
\delta_l[f,r,s]
X_l[c,rS+i-P,\;sS+j-P]
}
$$

Define

$$
\text{grad}_l[f,c,i,j]
=
\frac{\partial L}{\partial W_l[f,c,i,j]}
$$

so

$$
\boxed{
\text{grad}_l[f,c,i,j]
=
\sum_{r,s}
\delta_l[f,r,s]
X_l[c,rS+i-P,\;sS+j-P]
}
$$

This is the convolution equivalent of the dense-layer relation

$$
\frac{\partial L}{\partial W_{n,x}}
=
\delta_n X_x
$$

except that a convolution weight is shared across many spatial positions, so its gradient is accumulated over all of them.

---

## 4. Bias Gradient

Each bias `b_l[f]` is shared across every spatial position of output filter `f`.

Since

$$
\frac{\partial Z_l[f,r,s]}
{\partial b_l[f]}
=1
$$

we obtain

$$
\boxed{
\frac{\partial L}{\partial b_l[f]}
=
\sum_{r,s}
\delta_l[f,r,s]
}
$$

---

# 5. Conv -> Activation -> Conv

Consider

```text
Conv(l)
   |
   v
Z_l
   |
   v
Activation
   |
   v
A_l
   |
   v
Conv(l+1)
```

For the current convolution layer,

$$
\delta_l[f,x,y]
=
\frac{\partial L}{\partial Z_l[f,x,y]}
$$

Applying the chain rule through the activation,

$$
\boxed{
\delta_l[f,x,y]
=
\frac{\partial L}{\partial A_l[f,x,y]}
\phi'(Z_l[f,x,y])
}
$$

For ReLU,

$$
\boxed{
\delta_l[f,x,y]
=
\frac{\partial L}{\partial A_l[f,x,y]}
ReLU'(Z_l[f,x,y])
}
$$

The activation map `A_l[f]` becomes input channel `f` of the next convolution layer:

$$
A_l[f] = X_{l+1}[f]
$$

A single activation

$$
A_l[f,x,y]
$$

may participate in many output values of layer `l+1` because:

1. every filter `g` in layer `l+1` consumes input channel `f`, and
2. sliding convolution means the same activation may fall under several kernel windows.

Therefore,

$$
\frac{\partial L}{\partial A_l[f,x,y]}
=
\sum_{g,u,v,i,j}
\delta_{l+1}[g,u,v]
W_{l+1}[g,f,i,j]
$$

but **only for terms satisfying**

$$
x = uS + i - P
$$

$$
y = vS + j - P
$$

where `S` and `P` are the stride and padding of layer `l+1`.

Thus,

$$
\boxed{
\delta_l[f,x,y]
=
\left(
\sum_{
\substack{g,u,v,i,j \\
 x=uS+i-P \\
 y=vS+j-P}
}
\delta_{l+1}[g,u,v]
W_{l+1}[g,f,i,j]
\right)
\phi'(Z_l[f,x,y])
}
$$

In implementation, it is usually easier not to solve the coordinate equations backward. Instead, during the backward pass of layer `l+1`, iterate over its output positions and kernel coordinates exactly as in the forward pass and accumulate

```text
dInput[input_index] += delta_next[output_index] * kernel[kernel_index]
```

The resulting `dInput` of layer `l+1` is exactly

$$
\frac{\partial L}{\partial A_l}
$$

because

$$
X_{l+1}=A_l
$$

---

# 6. Conv -> ReLU -> Pool

Consider

```text
Conv(l)
   |
   v
Z_l
   |
   v
ReLU
   |
   v
A_l
   |
   v
Pool
   |
   v
P_l
```

The convolution delta is still

$$
\boxed{
\delta_l[f,x,y]
=
\frac{\partial L}{\partial A_l[f,x,y]}
ReLU'(Z_l[f,x,y])
}
$$

The difference is that

$$
\frac{\partial L}{\partial A_l}
$$

must now be obtained by backpropagating through the pooling operation.

---

## 6.1 Max-Pooling Backward

Suppose pooling output

$$
P_l[f,u,v]
$$

is the maximum value inside its pooling window.

For an input activation `A_l[f,x,y]`,

$$
\frac{\partial P_l[f,u,v]}
{\partial A_l[f,x,y]}
=
\begin{cases}
1, & A_l[f,x,y] \text{ was the selected maximum} \\
0, & \text{otherwise}
\end{cases}
$$

Therefore,

$$
\boxed{
\frac{\partial L}{\partial A_l[f,x,y]}
=
\sum_{u,v}
\frac{\partial L}{\partial P_l[f,u,v]}
\mathbf{1}
\left[(x,y)=\operatorname{argmax}(u,v)\right]
}
$$

The sum is necessary because pooling windows may overlap.

For the common non-overlapping case, such as `2x2` max pooling with stride `2`, each input activation belongs to at most one pooling window.

A practical implementation should save the argmax input index during the forward pass:

```text
max_indices[pool_output_index] = selected_input_index
```

Then backward becomes conceptually:

```text
dInput[max_indices[out]] += dOutput[out]
```

---

## 6.2 Average-Pooling Backward

For a `K x K` average-pooling window,

$$
P_l[f,u,v]
=
\frac{1}{K^2}
\sum_{(x,y)\in window(u,v)}
A_l[f,x,y]
$$

Therefore,

$$
\frac{\partial P_l[f,u,v]}
{\partial A_l[f,x,y]}
=
\frac{1}{K^2}
$$

for every activation inside that pooling window.

Thus,

$$
\boxed{
\frac{\partial L}{\partial A_l[f,x,y]}
=
\sum_{
\substack{u,v \\
(x,y)\in window(u,v)}
}
\frac{1}{K^2}
\frac{\partial L}{\partial P_l[f,u,v]}
}
$$

For a `2x2` average-pooling window,

$$
\frac{1}{K^2}=\frac14
$$

If pooling windows overlap, one input activation may receive gradient contributions from multiple pooling outputs.

---

# 7. Pool -> Conv

Suppose

```text
Pool
   |
   v
P_l
   |
   v
Conv(l+1)
```

The pooling output becomes the next convolution input:

$$
P_l = X_{l+1}
$$

Therefore,

$$
\boxed{
\frac{\partial L}{\partial P_l}
=
\frac{\partial L}{\partial X_{l+1}}
}
$$

The backward pass of convolution layer `l+1` naturally computes

$$
\frac{\partial L}{\partial X_{l+1}}
$$

using

$$
\boxed{
dX_{l+1}[f,x,y]
=
\sum
\delta_{l+1}[g,u,v]
W_{l+1}[g,f,i,j]
}
$$

for all paths satisfying

$$
x=uS+i-P
$$

$$
y=vS+j-P
$$

This gradient is then passed into the pooling layer's backward operation.

---

# 8. Final CNN Layer -> Flatten -> Dense

Suppose the last convolutional/pooling output has shape

```text
H x W x C
```

Flattening converts it into a vector of length

$$
HWC
$$

without changing the values.

If `x` denotes one element of this flattened vector, and the first dense layer contains neurons indexed by `n`, then

$$
Z_{dense}[n]
=
\sum_x W_{dense}[n,x]X_{flat}[x]
+b_{dense}[n]
$$

If

$$
\delta_{dense}[n]
=
\frac{\partial L}{\partial Z_{dense}[n]}
$$

then the gradient with respect to flattened input element `x` is

$$
\boxed{
\frac{\partial L}{\partial X_{flat}[x]}
=
\sum_n
\delta_{dense}[n]
W_{dense}[n,x]
}
$$

The flatten operation itself has no learnable parameters. During backward propagation, this gradient vector is simply reshaped back into the original convolutional tensor layout.

For example,

```text
H x W x C
   |
   | flatten
   v
HWC
```

becomes during backward

```text
d(HWC)
   |
   | reshape
   v
d(H x W x C)
```

---

# 9. General Layerwise View of Backpropagation

A useful way to design the implementation is:

> Every operation receives the gradient of the loss with respect to its output and returns the gradient of the loss with respect to its input.

In symbols,

$$
\boxed{
\frac{\partial L}{\partial output}
\longrightarrow
\text{Layer Backward}
\longrightarrow
\frac{\partial L}{\partial input}
}
$$

For trainable layers such as convolution and dense layers, backward propagation additionally computes parameter gradients.

### Convolution layer

```text
dOutput
   |
   v
Activation derivative (if activation is fused)
   |
   v
delta = dL/dZ
   |
   +--> dKernel
   |
   +--> dBias
   |
   v
dInput
```

### Pooling layer

```text
dOutput
   |
   v
Pool backward
   |
   v
dInput
```

### Dense layer

```text
dOutput
   |
   v
Activation derivative
   |
   v
delta
   |
   +--> dWeights
   |
   +--> dBias
   |
   v
dInput
```

---

# 10. Summary of Important Equations

## Convolution kernel gradient

$$
\boxed{
\frac{\partial L}{\partial W_l[f,c,i,j]}
=
\sum_{r,s}
\delta_l[f,r,s]
X_l[c,rS+i-P,\;sS+j-P]
}
$$

## Convolution bias gradient

$$
\boxed{
\frac{\partial L}{\partial b_l[f]}
=
\sum_{r,s}
\delta_l[f,r,s]
}
$$

## Activation delta

$$
\boxed{
\delta_l[f,r,s]
=
\frac{\partial L}{\partial A_l[f,r,s]}
\phi'(Z_l[f,r,s])
}
$$

For ReLU,

$$
\boxed{
\delta_l[f,r,s]
=
\frac{\partial L}{\partial A_l[f,r,s]}
ReLU'(Z_l[f,r,s])
}
$$

## Gradient passed through the next convolution

$$
\boxed{
\frac{\partial L}{\partial X_{l+1}[f,x,y]}
=
\sum_{
\substack{g,u,v,i,j \\
 x=uS+i-P \\
 y=vS+j-P}
}
\delta_{l+1}[g,u,v]
W_{l+1}[g,f,i,j]
}
$$

## Max-pooling backward

$$
\boxed{
\frac{\partial L}{\partial A_l[f,x,y]}
=
\sum_{u,v}
\frac{\partial L}{\partial P_l[f,u,v]}
\mathbf{1}
\left[(x,y)=\operatorname{argmax}(u,v)\right]
}
$$

## Average-pooling backward

$$
\boxed{
\frac{\partial L}{\partial A_l[f,x,y]}
=
\sum_{
\substack{u,v \\
(x,y)\in window(u,v)}
}
\frac{1}{K^2}
\frac{\partial L}{\partial P_l[f,u,v]}
}
$$

## Dense gradient back into flattened CNN output

$$
\boxed{
\frac{\partial L}{\partial X_{flat}[x]}
=
\sum_n
\delta_{dense}[n]
W_{dense}[n,x]
}
$$

---

# 11. Important Implementation Notes

1. **`S` means stride, not kernel size.** Use `K` for kernel side length.
2. Kernel gradients must accumulate over all spatial positions because convolution shares weights.
3. Input gradients must also accumulate because one input activation may affect multiple outputs and multiple filters.
4. Max pooling should usually cache the argmax index during the forward pass.
5. Average pooling distributes the upstream gradient evenly across the pooling window.
6. Flattening does not modify gradient values; it only changes the tensor shape/index interpretation.
7. Bias gradients are the sum of deltas over all spatial positions for each output filter.
8. For a linear architecture, backpropagation can be implemented by traversing operations in reverse order and repeatedly passing `dOutput -> dInput`.
9. Architectures with branches or skip connections, such as ResNet, require summing gradient contributions from every outgoing computational path before continuing backward.

---

# 12. Relation to Dense-Layer Backpropagation

The CNN equations are the same chain-rule ideas used in a dense neural network.

For a dense layer,

$$
\frac{\partial L}{\partial W[n,x]}
=
\delta[n]X[x]
$$

For convolution,

$$
\frac{\partial L}{\partial W[f,c,i,j]}
=
\sum_{r,s}
\delta[f,r,s]
X[c,rS+i-P,sS+j-P]
$$

The key difference is **weight sharing**: a dense weight is normally used once per sample, while a convolution kernel weight is reused across many spatial positions, so all of those gradient contributions must be summed.
