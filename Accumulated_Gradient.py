## Gradient Accumulate - fitting a line
## Code By Huankang Guan
## --------------------------------- ##

import torch
tot = 10000
## *100 +100 will degrade the performance since the distribution \
##  of training data is not sensitive for model's parameters(Same to BN layers? It might not!)
x = torch.rand((tot*10,))
k = 2.5; b = 3.2
y = k*x+b
K = torch.rand(1, requires_grad=True)
B = torch.rand(1, requires_grad=True)

## Giva a fixed value to K
# K.data.add_(2.7090-K.item())
# print(K, K.grad_fn)

beta = 0.01
for i in range(int(tot)):
  # y_ = K*x[i*10:i*10+10]+B
  # loss = (y_-y[i*10:i*10+10]).abs().mean()
  y_ = K*x[i:i+1]+B
  loss = (y_-y[i:i+1]).abs().mean()/10.0
  loss.backward()
  if (i+1)%10==0:
    K.data.sub_(K.grad.data*beta)
    B.data.sub_(B.grad.data*beta)
    K.grad.zero_()
    B.grad.zero_()

print(K, B)
## output // you may get a diffirent result but K,B will equal to k,b approximatelly.
## tensor([2.5066], requires_grad=True) tensor([3.2047], requires_grad=True)

