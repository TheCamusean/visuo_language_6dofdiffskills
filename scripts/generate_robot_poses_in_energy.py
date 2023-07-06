import torch



l0 = .3
l1 = .3

B = 100
theta = 10.*(torch.rand(B, 2)-.5)


x1 = l0*torch.cos(theta[:,0])
y1 = l0*torch.sin(theta[:,0])
x2 = x1 + l1*torch.cos(theta[:,0] + theta[:,1])
y2 = y1 + l1*torch.sin(theta[:,0] + theta[:,1])



import matplotlib.pyplot as plt

xy_0 = torch.zeros(B,2)
xy_1 = torch.cat((x1[:,None],y1[:,None]), dim=-1)
xy_2 = torch.cat((x2[:,None],y2[:,None]), dim=-1)


def energy(xy):
    xy_tar1 = torch.Tensor([0.4, 0.0])
    d1 = 5.*torch.sin(1.*xy[:,1]).pow(2)

    xy_tar2 = torch.Tensor([0., 0.4])
    d2 = (xy - xy_tar2[None,:]).pow(2).sum(-1)

    xy_tar3 = torch.Tensor([-0.4, 0.0])
    d3 = (xy - xy_tar3[None,:]).pow(2).sum(-1)

    eps = 1e-3
    return torch.log((torch.exp(-d1/0.01)) + eps)


x = torch.linspace(-.8,.8,50)
xx, yy = torch.meshgrid((x,x))
xxyy = torch.cat((xx.reshape(-1,1), yy.reshape(-1,1)),dim=-1)
e_xy = energy(xxyy).reshape(xx.shape[0], xx.shape[1])


e = energy(xy_2).numpy()
mask = e>-1.2

trj = torch.cat((xy_0[:,None,:], xy_1[:,None,:], xy_2[:,None,:],), dim=1).numpy()

plt.pcolormesh(xx, yy, e_xy)
for k in range(B):
    if mask[k]==True:
        plt.plot(trj[k,:,0], trj[k,:,1],
                 marker = 'o')
plt.show()

