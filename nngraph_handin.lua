-- require packages
require 'nn'
require 'nngraph'
-- set seed to generate same random parameter
torch.manualSeed(46)
-- test sample
x1=torch.Tensor{2,3,6,7,3}
x2=torch.Tensor{6,5,4,6,3}
x3=torch.Tensor{9,3,6,5,2}
-- linear(x3) 
m1=nn.Linear(5,5)()
-- x2*linear(x3) 
p2=nn.Identity()()
m2=nn.CMulTable()({p2,m1})
-- x1+x2*linear(x3)
p1=nn.Identity()()
m3=nn.CAddTable()({m2,p1})
-- wrap graph node into module
model1=nn.gModule({p1,p2,m1},{m3})
-- generate output
out1=model1:forward({x1,x2,x3})
-- generate validation output
torch.manualSeed(46)
out2=nn.CAddTable():forward({nn.CMulTable():forward({x2,nn.Linear(5,5):forward(x3)}),x1})
-- print output
print('nngraph output')
print (out1)
print('nn output')
print (out2)