import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

# class SphereProduct(nn.Module):
#     def __init__(self, in_features, out_features, m=4):
#         super(SphereProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.m = m
#         self.base = 1000.0
#         self.gamma = 0.12
#         self.power = 1
#         self.LambdaMin = 5.0
#         self.iter = 0
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform(self.weight)

#         # duplication formula
#         # 将x\in[-1,1]范围的重复index次映射到y\[-1,1]上
#         self.mlambda = [
#             lambda x: x ** 0,
#             lambda x: x ** 1,
#             lambda x: 2 * x ** 2 - 1,
#             lambda x: 4 * x ** 3 - 3 * x,
#             lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
#             lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]
#     def forward(self, input, label):
#         # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
#         self.iter += 1
#         self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
#         cos_theta = cos_theta.clamp(-1, 1)
#         cos_m_theta = self.mlambda[self.m](cos_theta)
#         theta = cos_theta.data.acos()
#         k = (self.m * theta / 3.14159265).floor()
#         phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
#         NormOfFeature = torch.norm(input, 2, 1)
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cos_theta.size())
#         one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
#         one_hot.scatter_(1, label.view(-1, 1), 1)
#         # --------------------------- Calculate output ---------------------------
#         output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
#         output *= NormOfFeature.view(-1, 1)
#         return output
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) \
#                + ', m=' + str(self.m) + ')'

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x]
    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features
        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum
        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)
        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)
        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data.clone() * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.bool()#index.byte()
        index = Variable(index).clone()

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = Variable(cos_theta,requires_grad=True).clone() * 1.0 #size=(B,Classnum)
        #output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] = output[index].clone()-(cos_theta[index].clone()*(1.0+0)/(1+self.lamb))
        #output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] = output[index].clone()+(phi_theta[index].clone()*(1.0+0)/(1+self.lamb))
        
        logpt = F.log_softmax(output.clone(), dim=0)
        logpt = logpt.gather(1,target.clone())
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp(),requires_grad=True).clone()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class sphere20a(nn.Module):
    def __init__(self,classnum=10574,feature=False,feat_dim=512):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64,)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*9*10,512)#(512*7*6,512)，7=112/16,6=96/16 ，(160,144)-[29,80][159,80][96,157]
        self.fc6 = nn.Linear(512, feat_dim)
        self.fc7 = AngleLinear(feat_dim, self.classnum)
        


    def forward(self, x ):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc5(x)
       
        z = x
        if self.feature: return z
        
        #feat,train_loss = self.fc777 = AngleSoftmax
        
        #y = x.clone()
        #y = self.fc_features(y)
        #x = self.fc6(x)
        x = self.fc6(x)
        #xy = self.fc6_7(x)
        y = self.fc7(x)
        
        return x,y
        
        #return z
        #return y,x 
        #return y,xy,z


class AngleSoftmax(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, m=4, lambda_max=1000.0, lambda_min=5.0, 
                 power=1.0, gamma=0.1, loss_weight=1.0):
        '''
        :param input_size: Input channel size.
        :param output_size: Number of Class.
        :param normalize: Whether do weight normalization.
        :param m: An integer, specifying the margin type, take value of [0,1,2,3,4,5].
        :param lambda_max: Starting value for lambda.
        :param lambda_min: Minimum value for lambda.
        :param power: Decreasing strategy for lambda.
        :param gamma: Decreasing strategy for lambda.
        :param loss_weight: Loss weight for this loss. 
        '''
        super(AngleSoftmax, self).__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(int(output_size), input_size))
        nn.init.kaiming_uniform_(self.weight, 1.0)
        self.m = m

        self.it = 0
        self.LambdaMin = lambda_min
        self.LambdaMax = lambda_max
        self.gamma = gamma
        self.power = power

    def forward(self, x, y):

        if self.normalize:
            wl = self.weight.pow(2).sum(1).pow(0.5)
            wn = self.weight / wl.view(-1, 1)
            self.weight.data.copy_(wn.data)
        if self.training:
            lamb = max(self.LambdaMin, self.LambdaMax / (1 + self.gamma * self.it)**self.power)
            self.it += 1
            phi_kernel = PhiKernel(self.m, lamb)
            feat = phi_kernel(x, self.weight, y)
            loss = F.nll_loss(F.log_softmax(feat, dim=1), y)
        else:
            feat = x.mm(self.weight.t())
            loss = F.nll_loss(F.log_softmax(feat, dim=1), y)
        
        return feat,loss.mul_(self.loss_weight)


