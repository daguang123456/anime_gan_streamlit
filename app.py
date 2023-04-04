import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import random

st.title("使用 GAN 随机生成的图像")
st.write("教程"" [link](https://www.youtube.com/watch?v=OXWvrRLzEaU&t=45s)")


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
ngpu = 0

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50 # Original is 5 on a dataset of 1 million

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Generator(ngpu).to(device)


# Load the trained model
model = torch.load("anime_gan_generator100input1.pt",map_location=torch.device('cpu') )


pp1=st.slider("p1",-5.01,5.00)
pp2=st.slider("p2",-5.01,5.00)
pp3=st.slider("p3",-5.01,5.00)
pp4=st.slider("p4",-5.01,5.00)
pp5=st.slider("p5",-5.01,5.00)
pp6=st.slider("p6",-5.01,5.00)
pp7=st.slider("p7",-5.01,5.00)
pp8=st.slider("p8",-5.01,5.00)

# fixed_noise = torch.tensor([[[pp1]],[[pp2]],[[pp3]],[[pp4]],[[pp5]],[[pp6]],[[pp7]],[[pp8]]])

# fixed_noise = torch.tensor([[[[pp8]],[[[pp1]]],[[pp2]],[[pp3]],[[pp4]],[[pp5]],[[pp6]],[[pp7]]]])

# print(torch.randn(1, 8, 1, 1))

model.eval()
bla = [pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8]
randomlist = []
for i in range(0,92):
    n = random.random()
    randomlist.append(n)
res = bla + randomlist
# print(res)

fixed_noise = torch.tensor(res).reshape(1,100,1,1)



# fixed_noise = torch.randn(1, nz, 1, 1, device=device)
print(fixed_noise)
fake = model(fixed_noise)


# fake = model(fixed_noise)
# print("fakeshape",fake.shape)
# .reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
# print(fixed_noise)
# data = fake.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
# print(data)
# img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

# white_torch = torchvision.io.read_image('white_horse.jpg')

# plt.imshow(data.detach().squeeze().cpu())
# plt.show()
# imagee = im.fromarray(data.detach().squeeze().cpu().numpy())
# st.image(data.detach().squeeze().cpu(), caption='上传了核磁共振成像。', use_column_width=True)
# plt.show()

fig1 = plt.figure(figsize=(14,8))

fig1.suptitle("随机生成的动漫脸")
# data = fakework.detach().cpu().swapaxes(0, 1)
# data = data.swapaxes(1, 2)
# # data = fake.detach().cpu()
# plt.imshow(data)   

plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))  

st.pyplot(fig1)