#%% IMPORT LIBRARIES
# we want to recreate the results of the Rao & Ballard paper

# import libraries
from PredictiveNetwork import PredictiveCodingNetwork
import scipy.io
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import torch 
from fancy_einsum import einsum

# %% LOAD DATA
# first let's load the data
IMAGE_FOLDER = './RaoBallardImages/'

# load the .mat file in the folder, shape is (512, 512, 10) which is (height, width, number of images)
images = scipy.io.loadmat(IMAGE_FOLDER + 'IMAGES_RAW.mat')['IMAGESr']

# %% LOOK AT DATA
# let's look at the images, display them in a 2x5 grid
fig, ax = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(images[:, :, i*5+j], cmap='gray')
        ax[i, j].axis('off')
plt.show()

# %% FILTERS
# now we will do some filtering
# a difference of guasians filter
def dog_filter(image, sigma1, sigma2):
    # calculate the difference of gaussian
    dog = scipy.ndimage.gaussian_filter(image, sigma1) - \
            scipy.ndimage.gaussian_filter(image, sigma2)
    return dog

def GaussianMask(sizex=16, sizey=16, sigma=5):
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    x0 = sizex // 2
    y0 = sizey // 2
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*(sigma**2)))
    return mask / np.sum(mask)

# apply the filter to the images, and call them filt_imgs
filt_imgs = np.zeros(images.shape)
for i in range(images.shape[2]):
    filt_imgs[:, :, i] = dog_filter(images[:, :, i], 1.3, 1.6)

# plot the filtered images
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(filt_imgs[:, :, i*5+j], cmap='gray')
        ax[i, j].axis('off')

# %% FIT THE MODEL

# change images to be (batch_size, channels, height, width)
input_images = images[:, :, 0:10].astype(np.float32)
input_images = torch.from_numpy(input_images)
input_images = input_images.permute(2, 0, 1).unsqueeze(1)
input_size = input_images.shape[1:]
# normalize input images to be mean zero and std=1, do so independently for each image
for i in range(input_images.shape[0]):
    input_images[i] = (input_images[i] - input_images[i].mean()) / input_images[i].std()

model = PredictiveCodingNetwork(input_size=input_size, n_layers=2, n_causes=[32, 24],
                                kernel_size=[(16,16), (1,3)], stride=[(5,5),(1,3)], padding=0,
                                lam=1.0, alpha=[10.0, 1.00], k1=.2, k2=.2, sigma2=[.5, 1.])

# run the model for 500 timesteps
# do 1 image at a time, for 50 timesteps each

"""for i in range(3):
    # get the ith image
    image = input_images[i]
    # reshape to be (batch_size, channels, height, width)
    image = image.unsqueeze(0)
    # run the model
    _ = model(image, timesteps=100, train_U=True)
    # multiply k2 by 0.9 in each layer
    for j in range(len(model.layers)):
        model.layers[j].k2 *= 1.0"""

# train the model wit a batch of thelast 5 images
# run the model for 500 timesteps
ims = input_images[5:10]
# run the model
_ = model(ims, timesteps=500, train_U=True)


#%%
# print the shape of U and r for each level descriptively
for i in range(len(model.layers)):
    print('U{}: {}'.format(i, model.layers[i].U.shape))
    print('r{}: {}'.format(i, model.layers[i].r.shape))

# print the mean and std of the U and r for each level
for i in range(len(model.layers)):
    print('U{}: mean: {}, std: {}'.format(i, model.layers[i].U.mean(), model.layers[i].U.std()))
    print('r{}: mean: {}, std: {}'.format(i, model.layers[i].r.mean(), model.layers[i].r.std()))

# print the mean and std of the images
print('input mean: {}, std: {}'.format(input_images.mean(), input_images.std()))
# %%
# now let's look at the U and r for each level
# first let's look at the U, print out shape of U
print(model.layers[0].U.shape)
# plot the 32 U's for the first level
fig, ax = plt.subplots(4, 8, figsize=(10, 5))
for i in range(4):
    for j in range(8):
        to_plot = model.layers[0].U[i*8+j]
        # convert to numpy array
        to_plot = to_plot.detach().numpy()
        # reshape to be (height, width)
        to_plot = to_plot.reshape(16, 16)
        # plot
        ax[i, j].imshow(to_plot, cmap='gray')
        ax[i, j].axis('off')

# make a title for the whole figure
fig.suptitle('U for level 0', fontsize=16)

# %%

# get two images
x = input_images[0:1]
# reshape x to be (batch_size, channels, height, width)
_ = model(x, timesteps=300, train_U=True)

# get the final value of r in the first layer
r0 = model.layers[0].r.detach().numpy()
# shape of r is (batch_size, n_causes, n_patches_h, n_patches_w)
print(r0.shape)
# get the final value of U in the first layer
U0 = model.layers[0].U.detach().numpy()
# shape of U is (n_causes, channels, kernel_height, kernel_width)
print(U0.shape)

# combine U0 and r0 to get the final value of the first layer
# shape of x_hat is (batch_size, channels, height, width)
prediction = einsum('causes chan kernh kernw, batch causes npatchesh npatchesw -> batch chan kernh kernw npatchesh npatchesw', U0, r0)
print(prediction.shape) # (batch_size, channels, height, width, n_patches_h, n_patches_w)
# fold this back up to be an image
# get the unfold from the first layer
unfold = model.layers[0].unfold
# get the unfold params
unfold_params = unfold.parameters()
# PRINT THE PARAMS
# get stride of the unfold
stride = unfold.stride

# get the height and width of the image
output_size = (x.shape[2], x.shape[3])
fold = torch.nn.Fold(output_size, kernel_size=unfold.kernel_size, stride=stride)
# input_ones is a torch.ones with size (batch_size, channels, height, width)
input_ones = torch.ones(x.shape) # x is shape (batch_size, channels, height, width)
prediction = prediction.reshape(prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3], prediction.shape[4]*prediction.shape[5])
# rewrite the previous line
# fold the prediction, conver to torch
prediction = fold(torch.from_numpy(prediction))
# (batch_size, channels, height, width)
# we have to divide the prediction because of fold
divisor = fold(unfold(input_ones))
prediction = prediction / divisor

# %%
# look at the prediction in other subplot
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 2)
plt.imshow(np.transpose(prediction[0].detach().numpy(), (1, 2, 0)), cmap='gray')
plt.axis('off')

# put a title that is the label
# put a title on the left for the original image
plt.subplot(1, 3, 1)
plt.title('True Image')
plt.imshow(np.transpose(x[0].detach().numpy(), (1, 2, 0)), cmap='gray')
plt.axis('off')

# put a title on the right for the prediction
plt.subplot(1, 3, 2)
plt.title('Prediction')
# plot the diff
plt.subplot(1, 3, 3)
plt.imshow(np.transpose((prediction[0] - x[0]).detach().numpy(), (1, 2, 0)), cmap='gray')
# turn off all axes
plt.axis('off')
plt.title('Difference')
# %% Look at the second layer filters
# get the second layer filters
U1 = model.layers[1].U.detach().numpy()
U0 = model.layers[0].U.detach().numpy()
# shape of U is (n_causes, channels, kernel_height, kernel_width)
print(U0.shape, U1.shape)
# plot the 32 U's for the first level
# combine U0 and U1 with einsum, keeping in mind that
# the 0th dim of U0 is the same as the 1st dim of U1
U = np.einsum('ijkl, jxyz -> iklxyz', U1, U0)
print(U.shape)
# plot the 32 U's for the first level
# plot U[0,0,:,:,0,0], U[0,0,:,:,0,1], U[0,0,:,:,0,2] in 3 columns
fig, ax = plt.subplots(1,3, figsize=(10, 5))
for i in range(3):
    to_plot = U[,0,i,0,:,:]
    # convert to numpy array
    to_plot = to_plot
    # reshape to be (height, width)
    to_plot = to_plot.reshape(16, 16)
    # plot
    ax[i].imshow(to_plot)
    ax[i].axis('off')
# %%
U0 = model.layers[0].U.detach().numpy()
U1 = model.layers[1].U.detach().numpy()

# reshape U0 so that the first axes is the last
U0 = np.moveaxis(U0, 0, -1)
# and swap the first two dimensions of U1
U1 = np.swapaxes(U1, 0, 1)
print(U0.shape, U1.shape)
# for each of the last axes in U1, combine with U0
for i in range(U1.shape[3]):
    U1_ = U1[:,:,:,i]  # (128,32,1)
    print(U0.shape, U1_.shape)
    # now dot product with U0
    for j in range(U0.shape[3]):
        U0
    # print the shape of U
    print(U.shape)
    

# plot the first 32 U's
fig, ax = plt.subplots(4,5, figsize=(10, 5))
for i in range(4):
    to_plot = U[0,:,:,i,:]
    plt.imshow(to_plot)
    plt.axis('off')

# %%
