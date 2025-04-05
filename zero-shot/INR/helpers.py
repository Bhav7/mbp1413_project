import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torch.autograd import Variable

from PIL import Image, ImageFilter
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
os.environ['MPLCONFIGDIR'] = '/home/zachvav/scratch/matplotlib_tmp'
#disable matplotlib to screen
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage.metrics import peak_signal_noise_ratio

import time

def calc_psnr(img1, img2, normalize=False):
    if normalize:
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    mse = np.mean((img1 - img2)**2)
    return 10 * np.log10(1/mse)

def compute_TV(img):
    #compute total variation of image
    tv = 0
    tv += torch.sum(torch.abs(img[:, :-1] - img[:, 1:]))
    tv += torch.sum(torch.abs(img[:-1, :] - img[1:, :]))
    return tv

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def gradient_entropy(y, shape=(320,320)):
    grad = calc_gradient(y, shape=shape)
    grad = grad.norm(dim=-1) #combine x and y gradients into a single tensor

    entropy = -torch.sum(grad * torch.log2(grad + 1e-10), dim=-1)

    return entropy

def blur(y, shape=(320,320), kernel_size=3):
    #reshape y
    y = y.view(1, 1, shape[0], shape[1])

    #pass through blur convlution first with kernel size
    blur_weights = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2
    conv0=nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding='same', bias=False)
    conv0.weight=nn.Parameter(torch.from_numpy(blur_weights).float().unsqueeze(0).unsqueeze(0))

    y = conv0(y).squeeze(0)

    return y

def calc_gradient(y, shape=(320,320), blur = False, blur_kernel=3):

    # print(y.shape)

    #reshape y
    y = y.view(1, 1, shape[0], shape[1])

    if blur:
        #pass through blur convlution first with kernel size
        blur_weightse = np.ones((blur_kernel, blur_kernel), dtype=np.float32) / blur_kernel**2
        conv0=nn.Conv2d(1, 1, kernel_size=blur_kernel, stride=1, padding='same', bias=False)
        conv0.weight=nn.Parameter(torch.from_numpy(blur_weightse).float().unsqueeze(0).unsqueeze(0))
        conv0.cuda()

        y = conv0(y)

    #pass through convolution to compute gradient
    a = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    conv1.cuda()

    G_x=conv1(y)

    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    conv2.cuda()

    G_y=conv2(y)

    # print(G_x.shape)
    # print(G_y.shape)

    G_x = G_x.view(-1)
    G_y = G_y.view(-1)

    # print(G_x.shape)
    # print(G_y.shape)

    #combine G_x and G_y into a single tensor
    G = torch.stack((G_x, G_y), dim=-1)
    G = G.unsqueeze(0)

    #threshold the gradient to remove noise
    # G = torch.where(G > 0.1, G, torch.zeros_like(G))

    # print(G.shape)

    # quit()

    return G

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())    

    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)

    print(img.shape)
    print(img.dtype)
    print(img.min())
    print(img.max())
    print(img.mean())
    print(img.std())

    return img

def get_mri_tensor(sidelength, img_path='mri.png'):
    img = Image.open(img_path)
    #conver to greyscale
    img = img.convert('L')

    #crop top 25% and bottom 25% of image
    img = img.crop((0, int(160), img.size[0], int(480)))

    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)

    #normalize to 0-1
    img = (img - img.min()) / (img.max() - img.min())

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min: {img.min()}")
    print(f"Image max: {img.max()}")
    print(f"Image mean: {img.mean()}")
    print(f"Image std: {img.std()}")

    return img


class ImageFitting(Dataset):
    def __init__(self, sidelength, noise_std=0.1, img_path='mri.png'):
        super().__init__()
        # img = get_cameraman_tensor(sidelength)
        img = get_mri_tensor(sidelength, img_path=img_path)

        #set random seed for reproducibility
        torch.manual_seed(0)

        #add gaussian noise to image
        noise = torch.randn_like(img) * noise_std
        img = img + noise

        # #blur image
        # with torch.no_grad():
        #     img = blur(img, shape=(sidelength, sidelength), kernel_size=9)

        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        # self.pixels = blur(self.pixels, shape=(sidelength, sidelength), kernel_size=3)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

#==================================================================

#loop over test dir
for scan in os.listdir('test'):
    for file in os.listdir(os.path.join('test', scan)):
        if 'axial' in file and file.endswith('.png'):

            out_dir = os.path.join('out', scan)
            os.makedirs(out_dir, exist_ok=True)

            #if out_dir/training.gif exists, skip this file
            # if os.path.exists(os.path.join(out_dir, 'training.gif')):
            #     print("Skipping %s/%s" % (scan, file))
            #     continue
            
            img_path = os.path.join('test', scan, file)

            print(scan)

            sidelength = 320
            NOISE_STD = 0.03
            total_steps = 3000
            steps_til_summary = 10
            # TV_weight = 1e-7
            TV_weight = 0
            grad_weight = 0
            omega = 30.0
            exp_weight=0.99

            clean_img = get_mri_tensor(sidelength, img_path=img_path).cuda()
            with torch.no_grad():
                clean_img_gradient = calc_gradient(clean_img, shape=(sidelength, sidelength), blur=False)

            # clean_img_gradient = np.gradient(clean_img.cpu().view(sidelength,sidelength).detach().numpy())
            # clean_img_gradient = torch.tensor(clean_img_gradient, dtype=torch.float32).cuda()
            # clean_img_gradient = clean_img_gradient.permute(1, 2, 0).view(-1, 2)
            # clean_img_gradient = clean_img_gradient.unsqueeze(0)
            #normalize the gradient between 0 and 1
            # clean_img_gradient = clean_img_gradient - clean_img_gradient.min()
            # clean_img_gradient = clean_img_gradient / clean_img_gradient.max()

            cameraman = ImageFitting(sidelength, noise_std=NOISE_STD, img_path=img_path)
            dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

            torch.manual_seed(5)

            noised_img = cameraman.pixels.cuda()
            noise_img_range = noised_img.cpu().view(sidelength,sidelength).detach().numpy().max() - noised_img.cpu().view(sidelength,sidelength).detach().numpy().min()
            with torch.no_grad():
                noised_img_gradient = calc_gradient(noised_img, shape=(sidelength, sidelength), blur=False)

            #save noised image to file
            noised_img_PIL = noised_img.cpu().view(sidelength,sidelength).detach().numpy()
            noised_img_PIL = (noised_img_PIL - noised_img_PIL.min()) / (noised_img_PIL.max() - noised_img_PIL.min())
            noised_img_PIL = (noised_img_PIL * 255).astype(np.uint8)
            noised_img_PIL = Image.fromarray(noised_img_PIL)
            noised_img_PIL = noised_img_PIL.convert('RGB')
            noised_img_PIL.save(os.path.join(out_dir, "noised_img.png"))

            # noised_img_gradient = np.gradient(noised_img)
            # noised_img_gradient = torch.tensor(noised_img_gradient, dtype=torch.float32).cuda()
            # noised_img_gradient = noised_img_gradient.permute(1, 2, 0).view(-1, 2)
            # noised_img_gradient = noised_img_gradient.unsqueeze(0)
            #normalize the gradient between 0 and 1
            # noised_img_gradient = noised_img_gradient - noised_img_gradient.min()
            # noised_img_gradient = noised_img_gradient / noised_img_gradient.max()

            img_siren = Siren(in_features=2, out_features=1, hidden_features=320, 
                            hidden_layers=3, outermost_linear=True, first_omega_0=omega, hidden_omega_0=omega)

            # #load model.pth if it exists
            # if os.path.exists('model.pth'):
            #     img_siren.load_state_dict(torch.load('model.pth'))

            img_siren.cuda()

            #print number of parameters in the model
            print("Number of parameters in the model: %d" % sum(p.numel() for p in img_siren.parameters()))

            psnr_base = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                noised_img.cpu().view(sidelength,sidelength).detach().numpy())
            print("PSNR (Base): %0.3f" % psnr_base)

            # optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

            #add selective weight decay to the last two layers of the network
            optim = torch.optim.Adam(lr=1e-4, params=[
                {'params': img_siren.net[0].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[1].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[2].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[3].parameters(), 'weight_decay': 0.0},
                {'params': img_siren.net[4].parameters(), 'weight_decay': 0.0}
            ])

            # optim = torch.optim.Adam(lr=1e-4, params=[
            #     {'params': img_siren.net[0].parameters(), 'weight_decay': 0.0},
            #     {'params': img_siren.net[1].parameters(), 'weight_decay': 0.0},
            #     {'params': img_siren.net[2].parameters(), 'weight_decay': 0.0},
            #     {'params': img_siren.net[3].parameters(), 'weight_decay': 0.0},
            #     {'params': img_siren.net[4].parameters(), 'weight_decay': 0.001},
            #     {'params': img_siren.net[5].parameters(), 'weight_decay': 0.001},
            #     {'params': img_siren.net[6].parameters(), 'weight_decay': 0.0}
            # ])


            model_input, ground_truth = next(iter(dataloader))
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

            psnr_clean_history = []
            psnr_clean_avg_history = []
            psnr_noised_history = []
            entropy_history = []

            out_avg = None

            for step in range(total_steps):
                # print(model_input.shape)
                # print(model_input[0,0:10])
                # print(ground_truth.shape)
                # print(ground_truth[0,0:10])

                # print(model_input.mean())
                # print(ground_truth.mean())
                # print(ground_truth.std())

                # #print input data_type
                # print(model_input.dtype)
                # print(ground_truth.dtype)

                model_output, coords = img_siren(model_input)

                if out_avg is None:
                    out_avg = model_output.detach()
                else:
                    out_avg = out_avg * exp_weight + model_output.detach() * (1 - exp_weight)

                #clamp model output to be between 0 and 1
                # model_output = torch.clamp(model_output, 0, 1)
                
                # img_grad = gradient(model_output, coords)
                img_grad = calc_gradient(model_output, shape=(sidelength, sidelength))
                #normalize the gradient between 0 and 1
                # img_grad = img_grad - img_grad.min()
                # img_grad = img_grad / img_grad.max()

                loss = ((model_output - ground_truth)**2).mean()
                
                loss_TV = compute_TV(model_output.view(sidelength, sidelength))     #add TV term
                loss_GRAD = ((img_grad - noised_img_gradient)**2).mean() #add gradient MSE loss

                # loss += loss_GRAD
                loss = (1 - grad_weight) * loss + grad_weight * loss_GRAD + TV_weight * loss_TV


                psnr_clean = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                                     model_output.cpu().view(sidelength,sidelength).detach().numpy())
                psnr_clean_avg = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                                         out_avg.cpu().view(sidelength,sidelength).numpy())
                psnr_noised = peak_signal_noise_ratio(noised_img.cpu().view(sidelength,sidelength).detach().numpy(),
                                                      model_output.cpu().view(sidelength,sidelength).detach().numpy(), data_range=noise_img_range)
                entropy = gradient_entropy(model_output, shape=(sidelength, sidelength))
                
                psnr_clean_history.append(psnr_clean)
                psnr_clean_avg_history.append(psnr_clean_avg)
                psnr_noised_history.append(psnr_noised)
                entropy_history.append(entropy.item())
                
                if not step % steps_til_summary:
                    # img_laplacian = laplace(model_output, coords)

                    print("Step %d, Total loss %0.6f loss (GRAD) %0.6f PSNR (Clean) %0.3f PSNR SM (Clean) %0.3f PSNR (Noised) %0.3f Gradient Entropy %0.5f" % (step, loss.item(), loss_GRAD.item(), psnr_clean, psnr_clean_avg, psnr_noised, entropy.item()))

                    fig, axes = plt.subplots(2,4, figsize=(24,12))

                    #set plot title
                    fig.suptitle("Step %d    Noised → Clean PSNR %.2f" % (step, psnr_base), fontsize=16)

                    #set axes titles
                    axes[0,0].set_title("PSNR History")
                    axes[0,1].set_title("Model Output")
                    axes[0,2].set_title("Output → Clean PSNR: %.2f" % psnr_clean)
                    axes[0,3].set_title("Output → Noised PSNR: %.2f" % psnr_noised)
                    axes[1,0].set_title("Gradient Entropy History")
                    axes[1,1].set_title("Output Gradient")
                    axes[1,2].set_title("Clean Gradient")
                    axes[1,3].set_title("Noised Gradient")

                    #set space between subplots
                    plt.subplots_adjust(wspace=0.05, hspace=0.1)

                    vmin = min(model_output.min(), clean_img.min(), noised_img.min()).item()
                    vmax = max(model_output.max(), clean_img.max(), noised_img.max()).item()
                    vmin_grad = min(img_grad.min(), clean_img_gradient.min(), noised_img_gradient.min()).item()
                    vmax_grad = max(img_grad.max(), clean_img_gradient.max(), noised_img_gradient.max()).item()

                    #plot PSNR histroy in subplot 0,0
                    axes[0,0].plot(psnr_clean_history, label='PSNR (Clean)', color='orange')
                    axes[0,0].plot(psnr_noised_history, label='PSNR (Noised)', color='blue')
                    # axes[0,0].plot(psnr_clean_avg_history, label='PSNR (Clean) + SM', color='green')
                    axes[0,0].set_xlabel('Step')
                    axes[0,0].set_ylabel('PSNR')
                    axes[0,0].set_title('PSNR History')
                    axes[0,0].legend()

                    #add dashed horizontal line to show PSNR of base image
                    # axes[0,0].axhline(y=psnr_base, color='r', linestyle='--', label='Base PSNR')

                    #plot gradient entropy history in subplot 1,0
                    axes[1,0].plot(entropy_history, label='Gradient Entropy', color='green')
                    axes[1,0].set_xlabel('Step')
                    axes[1,0].set_ylabel('Entropy')
                    axes[1,0].set_title('Gradient Entropy History')
                    axes[1,0].legend()

                    axes[0,1].imshow(model_output.cpu().view(sidelength,sidelength).detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0,2].imshow(clean_img.cpu().view(sidelength,sidelength).detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0,3].imshow(noised_img.cpu().view(sidelength,sidelength).detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
                    axes[1,1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy(), cmap='hot', vmin=vmin_grad, vmax=vmax_grad)
                    axes[1,2].imshow(clean_img_gradient.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy(), cmap='hot', vmin=vmin_grad, vmax=vmax_grad)
                    axes[1,3].imshow(noised_img_gradient.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy(), cmap='hot', vmin=vmin_grad, vmax=vmax_grad)

                    #turn off axis for all subplots
                    axes[0,1].axis('off')
                    axes[0,2].axis('off')
                    axes[0,3].axis('off')
                    axes[1,1].axis('off')
                    axes[1,2].axis('off')
                    axes[1,3].axis('off')

                    # axes[2].imshow(img_laplacian.cpu().view(256,256).detach().numpy())
                    #save plot
                    plt.savefig(os.path.join(out_dir, "%04d.png" % step))
                    plt.close(fig)

                    #save model output in single PNG file
                    model_out_img = model_output.cpu().view(sidelength,sidelength).detach().numpy()
                    model_out_img = (model_out_img - model_out_img.min()) / (model_out_img.max() - model_out_img.min())
                    model_out_img = (model_out_img * 255).astype(np.uint8)
                    model_out_img = Image.fromarray(model_out_img)
                    model_out_img = model_out_img.convert('RGB')
                    model_out_img.save(os.path.join(out_dir, "%04d_output.png" % step))

                    #save model output to numpy file
                    model_out_np = model_output.cpu().view(sidelength,sidelength).detach().numpy()
                    np.save(os.path.join(out_dir, "%04d_output.npy" % step), model_out_np)

                    #save smoothed model output in single PNG file
                    model_out_avg_img = out_avg.cpu().view(sidelength,sidelength).detach().numpy()
                    model_out_avg_img = (model_out_avg_img - model_out_avg_img.min()) / (model_out_avg_img.max() - model_out_avg_img.min())
                    model_out_avg_img = (model_out_avg_img * 255).astype(np.uint8)
                    model_out_avg_img = Image.fromarray(model_out_avg_img)
                    model_out_avg_img = model_out_avg_img.convert('RGB')
                    model_out_avg_img.save(os.path.join(out_dir, "%04d_output_avg.png" % step))

                    #save smoothed model output to numpy file
                    model_out_avg_np = out_avg.cpu().view(sidelength,sidelength).detach().numpy()
                    np.save(os.path.join(out_dir, "%04d_output_avg.npy" % step), model_out_avg_np)

                optim.zero_grad()
                loss.backward()
                optim.step()

            with torch.no_grad():
                #plot PSNR history
                plt.plot(psnr_clean_history, label='PSNR (Clean)', color='orange')
                plt.plot(psnr_noised_history, label='PSNR (Noised)', color='blue')
                plt.xlabel('Step')
                plt.ylabel('PSNR')
                plt.title('PSNR History')
                plt.legend()
                plt.savefig(os.path.join(out_dir, "psnr_history.png"))
                plt.close()

                #plot grad entropy history
                plt.plot(entropy_history, label='Gradient Entropy', color='green')
                plt.xlabel('Step')
                plt.ylabel('Entropy')
                plt.title('Gradient Entropy History')
                plt.legend()
                plt.savefig(os.path.join(out_dir, "entropy_history.png"))
                plt.close()

                #output super-resolution image
                upsample_factor = 4
                coords_superres = get_mgrid(sidelength*upsample_factor, 2).cuda()
                model_output_superres, _ = img_siren(coords_superres)
                model_output_superres = model_output_superres.view(sidelength*upsample_factor, sidelength*upsample_factor).cpu().detach().numpy()

                #get downscaled super-resolved image
                model_output_superres_down = Image.fromarray(model_output_superres)
                model_output_superres_down = model_output_superres_down.resize((sidelength, sidelength), Image.BICUBIC)
                model_output_superres_down = np.array(model_output_superres_down)

                #calc PSNR of super-resolved image
                psnr_superres = peak_signal_noise_ratio(clean_img.cpu().view(sidelength,sidelength).detach().numpy(), model_output_superres_down)
                print(f"PSNR (Super-resolved): {psnr_superres:.3f}")

                #save super-resolved image=======================================
                model_output_superres = (model_output_superres - model_output_superres.min()) / (model_output_superres.max() - model_output_superres.min())
                model_output_superres = (model_output_superres * 255).astype(np.uint8)
                model_output_superres = Image.fromarray(model_output_superres)
                model_output_superres = model_output_superres.convert('RGB')
                model_output_superres.save(os.path.join(out_dir, "superres.png"))

                #save down-scaled super-resolved image===========================
                model_output_superres_down = (model_output_superres_down - model_output_superres_down.min()) / (model_output_superres_down.max() - model_output_superres_down.min())
                model_output_superres_down = (model_output_superres_down * 255).astype(np.uint8)
                model_output_superres_down = Image.fromarray(model_output_superres_down)
                model_output_superres_down = model_output_superres_down.convert('RGB')
                model_output_superres_down.save(os.path.join(out_dir, "superres_down.png"))

                #save GIF of training images
                images = []

                for step in range(total_steps):
                    if step % steps_til_summary == 0:
                        img = Image.open("out_img/%04d.png" % step)
                        images.append(img)

                images[0].save(os.path.join(out_dir,'training.gif'), save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

                #save model
                torch.save(img_siren.state_dict(), os.path.join(out_dir, "model.pth"))