import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_penalty_conditional(disc, x, y_real, y_fake, device, lambda_gp=10.0):
    B, P, Alt, Cy = y_real.shape

    alpha = torch.rand(B, 1, 1, 1, device=device)
    alpha = alpha.expand(B, P, Alt, Cy)
    
    y_interpolates = alpha * y_real + (1 - alpha) * y_fake
    y_interpolates = y_interpolates.detach()
    y_interpolates.requires_grad_(True)
    disc_interpolates = disc(x, y_interpolates)
    grads = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=y_interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.reshape(grads.size(0), -1)
    grad_norm = grads.norm(2, dim=1)
    penalty = ((grad_norm - 1.0)**2).mean() * lambda_gp
    
    return penalty

class PNet2d(nn.Module):
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 64, 128]):
        super().__init__()
        layers = []
        inC = in_channels
        # 1er bloc
        layers.append(nn.Conv2d(inC, hidden_channels[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU())
        inC = hidden_channels[0]

        # 2e bloc
        layers.append(nn.Conv2d(inC, hidden_channels[1], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[1]))
        layers.append(nn.LeakyReLU())
        inC = hidden_channels[1]

        # 3e bloc
        layers.append(nn.Conv2d(inC, hidden_channels[2], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[2]))
        layers.append(nn.LeakyReLU())
        inC = hidden_channels[2]

        # Dernier bloc : output = out_channels * 2
        layers.append(nn.Conv2d(inC, out_channels * 2, kernel_size=3, padding=1))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        """
        x.shape = (B, P, A, Cin)
        => on permute (B, Cin, P, A)
        => conv2d => (B, 2*outC, P, A)
        => on retranspose => (B, P, A, 2*outC)
        => on split => mu, sigma
        """
        x = x.permute(0, 3, 1, 2)
        out = self.conv2d(x)  # => (B, 2*outC, P, A)
        out = out.permute(0, 2, 3, 1)  
        mu, log_sigma = torch.chunk(out, 2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class KNet2d(nn.Module):
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 64]):
        super().__init__()
        layers = []
        inC = in_channels
        # Bloc 1
        layers.append(nn.Conv2d(inC, hidden_channels[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU())
        # Bloc 2
        layers.append(nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[1]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(hidden_channels[1], out_channels, kernel_size=3, padding=1))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        """
        x.shape = (B, P, A, Cin)
        """
        B, P, A, Cin = x.shape
        x = x.permute(0, 3, 1, 2)           # => (B, Cin, P, A)
        out = self.conv2d(x)               # => (B, out_channels, P, A)
        out = out.permute(0, 2, 3, 1)      # => (B, P, A, out_channels)
        out = F.softplus(out)
        return out
    
class BNet2d(nn.Module):
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 64]):
        super().__init__()
        layers = []
        inC = in_channels
        layers.append(nn.Conv2d(inC, hidden_channels[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[1]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(hidden_channels[1], out_channels, kernel_size=3, padding=1))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        B, P, A, Cin = x.shape
        x = x.permute(0, 3, 1, 2)  # => (B, Cin, P, A)
        out = self.conv2d(x)      # => (B, outC, P, A)
        out = out.permute(0, 2, 3, 1)  # => (B, P, A, outC)
        return out

class ConditionalAffineFlow2d(nn.Module):
    """
    Affine flow de type z_out = k(x)*z + b(x), pixel-wise.
    Retourne aussi un log_det pixel-wise de shape (B, P, A, C).
    """
    def __init__(self, context_channels=6, latent_channels=10):
        super().__init__()
        self.k_net = KNet2d(in_channels=context_channels, out_channels=latent_channels)
        self.b_net = BNet2d(in_channels=context_channels, out_channels=latent_channels)
        self.eps = 1e-6

    def forward(self, z, x):
        """
        z.shape = (B, P, A, latent_channels)
        x.shape = (B, P, A, context_channels)

        Retourne (z_out, log_det_pixelwise) où:
          z_out: (B, P, A, latent_channels)
          log_det_pixelwise: (B, P, A, latent_channels)
        """
        k = self.k_net(x)  # => (B, P, A, latent_channels)
        b = self.b_net(x)  # => (B, P, A, latent_channels)

        z_out = k * z + b

        log_det_pixelwise = torch.log(k + self.eps)  # => (B, P, A, C)

        return z_out, log_det_pixelwise

    def inverse(self, z, x):
        """
        Inversion: z_in = (z_out - b(x)) / k(x)
        """
        k = self.k_net(x)
        b = self.b_net(x)
        z_in = (z - b) / (k + self.eps)
        return z_in

class ConditionalFlowGenerator2d(nn.Module):
    """
    Flow complet : p_net (Gauss) + plusieurs affines flows.
    """
    def __init__(self, context_channels=6, latent_channels=10, num_flows=4):
        super().__init__()
        self.p_net = PNet2d(
            in_channels=context_channels, 
            out_channels=latent_channels
        )
        self.flows = nn.ModuleList([
            ConditionalAffineFlow2d(context_channels, latent_channels)
            for _ in range(num_flows)
        ])

    def forward(self, x):
        """
        Renvoie (z_final, log_det_pixelwise_total, mu, sigma).

        log_det_pixelwise_total a shape (B, P, A, C).
        """
        mu, sigma = self.p_net(x)  # => shape (B,P,A,C)
        # Echantillon latents
        z = mu + sigma * torch.randn_like(sigma)

        # On accumule le log_det de chaque flow (pixel-wise)
        total_log_det_px = torch.zeros_like(z)

        for flow in self.flows:
            z, log_det_px = flow(z, x)  # => shapes (B,P,A,C) pour z et log_det_px
            total_log_det_px += log_det_px  # pixel-wise

        return z, total_log_det_px, mu, sigma

    def sample(self, x):
        """
        Génère un échantillon final dans l'espace y, en forward.
        """
        z, _, _, _ = self.forward(x)
        return z
    
    def sample_mode(self, x):
        """
        Génère le sample le plus probable (mode) en utilisant mu sans bruit.
        """
        mu, _ = self.p_net(x)  # Use only the mean
        z = mu  # deterministic sample: most likely latent vector
        for flow in self.flows:
            z, _ = flow(z, x)
        return z
    
    def sample_most_probable(self, x, num_samples=100):
        """
        Generate multiple samples and return the most probable value for each pixel position.
        
        Args:
            x: Input context tensor of shape (B, P, A, C_in)
            num_samples: Number of samples to generate
            
        Returns:
            Tensor with the most probable value for each pixel position
        """
        # Generate multiple samples and compute their log probabilities
        all_samples = []
        all_log_probs = []
        
        for _ in range(num_samples):
            sample = self.sample(x)
            log_prob = self.log_prob(sample, x)
            
            all_samples.append(sample.detach().cpu())
            all_log_probs.append(log_prob.detach().cpu())
        
        # Stack tensors along a new dimension
        all_samples = torch.stack(all_samples)
        all_log_probs = torch.stack(all_log_probs)
        
        # Find indices of max log probabilities along the sample dimension
        max_indices = torch.argmax(all_log_probs, dim=0)
        
        # Create output tensor
        best_sample = torch.zeros_like(all_samples[0])
        
        # Extract the most probable value for each position
        for i in range(num_samples):
            mask = (max_indices == i)
            best_sample[mask] = all_samples[i][mask]
        
        return best_sample.to(x.device)

    def log_prob(self, y, x):
        """
        Retourne le log_prob pixel-wise: shape (B, P, A, C).
        
        1) Inverse flows pour retrouver z.
        2) log-prob gauss pixel-wise
        3) repasse flows forward pour sommer log_det pixel-wise
        4) log_p = log_gauss + sum_log_det.
        """
        mu, sigma = self.p_net(x)
        # 1) Inversion
        z = y
        for flow in reversed(self.flows):
            z = flow.inverse(z, x)

        # 2) log-prob gauss pixel-wise (B,P,A,C)
        diff = (z - mu) / (sigma + 1e-6)
        lp_gauss_px = -0.5 * (
            diff**2 + torch.log(2 * torch.pi * sigma**2 + 1e-6)
        )  # => (B,P,A,C)

        # 3) On refait un forward flow pour accumuler le log_det
        z2 = z
        total_log_det_px = torch.zeros_like(z2)
        for flow in self.flows:
            z2, log_det_px = flow(z2, x)  # => (B,P,A,C), (B,P,A,C)
            total_log_det_px += log_det_px

        # 4) log_p final, pixel-wise
        lp_pixelwise = lp_gauss_px + total_log_det_px
        return lp_pixelwise


class ConditionalWGANGPDiscriminator2d(nn.Module):
    def __init__(self, in_channels_x=6, in_channels_y=10, hidden_channels_params=[(64,3,1),(32,3,1),(16,3,1)]):
        super().__init__()
        total_in = in_channels_x + in_channels_y
        layers = []
        inC = total_in
        for hc, ker, pad in hidden_channels_params:
            layers.append(nn.BatchNorm2d(inC))
            layers.append(nn.Conv2d(inC, hc, kernel_size=ker, padding=pad))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.1))
            inC = hc
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels_params[-1][0], 1)

    def forward(self, x, y):
        """
        x.shape = (B, P, A, Cx)
        y.shape = (B, P, A, Cy)
        => concat sur canaux => permute => conv2d => global mean => fc
        """
        B, P, A, Cx = x.shape
        Cy = y.shape[-1]
        combined = torch.cat((x, y), dim=-1)  # => (B, P, A, Cx+Cy)
        combined = combined.permute(0, 3, 2, 1)  # => (B, C_in, A, P)

        out = self.conv(combined)  # => (B, hiddenC, A, P)
        out = out.mean(dim=(2,3))  # => (B, hiddenC)
        score = self.fc(out)       # => (B, 1)
        return score.squeeze(-1)   # => (B,)
