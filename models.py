import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):

    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[16, 32, 64]):
        super(PNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(),

            nn.BatchNorm1d(hidden_channels[0]),
            nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.LeakyReLU(),

            nn.BatchNorm1d(hidden_channels[1]),
            nn.Conv1d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1),
            nn.LeakyReLU(),

            nn.BatchNorm1d(hidden_channels[2]),
            nn.Conv1d(hidden_channels[2], out_channels * 2, kernel_size=3, padding=1),
        )
        # Le *2 final permet de séparer la sortie en (mu, log_sigma)

    def forward(self, x):
        """
        x.shape = (B, Pos, Alt, C_in).
        On va aplatir (B, Pos) en un seul batch pour appliquer la conv1d sur Alt.
        """
        B, P, Alt, Cin = x.shape
        x = x.view(B * P, Alt, Cin)

        # On veut shape (batch=B*P, in_channels=Cin, width=Alt)
        x = x.permute(0, 2, 1) 

        out = self.conv_layers(x)  
        out = out.permute(0, 2, 1)
        out = out.view(B, P, Alt, -1) 
        mu, log_sigma = torch.chunk(out, 2, dim=-1)
        sigma = torch.exp(log_sigma)
        return mu, sigma

class KNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 64]):
        super(KNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels[1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):

        B, P, Alt, Cin = x.shape
        x = x.view(B * P, Alt, Cin)
        x = x.permute(0, 2, 1)  # (B*P, Cin, Alt)
        out = self.conv_layers(x)  # (B*P, out_channels, Alt)
        out = out.permute(0, 2, 1)  # (B*P, Alt, out_channels)
        out = out.view(B, P, Alt, -1)
        out = F.softplus(out)
        return out


class BNet(nn.Module):

    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 64]):
        super(BNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels[1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, P, Alt, Cin = x.shape
        x = x.view(B * P, Alt, Cin)
        x = x.permute(0, 2, 1)  # (B*P, Cin, Alt)
        out = self.conv_layers(x)  # (B*P, out_channels, Alt)
        out = out.permute(0, 2, 1)  # (B*P, Alt, out_channels)
        out = out.view(B, P, Alt, -1)
        return out


class ConditionalAffineFlow(nn.Module):

    def __init__(self, context_channels=6, latent_channels=10):
        super(ConditionalAffineFlow, self).__init__()
        self.k_net = KNet(in_channels=context_channels, out_channels=latent_channels)
        self.b_net = BNet(in_channels=context_channels, out_channels=latent_channels)
        self.eps = 1e-6

    def forward(self, z, x):

        k = self.k_net(x)  # (B, Pos, Alt, latent_channels)
        b = self.b_net(x)
        z_out = k * z + b

        log_det = torch.log(k + self.eps)
        log_det = torch.sum(log_det, dim=(1, 2, 3))  # -> (B,)
        
        return z_out, log_det

    def inverse(self, z, x):

        k = self.k_net(x)
        b = self.b_net(x)
        z_in = (z - b) / (k + self.eps)
        return z_in
    
class ConditionalFlowGenerator(nn.Module):

    def __init__(self, context_channels=6, latent_channels=10, num_flows=4):
        super(ConditionalFlowGenerator, self).__init__()
        self.p_net = PNet(in_channels=context_channels, out_channels=latent_channels)
        self.flows = nn.ModuleList(
            [ConditionalAffineFlow(context_channels, latent_channels) 
             for _ in range(num_flows)]
        )

    def forward(self, x):

        mu, sigma = self.p_net(x)  
        z = mu + sigma * torch.randn_like(sigma)  
        total_log_det = 0.0
        for flow in self.flows:
            z, log_det = flow(z, x)  
            total_log_det += log_det
        return z, total_log_det, mu, sigma

    def sample(self, x):

        u, _, _, _ = self.forward(x)
        return u

    def log_prob(self, u, x):

        mu, sigma = self.p_net(x)

        z = u
        inv_log_det = 0.0
        for flow in reversed(self.flows):
            z = flow.inverse(z, x)


        diff = (z - mu) / (sigma + 1e-6)
        lp_gauss = -0.5 * torch.sum(diff**2 + torch.log(2 * torch.pi * sigma**2 + 1e-6), 
                                    dim=(1,2,3))
        z2 = z
        total_log_det = 0.
        for flow in self.flows:
            z2, log_det = flow(z2, x)
            total_log_det += log_det

        return lp_gauss + total_log_det

class ConditionalWGANGPDiscriminator(nn.Module):

    def __init__(self, in_channels_x=6, in_channels_y=10, hidden_channels=[64, 64, 64]):
        super().__init__()
        self.in_channels_x = in_channels_x
        self.in_channels_y = in_channels_y
        total_in = in_channels_x + in_channels_y  

        layers = []
        inC = total_in
        for hc in hidden_channels:
            layers.append(nn.Conv1d(inC, hc, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            inC = hc
        self.conv = nn.Sequential(*layers)

        # Couche finale : on map vers 1 seul score (sans activation)
        self.fc = nn.Linear(hidden_channels[-1], 1)

    def forward(self, x, y):
        """
        x.shape = (B, Pos, Alt, C_x)
        y.shape = (B, Pos, Alt, C_y)

        """
        B, P, Alt, Cx = x.shape
        _, _, _, Cy = y.shape

        # Concat channels
        # => (B, Pos, Alt, Cx+Cy)
        combined = torch.cat((x, y), dim=-1)

        # Fusion (B,Pos) pour faire B' = B*Pos
        combined = combined.view(B * P, Alt, Cx + Cy)  # (B*P, Alt, C_in)
        combined = combined.permute(0, 2, 1)          # (B*P, C_in, Alt)

        out = self.conv(combined)                     # (B*P, hidden_channels[-1], Alt)
        out = out.mean(dim=2)                        
        out = self.fc(out)                            # (B*P, 1)
        return out.squeeze(-1)  
    
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
    )[0]  # shape: (B*P, Alt, Cy) selon le reorder

    grads = grads.reshape(grads.size(0), -1)  # (B*P, n_features)
    grad_norm = grads.norm(2, dim=1)          # (B*P,)

    penalty = ((grad_norm - 1.0)**2).mean() * lambda_gp
    return penalty
