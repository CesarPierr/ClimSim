import torch
import torch.nn as nn
import torch.nn.functional as F


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
    )[0]  # shape: (B*P, Alt, Cy) selon le reorder

    grads = grads.reshape(grads.size(0), -1)  # (B*P, n_features)
    grad_norm = grads.norm(2, dim=1)          # (B*P,)

    penalty = ((grad_norm - 1.0)**2).mean() * lambda_gp
    return penalty


class PNet2d(nn.Module):
    """
    Version 2D : On applique des Conv2d sur (Alt, Pos).
    On suppose l'entrée x a shape (B, Pos, Alt, C_in),
    et on veut en sortie mu, sigma de shape (B, Pos, Alt, out_channels).
    """
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[64, 64, 64]):
        super().__init__()
        # Les conv2d prendront in_channels=6 (C_in),
        # et on considère (Alt, Pos) comme (H, W).
        layers = []
        inC = in_channels
        # 1er bloc
        layers.append(nn.Conv2d(inC, hidden_channels[0], kernel_size=(7,9), padding=(3,4)))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU())
        inC = hidden_channels[0]

        # 2e bloc
        layers.append(nn.Conv2d(inC, hidden_channels[1], kernel_size=(5,7), padding=(2,3)))
        layers.append(nn.BatchNorm2d(hidden_channels[1]))
        layers.append(nn.LeakyReLU())
        
        inC = hidden_channels[1]

        # 3e bloc
        layers.append(nn.Conv2d(inC, hidden_channels[2], kernel_size=(3,5), padding=(1,2)))
        layers.append(nn.BatchNorm2d(hidden_channels[2]))
        layers.append(nn.LeakyReLU())
        
        inC = hidden_channels[2]

        # Dernier bloc : output = out_channels * 2
        layers.append(nn.Conv2d(inC, out_channels * 2, kernel_size=3, padding=1))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        """
        x.shape = (B, Pos, Alt, C_in).
        - On permute pour avoir (B, C_in, Alt, Pos).
        - On fait des conv2d.
        - On ressort (B, out_channels*2, Alt, Pos).
        - On retranspose pour retrouver (B, Pos, Alt, out_channels*2).
        - On sépare mu, log_sigma, exponentie => sigma
        """
        B, P, A, Cin = x.shape
        # Permuter => (B, Cin, A, P)
        x = x.permute(0, 3, 2, 1)  # (B, C_in, Alt, Pos)

        out = self.conv2d(x)  # => (B, out_channels*2, Alt, Pos)

        # Retranspose => (B, Alt, Pos, out_channels*2)
        out = out.permute(0, 2, 3, 1)  # (B, A, P, out_channels*2)
        # => (B, P, A, out_channels*2) si vous voulez vraiment P en 2e dim
        #   On suppose qu'on veut (B, P, A, channels) => permute(0, 3, 1, 2)?
        #   Mais vous disiez plus haut: shape finale = (B, P, Alt, ...).
        #   Alors on repasse: (B, A, P, ...) -> (B, P, A, ...)

        out = out.permute(0, 2, 1, 3)  # => (B, P, A, out_channels*2)

        mu, log_sigma = torch.chunk(out, 2, dim=-1)  # sépare sur la dernière dim
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class KNet2d(nn.Module):
    """
    Produit k(x) en 2D. shape finale (B, Pos, Alt, out_channels).
    """
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 32]):
        super().__init__()
        layers = []
        inC = in_channels
        # Bloc 1
        layers.append(nn.Conv2d(inC, hidden_channels[0], kernel_size=(7,7), padding=(3,3)))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU())
        # Bloc 2
        layers.append(nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=(5,5), padding=(2,2)))
        layers.append(nn.BatchNorm2d(hidden_channels[1]))
        layers.append(nn.LeakyReLU())
        # Bloc final
        layers.append(nn.Conv2d(hidden_channels[1], out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        """
        x.shape = (B, Pos, Alt, C_in).
        Sortie = (B, Pos, Alt, out_channels).
        On applique conv2d sur (Alt, Pos).
        On utilise un softplus pour la positivité.
        """
        B, P, A, Cin = x.shape
        x = x.permute(0, 3, 2, 1)  # => (B, Cin, A, P)
        out = self.conv2d(x)      # => (B, out_channels, A, P)
        out = out.permute(0, 3, 2, 1)  # => (B, P, A, out_channels)
        out = F.softplus(out)
        return out


class BNet2d(nn.Module):
    """
    Produit b(x) en 2D. shape finale (B, Pos, Alt, out_channels).
    """
    def __init__(self, in_channels=6, out_channels=10, hidden_channels=[32, 32]):
        super().__init__()
        layers = []
        inC = in_channels
        # Bloc 1

        layers.append(nn.Conv2d(inC, hidden_channels[0], kernel_size=7, padding=3))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU())
        # Bloc 2

        layers.append(nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=5, padding=2))
        layers.append(nn.BatchNorm2d(hidden_channels[1]))
        layers.append(nn.LeakyReLU())
        # Bloc final
        layers.append(nn.Conv2d(hidden_channels[1], out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        B, P, A, Cin = x.shape
        x = x.permute(0, 3, 2, 1)   # => (B, Cin, A, P)
        out = self.conv2d(x)       # => (B, out_channels, A, P)
        out = out.permute(0, 3, 2, 1)  # => (B, P, A, out_channels)
        return out
    
class ConditionalAffineFlow2d(nn.Module):
    def __init__(self, context_channels=6, latent_channels=10):
        super().__init__()
        self.k_net = KNet2d(in_channels=context_channels, out_channels=latent_channels)
        self.b_net = BNet2d(in_channels=context_channels, out_channels=latent_channels)
        self.eps = 1e-6

    def forward(self, z, x):
        """
        z.shape = (B, Pos, Alt, latent_channels)
        x.shape = (B, Pos, Alt, context_channels)
        """
        k = self.k_net(x)  # => (B, P, A, latent_channels)
        b = self.b_net(x)
        z_out = k * z + b

        log_det = torch.log(k + self.eps)
        log_det = torch.sum(log_det, dim=(1,2,3))  # => (B,)
        return z_out, log_det

    def inverse(self, z, x):
        k = self.k_net(x)
        b = self.b_net(x)
        z_in = (z - b) / (k + self.eps)
        return z_in


class ConditionalFlowGenerator2d(nn.Module):
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
        mu, sigma = self.p_net(x)   # => shape (B, P, A, latent_ch)
        z = mu + sigma * torch.randn_like(sigma)
        total_log_det = 0.
        for flow in self.flows:
            z, log_det = flow(z, x)
            total_log_det += log_det
        return z, total_log_det, mu, sigma

    def sample(self, x):
        u, _, _, _ = self.forward(x)
        return u

    def log_prob(self, u, x):
        """
        Même logique que dans votre code existant, 
        on inverse flow, calcule la log-prob gaussienne, etc.
        """
        mu, sigma = self.p_net(x)
        z = u
        # inverse flows
        for flow in reversed(self.flows):
            z = flow.inverse(z, x)

        # log-prob gauss
        diff = (z - mu) / (sigma + 1e-6)
        lp_gauss = -0.5 * torch.sum(
            diff**2 + torch.log(2*torch.pi*sigma**2 + 1e-6),
            dim=(1,2,3)
        )
        # forward flows pour sum log_det
        z2 = z
        total_log_det = 0.
        for flow in self.flows:
            z2, log_det = flow(z2, x)
            total_log_det += log_det

        return lp_gauss + total_log_det

class ConditionalWGANGPDiscriminator2d(nn.Module):
    def __init__(self, in_channels_x=6, in_channels_y=10, hidden_channels_params=[(64,7,3),(32,5,2),(16,3,1)]):
        super().__init__()
        total_in = in_channels_x + in_channels_y
        layers = []
        inC = total_in
        # on va faire 2D => conv2d(inC, hidden_channels[0], 3, padding=1), ...
        for hc,ker,pad in hidden_channels_params:
            layers.append(nn.BatchNorm2d(inC))
            layers.append(nn.Conv2d(inC, hc, kernel_size=ker, padding=pad))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            inC = hc
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels_params[-1][0], 1)

    def forward(self, x, y):
        """
        x.shape = (B, P, Alt, Cx)
        y.shape = (B, P, Alt, Cy)
        => On concatène sur la dimension canaux => (B, P, Alt, Cx+Cy).
        => On permute => (B, C_in, Alt, P).
        => On conv2d => (B, hiddenC, Alt, P).
        => On average pool => fc => (B, P*??, 1)? ou on fait un mean sur (Alt, P)? 
        """
        B, P, A, Cx = x.shape
        Cy = y.shape[-1]
        combined = torch.cat((x, y), dim=-1)  # => (B, P, A, Cx+Cy)
        # permute => (B, C_in, A, P)
        combined = combined.permute(0, 3, 2, 1)  # => (B, C_in, Alt, Pos)

        out = self.conv(combined)  # => (B, hidden_channels[-1], Alt, Pos)

        # Pooling global
        out = out.mean(dim=(2,3))  # => (B, hidden_channels[-1])

        score = self.fc(out)       # => (B, 1)
        return score.squeeze(-1)   # => (B,)
