import argparse
import os
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb
from PIL import Image  # Import nécessaire pour manipuler les images PIL
from dataset import ClimSimDataset
from models import (
    ConditionalFlowGenerator2d,
    ConditionalWGANGPDiscriminator2d,
    gradient_penalty_conditional,
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training WGAN-GP (conditional) with Flow + optional Reconstruction Loss + wandb + Checkpoints."
    )
    parser.add_argument("--resume_gen", type=str, default="", help="Path to a generator checkpoint to resume from.")
    parser.add_argument("--resume_disc", type=str, default="", help="Path to a discriminator checkpoint to resume from.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty coefficient.")
    parser.add_argument("--alpha_nll", type=float, default=0.5, help="Coefficient for the NLL part of loss.")
    parser.add_argument("--gamma_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--visual_interval", type=int, default=1, help="Epoch interval to log images with wandb.")
    parser.add_argument("--use_recon", action="store_true", help="Use MSE reconstruction loss or not.")
    parser.add_argument("--alpha_recon", type=float, default=1.0, help="Weight of the reconstruction loss if use_recon is True.")

    # Ajout de l'argument "project" pour configurer wandb
    parser.add_argument("--wandb_project", type=str, default="ClimSim", help="Wandb project name.")

    return parser.parse_args()

def fig_to_wandb_image(fig):
    """
    Convertit une figure Matplotlib en image utilisable par wandb.
    Retourne un objet `wandb.Image`.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")  # Conversion en image RGB pour éviter les problèmes de mode
    return wandb.Image(image)  # Passe l'image PIL à wandb.Image

def train_wgangp_conditional(
    gen,
    disc,
    loader,
    device,
    num_epochs=10,
    lr=1e-4,
    lr_discr=1e-5,
    lambda_gp=10.0,
    alpha_nll=1.0,
    gamma_clip=1.0,
    visual_interval=1,
    save_dir="checkpoints",
    use_recon=False,
    alpha_recon=1.0,
    initial_step=0  # Ajout d'un paramètre pour initialiser global_step
):
    """
    Entraînement du WGAN-GP conditionnel avec Flow + (optionnel) Reconstruction MSE.
    On log aussi des métriques (moyenne, variance) pour détecter un éventuel mode collapse, 
    mais tout se fait via wandb.
    """
    os.makedirs(save_dir, exist_ok=True)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr_discr, betas=(0.5, 0.9))

    gen.train()
    disc.train()

    global_step = initial_step  # Initialisation de global_step
    for epoch in range(num_epochs):
        epoch_disc_loss = 0.0
        epoch_gen_adv   = 0.0
        epoch_gen_nll   = 0.0
        epoch_gen_recon = 0.0
        epoch_gp        = 0.0

        for batch_data in tqdm(loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", total=len(loader)):
            x = batch_data["input"].to(device)   # (B, Pos, Alt, Cx)
            y = batch_data["target"].to(device)  # (B, Pos, Alt, Cy)

            B, P, Alt, Cy = y.shape
            total_dims = float(P * Alt * Cy)

            # ----------------------
            #   1) Discriminateur
            # ----------------------
            fake = gen.sample(x).detach()
            disc_real = disc(x, y)
            disc_fake = disc(x, fake)

            loss_d = -(disc_real.mean() - disc_fake.mean())  # WGAN
            gp = gradient_penalty_conditional(disc, x, y, fake, device, lambda_gp=lambda_gp)
            loss_d_total = loss_d + gp

            opt_disc.zero_grad()
            loss_d_total.backward()
            utils.clip_grad_norm_(disc.parameters(), gamma_clip)
            opt_disc.step()

            epoch_disc_loss += loss_d.item()
            epoch_gp        += gp.item()

            # ----------------------
            #   2) Générateur
            # ----------------------
            fake2 = gen.sample(x)
            disc_fake2 = disc(x, fake2)
            loss_g_adv = -disc_fake2.mean()

            log_p = gen.log_prob(y, x)  # (B,)
            loss_nll = - (log_p / total_dims).mean()

            loss_g_total = loss_g_adv + alpha_nll * loss_nll

            if use_recon:
                loss_recon = F.mse_loss(fake2, y)
                loss_g_total += alpha_recon * loss_recon
                epoch_gen_recon += loss_recon.item()

            opt_gen.zero_grad()
            loss_g_total.backward()
            utils.clip_grad_norm_(gen.parameters(), gamma_clip)
            opt_gen.step()

            epoch_gen_adv += loss_g_adv.item()
            epoch_gen_nll += loss_nll.item()

            # ----------------------
            #   3) Logging wandb
            # ----------------------
            # On loggue les scalaires avec global_step
            logs = {
                "Train/Discriminator": loss_d.item(),
                "Train/GP": gp.item(),
                "Train/Generator_adv": loss_g_adv.item(),
                "Train/Generator_NLL": loss_nll.item(),
            }
            if use_recon:
                logs["Train/Generator_Recon"] = loss_recon.item()

            # (optionnel) Suivi de la moyenne/variance Real vs Fake
            with torch.no_grad():
                real_mean = y.mean(dim=(1,2,3))     # (B,)
                real_var  = y.var(dim=(1,2,3))      # (B,)
                fake_mean = fake2.mean(dim=(1,2,3)) # (B,)
                fake_var  = fake2.var(dim=(1,2,3))  # (B,)

                mean_diff = (real_mean - fake_mean).abs().mean()
                var_diff  = (real_var - fake_var).abs().mean()

            logs["Dist/Mean_diff"] = mean_diff.item()
            logs["Dist/Var_diff"]  = var_diff.item()

            wandb.log(logs, step=global_step)

            global_step += 1

        # Moyennes sur l'époque
        nb_batches = len(loader)
        avg_disc_loss = epoch_disc_loss / nb_batches
        avg_gp        = epoch_gp / nb_batches
        avg_gen_adv   = epoch_gen_adv / nb_batches
        avg_gen_nll   = epoch_gen_nll / nb_batches
        avg_gen_recon = (epoch_gen_recon / nb_batches) if use_recon else 0.0

        print(
            f"Epoch[{epoch+1}/{num_epochs}] "
            f"D_loss={avg_disc_loss:.4f}, "
            f"GP={avg_gp:.4f}, "
            f"G_adv={avg_gen_adv:.4f}, "
            f"G_nll={avg_gen_nll:.4f}, "
            f"G_recon={avg_gen_recon:.4f} (if used)"
        )

        # On log les moyennes d'époque avec global_step
        epoch_logs = {
            "Epoch/D_loss": avg_disc_loss,
            "Epoch/GP": avg_gp,
            "Epoch/G_adv": avg_gen_adv,
            "Epoch/G_nll": avg_gen_nll,
        }
        if use_recon:
            epoch_logs["Epoch/G_recon"] = avg_gen_recon

        wandb.log(epoch_logs, step=global_step)

        # ------------------
        # Visualisation
        # ------------------
        if (epoch+1) % visual_interval == 0:
            gen.eval()
            with torch.no_grad():
                # On réutilise le dernier batch (x,y) pour la visualisation
                sample_x = x[:1]     # (1, P, Alt, Cx)
                sample_y = y[:1]     # (1, P, Alt, Cy)
                sample_fake = gen.sample(sample_x)  # (1, P, Alt, Cy)

                real_2d = sample_y[0, :, :, 0].cpu().numpy()
                fake_2d = sample_fake[0, :, :, 0].cpu().numpy()

                fig, axs = plt.subplots(1, 2, figsize=(10,4))
                axs[0].imshow(real_2d.T, aspect='auto', origin='lower')
                axs[0].set_title("Real (channel=0)")
                axs[1].imshow(fake_2d.T, aspect='auto', origin='lower')
                axs[1].set_title("Fake (channel=0)")
                plt.tight_layout()

                # Conversion en image wandb avec global_step
                wandb.log({"Real_vs_Fake": fig_to_wandb_image(fig)}, step=global_step)

                plt.close(fig)
            gen.train()

        # Sauvegarde checkpoints avec global_step
        checkpoint = {
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'global_step': global_step
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    print("Entraînement terminé.")

def main():
    args = parse_arguments()

    # Initialisation wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),  # enregistre tous les arguments
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    base_dir = "/home/ensta/ensta-cesar/dataset"  # Ajustez selon vos chemins
    dataset_dir = "/home/ensta/ensta-cesar/ClimSim"
    dataset = ClimSimDataset(
        base_dir=base_dir,
        dataset_dir=dataset_dir,  # Assurez-vous que l'argument est correct selon votre implémentation
        grid_file="ClimSim_low-res_grid-info.nc",
        normalize=True,
        data_split="train",
        regexps=[
            "E3SM-MMF.mli.000[1234567]-*-*-*.nc",
            "E3SM-MMF.mli.0008-01-*-*.nc",
        ],
        cnn_reshape=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    # Générateur et Discriminateur
    gen = ConditionalFlowGenerator2d(
        context_channels=6, 
        latent_channels=10, 
        num_flows=8 
    ).to(device)
    
    disc = ConditionalWGANGPDiscriminator2d(
        in_channels_x=6, 
        in_channels_y=10, 
        #hidden_channels=[64,64,64]
    ).to(device)

    # Initialisation de global_step
    initial_step = 0

    # Reprise d'entraînement si spécifié
    if args.resume_gen and args.resume_disc:
        print(f"==> Loading generator checkpoint from {args.resume_gen}")
        print(f"==> Loading discriminator checkpoint from {args.resume_disc}")
        checkpoint = torch.load(args.resume_gen, map_location=device)
        
        if isinstance(checkpoint, dict):
            gen.load_state_dict(checkpoint['gen_state_dict'])
            disc.load_state_dict(checkpoint['disc_state_dict'])
            initial_step = checkpoint.get('global_step', 0)
        else:
            # Si le checkpoint ne contient pas 'global_step', vous pouvez définir initial_step par ailleurs
            gen.load_state_dict(checkpoint)
            disc.load_state_dict(torch.load(args.resume_disc, map_location=device))
            # initial_step reste à 0 ou définissez-le selon votre logique

    # Entraînement
    train_wgangp_conditional(
        gen=gen,
        disc=disc,
        loader=loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lambda_gp=args.lambda_gp,
        alpha_nll=args.alpha_nll,
        gamma_clip=args.gamma_clip,
        visual_interval=args.visual_interval,
        save_dir=args.save_dir,
        use_recon=args.use_recon,
        alpha_recon=args.alpha_recon,
        initial_step=initial_step  # Passer l'étape initiale
    )

    # Fin de l'expérience wandb
    wandb.finish()

if __name__ == "__main__":
    main()
