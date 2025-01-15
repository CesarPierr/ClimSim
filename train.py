import argparse
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ClimSimDataset
from models import ConditionalFlowGenerator, ConditionalWGANGPDiscriminator
from models import gradient_penalty_conditional  


def parse_arguments():

    parser = argparse.ArgumentParser(description="Training WGAN-GP (conditional) with Flow + TensorBoard + Checkpoints.")
    parser.add_argument("--resume_gen", type=str, default="", help="Path to a generator checkpoint to resume from.")
    parser.add_argument("--resume_disc", type=str, default="", help="Path to a discriminator checkpoint to resume from.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty coefficient.")
    parser.add_argument("--alpha_nll", type=float, default=1.0, help="Coefficient for the NLL part of loss.")
    parser.add_argument("--gamma_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--visual_interval", type=int, default=1, help="Epoch interval to log images in TensorBoard.")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory for TensorBoard logs.")
    
    return parser.parse_args()


def train_wgangp_conditional(
    gen,
    disc,
    loader,
    device,
    num_epochs=10,
    lr=1e-4,
    lambda_gp=10.0,
    alpha_nll=1.0,
    gamma_clip=1.0,
    visual_interval=1,
    writer=None,
    save_dir="checkpoints"
):
    """ 
    Entraînement du WGAN-GP conditionnel avec Flow.
    - Logs TensorBoard (writer) si writer n'est pas None.
    - Sauvegarde des checkpoints à chaque epoch dans save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.9))

    gen.train()
    disc.train()

    global_step = 0  # Compteur d'itérations (pour TensorBoard)
    for epoch in range(num_epochs):
        epoch_disc_loss = 0.
        epoch_gen_adv = 0.
        epoch_gen_nll = 0.
        epoch_gp = 0.

        # ------------------
        # Boucle sur le DataLoader
        # ------------------
        for batch_data in tqdm(loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", total=len(loader)):
            x = batch_data["input"].to(device)   # (B, Pos, Alt, Cx)
            y = batch_data["target"].to(device)  # (B, Pos, Alt, Cy)

            B, P, Alt, Cy = y.shape
            total_dims = float(P * Alt * Cy)

            # 1) Mise à jour du Discriminateur
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

            # 2) Mise à jour du Générateur
            fake2 = gen.sample(x)
            disc_fake2 = disc(x, fake2)
            loss_g_adv = -disc_fake2.mean()

            log_p = gen.log_prob(y, x)  # (B,)
            loss_nll = - (log_p / total_dims).mean()

            loss_g_total = loss_g_adv + alpha_nll * loss_nll

            opt_gen.zero_grad()
            loss_g_total.backward()
            utils.clip_grad_norm_(gen.parameters(), gamma_clip)
            opt_gen.step()

            epoch_gen_adv += loss_g_adv.item()
            epoch_gen_nll += loss_nll.item()

            if writer is not None:
                writer.add_scalar("Train/Discriminator", loss_d.item(), global_step)
                writer.add_scalar("Train/GP", gp.item(), global_step)
                writer.add_scalar("Train/Generator_adv", loss_g_adv.item(), global_step)
                writer.add_scalar("Train/Generator_NLL", loss_nll.item(), global_step)

            global_step += 1

        nb_batches = len(loader)
        avg_disc_loss = epoch_disc_loss / nb_batches
        avg_gp        = epoch_gp / nb_batches
        avg_gen_adv   = epoch_gen_adv / nb_batches
        avg_gen_nll   = epoch_gen_nll / nb_batches

        print(f"Epoch[{epoch+1}/{num_epochs}] "
              f"D_loss={avg_disc_loss:.4f}, "
              f"GP={avg_gp:.4f}, "
              f"G_adv={avg_gen_adv:.4f}, "
              f"G_nll={avg_gen_nll:.4f}")

        if writer is not None:
            writer.add_scalar("Epoch/D_loss", avg_disc_loss, epoch+1)
            writer.add_scalar("Epoch/GP", avg_gp, epoch+1)
            writer.add_scalar("Epoch/G_adv", avg_gen_adv, epoch+1)
            writer.add_scalar("Epoch/G_nll", avg_gen_nll, epoch+1)


        if (epoch+1) % visual_interval == 0:
            gen.eval()
            with torch.no_grad():

                sample_x = x[:1]     # (1, Pos, Alt, Cx)
                sample_y = y[:1]     # (1, Pos, Alt, Cy)
                sample_fake = gen.sample(sample_x)  # (1, Pos, Alt, Cy)

                real_2d = sample_y[0, :, :, 0].cpu().numpy()
                fake_2d = sample_fake[0, :, :, 0].cpu().numpy()

                fig, axs = plt.subplots(1, 2, figsize=(10,4))
                axs[0].imshow(real_2d.T, aspect='auto', origin='lower')
                axs[0].set_title("Real (channel=0)")
                axs[1].imshow(fake_2d.T, aspect='auto', origin='lower')
                axs[1].set_title("Fake (channel=0)")
                plt.tight_layout()

                if writer is not None:
                    writer.add_figure("Real_vs_Fake", fig, global_step=epoch+1)
                
                plt.close(fig)
            gen.train()

        gen_ckpt_path  = os.path.join(save_dir, f"gen_epoch_{epoch+1}.pth")
        disc_ckpt_path = os.path.join(save_dir, f"disc_epoch_{epoch+1}.pth")
        torch.save(gen.state_dict(), gen_ckpt_path)
        torch.save(disc.state_dict(), disc_ckpt_path)

    print("Entraînement terminé.")


def main():
    args = parse_arguments()

    # Initialisation de TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prépare le dataset
    base_dir = "/home/deemel/pierre/dataset"
    dataset = ClimSimDataset(
        base_dir=base_dir,
        grid_file="ClimSim_low-res_grid-info.nc",
        normalize=True,
        data_split="train",
        regexps=[
            "E3SM-MMF.mli.000[1234567]-*-*-*.nc",
            "E3SM-MMF.mli.0008-01-*-*.nc",
        ],
        cnn_reshape=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Instanciation G et D
    gen = ConditionalFlowGenerator(context_channels=6, latent_channels=10, num_flows=6).to(device)
    disc = ConditionalWGANGPDiscriminator(in_channels_x=6, in_channels_y=10, hidden_channels=[64,64,64]).to(device)

    # Reprise d'entraînement si spécifié
    if args.resume_gen:
        print(f"==> Loading generator checkpoint from {args.resume_gen}")
        gen.load_state_dict(torch.load(args.resume_gen, map_location=device))
    if args.resume_disc:
        print(f"==> Loading discriminator checkpoint from {args.resume_disc}")
        disc.load_state_dict(torch.load(args.resume_disc, map_location=device))

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
        writer=writer,
        save_dir=args.save_dir
    )

    writer.close()


if __name__ == "__main__":
    main()
