import argparse
import os
import io
import glob
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb
from PIL import Image  

from dataset import load_dataset
from models import (
    ConditionalFlowGenerator2d,
    ConditionalWGANGPDiscriminator2d,
    gradient_penalty_conditional,
)
from visu import compute_animation_for_scalar, compute_animation_for_vector, denormalize_variable, compute_animation_for_temperature_difference, compute_animation_for_vector_difference, transform_longitude
from torch_uncertainty.metrics import AUSE

ause_metric = AUSE()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training WGAN-GP (conditional) with Flow + optional Reconstruction Loss + wandb + Checkpoints."
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Total number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr_discr", type=float, default=1e-4, help="Learning rate for the Discriminator.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty coefficient.")
    parser.add_argument("--alpha_nll", type=float, default=0.5, help="Coefficient for the NLL part of loss.")
    parser.add_argument("--gamma_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--visual_interval", type=int, default=1, help="Epoch interval to log images with wandb.")
    parser.add_argument("--use_recon", action="store_true", help="Use MSE reconstruction loss or not.")
    parser.add_argument("--alpha_recon", type=float, default=1.0, help="Weight of the reconstruction loss if use_recon is True.")
    parser.add_argument("--wandb_project", type=str, default="ClimSim", help="Wandb project name.")

    return parser.parse_args()

def fig_to_wandb_image(fig):
    """
    Convert a Matplotlib figure to a wandb-friendly image.
    Returns a `wandb.Image` object.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return wandb.Image(image)

def validate(gen, val_loader, device, norm_params=None, log_wandb=True, generate_video=False, fps=48, year=2000):
    """
    Validation function that computes the global MSE and uncertainty (AUSE) metrics,
    and—if requested—produces videos comparing denormalized predictions and ground truth.
    
    Assumes that:
      - Channel 0 is temperature (in normalized Kelvin, later converted to Celsius),
      - Channels 1 and 2 are wind components.
    
    norm_params is a dict with keys:
      '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind'
    that contain the corresponding min/max values.
    """
    if norm_params is None:
        norm_params = val_loader.dataset.get_norm_params()
    gen.eval()
    total_mse = 0.0
    n_samples = 0

    example_x = None
    example_y = None
    example_fake = None

    
    all_fakes = []
    all_reals = []
    
    
    all_incert_temp_pred = []
    all_incert_temp_real = []
    all_incert_wind_pred = []
    all_incert_wind_real = []
    
    
    ause_metric_temp = AUSE()
    ause_metric_wind = AUSE()

    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="[Validation]", leave=False):
            inputs = batch_data["input"].to(device)
            masks = batch_data["masks"].to(device)
            lat_coord = batch_data["coords"][0].unsqueeze(1).to(device)
            lon_coord = batch_data["coords"][1].unsqueeze(1).to(device)
            
            _ = batch_data["coords"][2].unsqueeze(1).to(device)
            coords = torch.cat([lat_coord, lon_coord], dim=1)
            x = torch.cat([inputs, masks, coords], dim=1)
            y = batch_data["target"].to(device)

            
            x = x.permute(0, 3, 2, 1)
            y = y.permute(0, 3, 2, 1)

            
            fake = gen.sample_most_probable(x,num_samples=3)
            all_fakes.append(fake.cpu())
            all_reals.append(y.cpu())

            
            loss_mse = F.mse_loss(fake, y, reduction='sum')
            total_mse += loss_mse.item()
            n_samples += y.numel()

            
            if example_x is None:
                example_x = x[0:1].cpu()
                example_y = y[0:1].cpu()
                example_fake = fake[0:1].cpu()

            
            lp_px = gen.log_prob(y, x)

            
            incert_pred_temp = -lp_px[..., 0]
            incert_real_temp = (fake[..., 0] - y[..., 0]).abs()
            all_incert_temp_pred.append(incert_pred_temp.cpu())
            all_incert_temp_real.append(incert_real_temp.cpu())
            ause_metric_temp.update(incert_pred_temp.reshape(-1), incert_real_temp.reshape(-1))

            
            incert_pred_wind = -(lp_px[..., 1:].mean(dim=-1))
            diff_u = (fake[..., 1] - y[..., 1])
            diff_v = (fake[..., 2] - y[..., 2])
            incert_real_wind = torch.sqrt(diff_u**2 + diff_v**2)
            all_incert_wind_pred.append(incert_pred_wind.cpu())
            all_incert_wind_real.append(incert_real_wind.cpu())
            ause_metric_wind.update(incert_pred_wind.reshape(-1), incert_real_wind.reshape(-1))
            


    avg_mse = total_mse / n_samples if n_samples > 0 else 0.0
    final_ause_temp = ause_metric_temp.compute()
    final_ause_wind = ause_metric_wind.compute()

    if log_wandb:
        wandb.log({
            "Val/MSE": avg_mse,
            "Val/AUSE_Temp": final_ause_temp,
            "Val/AUSE_Wind": final_ause_wind,
        })

    
    if generate_video and len(all_fakes) > 0:
        
        fakes_concat = torch.cat(all_fakes, dim=0)  
        reals_concat = torch.cat(all_reals, dim=0)

        fakes_concat = fakes_concat.permute(0, 3, 2, 1).cpu().numpy()  
        reals_concat = reals_concat.permute(0, 3, 2, 1).cpu().numpy()

        fakes_concat= transform_longitude(fakes_concat)
        reals_concat = transform_longitude(reals_concat)
        
        
        temp_pred = fakes_concat[:, 0]
        temp_real = reals_concat[:, 0]
        print("Temperature pred min before:", temp_pred.min(), temp_pred.max())
        print("Temperature real min before:", temp_real.min(), temp_real.max())
        
        if val_loader.dataset.normalize:
            temp_pred = denormalize_variable(temp_pred, norm_params['2m_temperature']) - 273.15
            temp_real = denormalize_variable(temp_real, norm_params['2m_temperature']) - 273.15

            
        else :
            temp_pred -= 273.15
            temp_real -= 273.15
        
        
        print("Temperature pred min:", temp_pred.min(), temp_pred.max())
        print("Temperature real min:", temp_real.min(), temp_real.max())
        wind_pred = fakes_concat[:, 1:3]
        wind_real = reals_concat[:, 1:3]

        
        print("Wind pred min before:", wind_pred.min(), wind_pred.max())
        print("Wind real min before:", wind_real.min(), wind_real.max())
        if val_loader.dataset.normalize:
            
            
            
            
            wind_real[:, 0] = denormalize_variable(wind_real[:, 0], norm_params['10m_u_component_of_wind'])
            wind_real[:, 1] = denormalize_variable(wind_real[:, 1], norm_params['10m_v_component_of_wind'])
            wind_pred[:, 0] = denormalize_variable(wind_pred[:, 0], norm_params['10m_u_component_of_wind'])
            wind_pred[:, 1] = denormalize_variable(wind_pred[:, 1], norm_params['10m_v_component_of_wind'])
        
        
        print("Wind pred min:", wind_pred.min(), wind_pred.max())
        print("Wind real min:", wind_real.min(), wind_real.max())
        nlat = temp_pred.shape[1]
        nlon = temp_pred.shape[2]
        lat_vals = np.linspace(-90, 90, nlat)
        lon_vals = np.linspace(-180, 180, nlon)

        
        N = min(temp_pred.shape[0], 24)

        
        compute_animation_for_scalar(
            true_data=temp_real[:N],
            predicted_data=temp_pred[:N],
            lat=lat_vals,
            lon=lon_vals,
            title="Temperature (°C)",
            save_path="val_temperature_prediction.mp4",
            year=year,
            fps=fps
        )

        compute_animation_for_temperature_difference(
            true_data=temp_real[:N],
            predicted_data=temp_pred[:N],
            lat=lat_vals,
            lon=lon_vals,
            title="Temperature (°C)",
            save_path="val_temperature_difference.mp4",
            year=year,
            fps=fps
        )

        
        compute_animation_for_vector(
            true_vector_data=wind_real[:N],
            predicted_vector_data=wind_pred[:N],
            lat=lat_vals,
            lon=lon_vals,
            title="Wind (m/s)",
            save_path="val_wind_prediction.mp4",
            year=year,
            fps=fps
        )

        compute_animation_for_vector_difference(
            true_vector_data=wind_real[:N],
            predicted_vector_data=wind_pred[:N],
            lat=lat_vals,
            lon=lon_vals,
            title="Wind (m/s)",
            save_path="val_wind_difference.mp4",
            year=year,
            fps=fps
        )

        
        
        temp_incert_pred = torch.cat(all_incert_temp_pred, dim=0).cpu().numpy()
        temp_incert_real = torch.cat(all_incert_temp_real, dim=0).cpu().numpy()
        
        temp_incert_pred = (temp_incert_pred - np.mean(temp_incert_pred)) / np.std(temp_incert_pred)
        temp_incert_real = (temp_incert_real - np.mean(temp_incert_real)) / np.std(temp_incert_real)
        compute_animation_for_scalar(
            true_data=temp_incert_real[:N],
            predicted_data=temp_incert_pred[:N],
            lat=lat_vals,
            lon=lon_vals,
            title="Temperature Uncertainty",
            save_path="val_temperature_uncertainty.mp4",
            year=year,
            fps=fps
        )

        
        wind_incert_pred = torch.cat(all_incert_wind_pred, dim=0).cpu().numpy()
        wind_incert_real = torch.cat(all_incert_wind_real, dim=0).cpu().numpy()
        wind_incert_pred = (wind_incert_pred - np.mean(wind_incert_pred)) / np.std(wind_incert_pred)
        wind_incert_real = (wind_incert_real - np.mean(wind_incert_real)) / np.std(wind_incert_real)
        compute_animation_for_scalar(
            true_data=wind_incert_real[:N],
            predicted_data=wind_incert_pred[:N],
            lat=lat_vals,
            lon=lon_vals,
            title="Wind Uncertainty",
            save_path="val_wind_uncertainty.mp4",
            year=year,
            fps=fps
        )
        
        wandb.log({
            "Val/Temperature Prediction": wandb.Video("val_temperature_prediction.mp4", fps=fps),
            "Val/Temperature Difference": wandb.Video("val_temperature_difference.mp4", fps=fps),
            "Val/Wind Prediction": wandb.Video("val_wind_prediction.mp4", fps=fps),
            "Val/Wind Difference": wandb.Video("val_wind_difference.mp4", fps=fps),
            "Val/Temperature Uncertainty": wandb.Video("val_temperature_uncertainty.mp4", fps=fps),
            "Val/Wind Uncertainty": wandb.Video("val_wind_uncertainty.mp4", fps=fps),
        })
    gen.train()
    return avg_mse, (example_x, example_y, example_fake)



def train_wgangp_conditional(
    gen,
    disc,
    loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    lr_discr=1e-4,
    lambda_gp=10.0,
    alpha_nll=1.0,
    gamma_clip=1.0,
    visual_interval=1,
    save_dir="checkpoints",
    use_recon=False,
    alpha_recon=1.0,
    initial_step=0,
    start_epoch=0,
    opt_gen_state=None,
    opt_disc_state=None
):
    os.makedirs(save_dir, exist_ok=True)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr_discr, betas=(0.5, 0.9))

    if opt_gen_state is not None:
        opt_gen.load_state_dict(opt_gen_state)
    if opt_disc_state is not None:
        opt_disc.load_state_dict(opt_disc_state)

    gen.train()
    disc.train()

    global_step = initial_step

    for epoch in range(start_epoch, num_epochs):
        epoch_disc_loss = 0.0
        epoch_gen_adv   = 0.0
        epoch_gen_nll   = 0.0
        epoch_gen_recon = 0.0
        epoch_gp        = 0.0

        for batch_data in tqdm(loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", total=len(loader)):
            global_step += 1

            inputs = batch_data["input"].to(device)
            mask = batch_data["masks"].to(device)
            lat = batch_data["coords"][0].unsqueeze(1).to(device)
            lon = batch_data["coords"][1].unsqueeze(1).to(device)
            coords = torch.concat([lat, lon], 1)
            x = torch.cat([inputs, mask, coords], dim=1) 
            y = batch_data["target"].to(device)
            
            x = x.permute(0, 3, 2, 1)  
            y = y.permute(0, 3, 2, 1) 

            fake = gen.sample(x).detach()
            disc_real = disc(x, y)
            disc_fake = disc(x, fake)
            loss_d = -(disc_real.mean() - disc_fake.mean())

            gp = gradient_penalty_conditional(disc, x, y, fake, device, lambda_gp=lambda_gp)
            loss_d_total = loss_d + gp

            opt_disc.zero_grad()
            loss_d_total.backward()
            utils.clip_grad_norm_(disc.parameters(), gamma_clip)
            opt_disc.step()

            epoch_disc_loss += loss_d.item()
            epoch_gp += gp.item()

            fake2 = gen.sample(x)
            disc_fake2 = disc(x, fake2)
            loss_g_adv = -disc_fake2.mean()

            lp_pixelwise = gen.log_prob(y, x)
            
            loss_nll = -(lp_pixelwise.mean())  

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

            logs = {
                "Train/Discriminator": loss_d.item(),
                "Train/GP": gp.item(),
                "Train/Generator_adv": loss_g_adv.item(),
                "Train/Generator_NLL": loss_nll.item(),
            }
            if use_recon:
                logs["Train/Generator_Recon"] = loss_recon.item()

            with torch.no_grad():
                real_mean = y.mean(dim=(1,2,3))
                real_var  = y.var(dim=(1,2,3))
                fake_mean = fake2.mean(dim=(1,2,3))
                fake_var  = fake2.var(dim=(1,2,3))

                mean_diff = (real_mean - fake_mean).abs().mean()
                var_diff  = (real_var - fake_var).abs().mean()

            logs["Dist/Mean_diff"] = mean_diff.item()
            logs["Dist/Var_diff"]  = var_diff.item()

            wandb.log(logs, step=global_step)
            


        nb_batches = len(loader)
        avg_disc_loss = epoch_disc_loss / nb_batches
        avg_gp        = epoch_gp / nb_batches
        avg_gen_adv   = epoch_gen_adv / nb_batches
        avg_gen_nll   = epoch_gen_nll / nb_batches
        avg_gen_recon = (epoch_gen_recon / nb_batches) if use_recon else 0.0

        print(
            f"Epoch[{epoch+1}/{num_epochs}] "
            f"D_loss={avg_disc_loss:.4f}, GP={avg_gp:.4f}, "
            f"G_adv={avg_gen_adv:.4f}, G_nll={avg_gen_nll:.4f}, "
            f"G_recon={avg_gen_recon:.4f}"
        )

        epoch_logs = {
            "Epoch/D_loss": avg_disc_loss,
            "Epoch/GP": avg_gp,
            "Epoch/G_adv": avg_gen_adv,
            "Epoch/G_nll": avg_gen_nll,
        }
        if use_recon:
            epoch_logs["Epoch/G_recon"] = avg_gen_recon

        wandb.log(epoch_logs, step=global_step)

        if val_loader is not None:
            video = (epoch+1)%5 == 0
            val_mse, (val_x, val_y, val_fake) = validate(
                gen, 
                val_loader, 
                device,
                log_wandb=True, 
                generate_video=video,
                fps=48
            )
            global_step += 1
            wandb.log({"Val/MSE": val_mse}, step=global_step)

            real_2d = val_y[0, :, :, 0].numpy()
            fake_2d = val_fake[0, :, :, 0].numpy()
            fig, axs = plt.subplots(1, 2, figsize=(10,4))
            axs[0].imshow(real_2d.T, aspect='auto', origin='lower')
            axs[0].set_title("Val: Real (ch=0)")
            axs[1].imshow(fake_2d.T, aspect='auto', origin='lower')
            axs[1].set_title("Val: Fake (ch=0)")
            plt.tight_layout()
            print("Logging Val/Real_vs_Fake")
            wandb.log({"Val/Real_vs_Fake": fig_to_wandb_image(fig)}, step=global_step)
            plt.close(fig)

        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'wandb_run_id': wandb.run.id,
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    print("Training finished.")

def find_last_checkpoint(save_dir):
    """
    Searches for the latest checkpoint (highest epoch) in save_dir.
    Returns (checkpoint_path, epoch) or (None, 0) if none found.
    """
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        return None, 0

    max_epoch = 0
    last_ckpt_path = None
    for ckpt_path in checkpoint_files:
        match = re.search(r"checkpoint_epoch_(\d+)\.pth", ckpt_path)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                last_ckpt_path = ckpt_path

    return last_ckpt_path, max_epoch

def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_ckpt_path, last_ckpt_epoch = find_last_checkpoint(args.save_dir)
    start_epoch = 0
    global_step = 0
    wandb_run_id = None

    checkpoint_opt_gen = None
    checkpoint_opt_disc = None

    if last_ckpt_path:
        print(f"==> Found checkpoint at {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        wandb_run_id = checkpoint.get('wandb_run_id', None)

        checkpoint_opt_gen = checkpoint.get('opt_gen_state_dict', None)
        checkpoint_opt_disc = checkpoint.get('opt_disc_state_dict', None)

        print(f"==> Resuming training from epoch {start_epoch}, global_step {global_step}")

    if wandb_run_id is not None:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            id=wandb_run_id,
            resume="allow"
        )
    else:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

    dataset_dir = "/home/ensta/ensta-cesar/era_5_data/"
    datasets = load_dataset(
        nb_file=10,
        train_val_split=0.8,
        year0=1979,
        root_dir=dataset_dir,
        normalize=True
    )
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=5)

    

    gen = ConditionalFlowGenerator2d(
        context_channels=7,
        latent_channels=3,
        num_flows=16
    ).to(device)
    
    disc = ConditionalWGANGPDiscriminator2d(
        in_channels_x=7,
        in_channels_y=3
    ).to(device)

    if last_ckpt_path:
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])

    train_wgangp_conditional(
        gen=gen,
        disc=disc,
        loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lr_discr=args.lr_discr,
        lambda_gp=args.lambda_gp,
        alpha_nll=args.alpha_nll,
        gamma_clip=args.gamma_clip,
        visual_interval=args.visual_interval,
        save_dir=args.save_dir,
        use_recon=args.use_recon,
        alpha_recon=args.alpha_recon,
        initial_step=global_step,
        start_epoch=start_epoch,
        opt_gen_state=checkpoint_opt_gen,
        opt_disc_state=checkpoint_opt_disc
    )

    wandb.finish()

if __name__ == "__main__":
    main()
