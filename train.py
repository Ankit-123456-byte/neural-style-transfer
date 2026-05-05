import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import VGGEncoder, Decoder, NSTModel
from dataset import StyleTransferDataset
from utils.utils import save_tensor, save_checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train AdaIN Neural Style Transfer")

    parser.add_argument("--content_dir", type=str, default="content_data")
    parser.add_argument("--style_dir",   type=str, default="style_data")
    parser.add_argument("--vgg",         type=str, default="vgg_normalised.pth",
                        help="Path to pre-trained normalised VGG weights")
    parser.add_argument("--experiment",  type=str, default="experiment1")

    parser.add_argument("--image_size",    type=int,   default=512)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--content_weight", type=float, default=1.0,
                        help="Weight for content loss")
    parser.add_argument("--style_weight",   type=float, default=10.0,
                        help="Weight for style loss")
    parser.add_argument("--alpha",         type=float, default=1.0,
                        help="Style strength: 0=content only, 1=full style")
    parser.add_argument("--save_every",    type=int,   default=5,
                        help="Save sample images every N epochs")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--augment",       action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Apply data augmentation during training (--no-augment to disable)")
    parser.add_argument("--mode",          type=str, default="adain",
                        choices=["adain", "wct"],
                        help="Style transfer mode: adain (fast) or wct (full covariance, slower)")
    return parser.parse_args()


def train():
    args = parse_arguments()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(save_dir / "args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # ------------------------------------------------------------------ model
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    model   = NSTModel(encoder, decoder).to(device)

    optimizer = Adam(decoder.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ----------------------------------------------------------------- data
    dataset = StyleTransferDataset(
        args.content_dir, args.style_dir,
        args.image_size, augment=args.augment,
    )
    pin_memory = device.type == "cuda"
    num_workers = 0 if device.type == "mps" else 4  # MPS + multiprocessing causes deadlocks on macOS
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Dataset: {len(dataset)} content images, "
          f"{len(dataset.style_paths)} style images  "
          f"(augment={args.augment})")

    # --------------------------------------------------------------- training
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_c, total_s = 0.0, 0.0

        for step, (content, style) in enumerate(loader):
            content, style = content.to(device), style.to(device)

            generated, loss_c, loss_s = model(
                content, style,
                alpha=args.alpha,
                content_weight=args.content_weight,
                style_weight=args.style_weight,
                mode=args.mode,
            )

            loss = loss_c + loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_c += loss_c.item()
            total_s += loss_s.item()

            if step % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{step}/{len(loader)}] "
                      f"Content: {loss_c.item():.4f}  "
                      f"Style: {loss_s.item():.4f}  "
                      f"Total: {loss.item():.4f}")

        scheduler.step()
        avg_c = total_c / len(loader)
        avg_s = total_s / len(loader)
        avg_total = avg_c + avg_s
        print(f"--- Epoch {epoch+1} | Content Loss: {avg_c:.4f} | "
              f"Style Loss: {avg_s:.4f} | Total: {avg_total:.4f} ---")

        # Track and save best model
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(decoder.state_dict(), str(save_dir / "decoder_best.pth"))
            print(f"    New best loss {best_loss:.4f} — saved decoder_best.pth")

        # Save sample outputs and periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_tensor(
                torch.cat([content[:4], style[:4], generated[:4]], dim=0),
                str(save_dir / f"samples_epoch_{epoch+1:04d}.png"),
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_c": avg_c,
                    "loss_s": avg_s,
                    "mode": args.mode,
                },
                str(save_dir / f"decoder_epoch_{epoch+1:04d}.pth"),
            )
            print(f"Saved checkpoint and samples for epoch {epoch+1}")

    # Final model save
    torch.save(decoder.state_dict(), str(save_dir / "decoder_final.pth"))
    print(f"Training complete. Model saved to {save_dir}/decoder_final.pth")


if __name__ == "__main__":
    train()
