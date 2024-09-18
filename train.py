
import torch
import yaml
from tqdm import tqdm
from nets.rtdetrv2 import get_model
from utils.postprocessor import RTDETRPostProcessor
from utils.criterion import get_criterion
from utils.warmup import LinearWarmup
from data.dataset import get_dataloaders
from utils import util, plots



def train(args,params):
    start_epoch = params["start_epoch"]

    model = get_model(args,params)
    model.to(args.device)
    train_dataloader, val_dataloader = get_dataloaders(args)
    criterion = get_criterion(params)
    criterion.to(args.device)
    postprocessor = RTDETRPostProcessor(
        num_classes=params["num_classes"],
        use_focal_loss=params["use_focal_loss"],
        num_top_queries=params["RTDETRPostProcessor"]["num_top_queries"],
    )

    # Optimizer
    opt_params = params["optimizer"]
    model_params = util.get_optim_params(opt_params, model)
    optimizer = torch.optim.AdamW(params=model_params, lr=opt_params["lr"], betas=opt_params["betas"], weight_decay=opt_params["weight_decay"])
    
    #Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, 
        milestones=params["lr_scheduler"]['milestones'], 
        gamma=params["lr_scheduler"]['gamma'], 
    )
    lr_warmup_scheduler = LinearWarmup(
        lr_scheduler=lr_scheduler, 
        warmup_duration=int(args.epochs * 0.1), 
    )

    amp_scale = torch.amp.GradScaler()


    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()
        p_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        avg_loss = 0.
        for i, (samples, targets) in p_bar:
            samples = samples.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            global_step = epoch * len(train_dataloader) + i
            metas = dict(epoch=epoch, step=i, global_step=global_step)
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            loss: torch.Tensor = sum(loss_dict.values())
            amp_scale.scale(loss).backward()
            
            amp_scale.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_max_norm'])  # clip gradients
            amp_scale.step(optimizer)  # optimizer.step
            amp_scale.update()
            
            optimizer.zero_grad()

            if not lr_warmup_scheduler.finished():
                lr_warmup_scheduler.step()
            else:
                lr_scheduler.step()

            avg_loss += loss.item()

        p_bar.set_description(
            f"Training : [Epoch {epoch}/{args.epochs}] "
            f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
            f"[average loss: {avg_loss/len(train_dataloader):.4f}]"
        )

        ### validation loop ###
        model.eval()
        criterion.eval()
        if epoch % 5:
            with torch.no_grad():
                epoch_val_loss = 0.0
                p_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
                for i, (samples, targets) in p_bar:

                    samples = samples.to(args.device)
                    targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

                    global_step = epoch * len(val_dataloader) + i
                    metas = dict(epoch=epoch, step=i, global_step=global_step)
                    outputs = model(samples, targets=targets)
                    loss_dict = criterion(outputs, targets, **metas)
                    loss: torch.Tensor = sum(loss_dict.values())

                    epoch_val_loss += loss.item()

                    # Print log
                    p_bar.set_description(
                        "Validating : [Epoch %d/%d] [Batch %d/%d] [lr %f] [total_loss: %f]"
                        % (
                            epoch,
                            args.epochs,
                            i,
                            len(val_dataloader),
                            optimizer.param_groups[0]["lr"],
                            loss.item(),
                        )
                    )
                samples = samples.to(args.device)
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
                outputs = model(samples, targets=targets)
                processed = postprocessor(outputs, torch.tensor(samples.shape[-2:]).to(args.device))
                img = plots.visualize_bboxes(samples.cpu(),targets,processed)
                epoch_val_loss /= len(val_dataloader)


if __name__ == "__main__":
    
    args = util.parse_args()

    with open(args.config, "r") as file:
        params = yaml.safe_load(file)
    util.setup_seed()
    if args.train:
        train(args, params)
