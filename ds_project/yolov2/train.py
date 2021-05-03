def train(loader, optimizer, model, device, EPOCHS, criterion):
    """
    Train loop

    Args:
        loader (Dataloader): torch dataloader
        optimizer (torch.optim): optimizer for model. Like SGD, Adam, etc.
        model (nn.Module): torch model class
        device (string): CPU or CUDA
        EPOCHS (int): number of epoches to learn
        criterion (loss._Loss): torch task specific loss
    """
    num_iter_per_epoch = len(loader)
    for epoch in range(EPOCHS):
        loss = []
        loss_coord = []
        loss_conf = []
        loss_cls = []
        for i, batch in enumerate(loader):
            image, label = batch
            image = image.to(device)
            
            optimizer.zero_grad()
            logits = model(image)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
            loss.backward()
            optimizer.step()
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                EPOCHS,
                i + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                loss_coord,
                loss_conf,
                loss_cls))