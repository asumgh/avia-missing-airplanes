import numpy as np
from tqdm import tqdm_notebook
import torch
from sklearn.metrics import accuracy_score
def train_model(model, device, loss_fn, optimizer, train_loader, val_loader, num_epoch, DECAY, n_fold):
    train_losses = []
    test_losses = []
    acc = []
    print(model)

    for i in range(num_epoch):
        epoch_train_losses = []
        model.train(True)
        for X_train, y_train in tqdm_notebook(train_loader):
            # Посчитаем предсказание и лосс
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            del y_pred

            # зануляем градиент
            optimizer.zero_grad()

            # backward
            loss.backward()

            # ОБНОВЛЯЕМ веса
            optimizer.step()

            # Запишем число (не тензор) в наши батчевые лоссы
            epoch_train_losses.append(loss.item())   
                    
        train_losses.append(np.mean(epoch_train_losses))
        
        # Теперь посчитаем лосс на вал
        with torch.no_grad():
            model.eval()
            epoch_test_losses = []
            epoch_acc = []
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                loss = loss_fn(y_pred, y_val)
            
                epoch_test_losses.append(loss.item())
                y_pred = y_pred.sigmoid().detach().cpu().numpy()
                y_pred = (y_pred>=0.5).astype(int)
                epoch_acc.append(accuracy_score(y_val.cpu(), y_pred))
                del y_pred

            test_losses.append(np.mean(epoch_test_losses))
            acc.append(np.mean(epoch_acc))
            
            torch.save(model.state_dict(), f'epoch_{i}_fold_{n_fold}.pth')  # сохраняем веса эпох

            print(
                'Train loss =', train_losses[-1],
                'Val loss =', test_losses[-1],
                'Val accuracy score =', acc[-1]
            )
        if i == 5:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*DECAY
                
    return train_losses, test_losses, acc
    

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs in dataloader:

            outputs = model(inputs.to(device))
            y_pred = outputs.sigmoid().detach().cpu().numpy()

            preds.append(y_pred)

    preds = np.concatenate(preds)
    
    return preds