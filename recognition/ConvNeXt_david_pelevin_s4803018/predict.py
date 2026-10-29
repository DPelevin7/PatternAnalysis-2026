from sklearn.metrics import roc_auc_score
import dataset, modules
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/test", "AD", "NC"), batch_size=32, shuffle=False)

check_model = modules.ConvNeXt()
check_model.to(device)
check_model.load_state_dict(torch.load('./model_weights/convnext.pth', weights_only=True, map_location=device))

check_model.eval()
correct, tp, fp, tn, fn, total = 0, 0, 0, 0, 0, 0
all_preds, all_labels = 0, 0
init_tensor = False
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = check_model(images)
        preds = outputs.argmax(dim=1)

        one = torch.ones(preds.size())
        one = one.to(device)

        matching = (preds == labels)
        not_matching = (preds != labels)
        ad_predicted = (preds == one)
        nc_predicted = (preds != one)

        correct += matching.sum().item()

        tp += torch.logical_and(matching, ad_predicted).sum().item()
        tn += torch.logical_and(matching, nc_predicted).sum().item()
        fp += torch.logical_and(not_matching, ad_predicted).sum().item()
        fn += torch.logical_and(not_matching, nc_predicted).sum().item()

        total += labels.size(0)

        if init_tensor:
            all_preds = torch.cat([all_preds, preds], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)
        else:
            init_tensor = True
            all_preds = preds
            all_labels = labels

precision_ad = tp / (tp + fp)
precision_nc = tn / (tn + fn)
recall_ad = tp / (tp + fn)
recall_nc = tn / (tn + fp)
f1_ad = 2 * precision_ad * recall_ad / (precision_ad + recall_ad)
f1_nc = 2 * precision_nc * recall_nc / (precision_nc + recall_nc)
macro_f1 = (0.5 * f1_ad) + (0.5 * f1_nc)

all_labels = all_labels.to('cpu')
all_preds = all_preds.to('cpu')
auroc = roc_auc_score(all_labels, all_preds)

print(f"Test accuracy: {correct/total:.4f}")
print(f"{tp}, {fp}, {tn}, {fn}, {(tp + tn) / (tp + tn + fp + fn):.4f}")
print(f"{precision_ad}, {precision_nc}, {recall_ad}, {recall_nc}, {f1_ad}, {f1_nc}, {macro_f1}, {auroc}")