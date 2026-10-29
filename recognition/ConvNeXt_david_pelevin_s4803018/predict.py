from sklearn.metrics import roc_auc_score
import dataset, modules
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_split, val_split = dataset.train_val_split("/content/drive/My Drive/AD_NC/train", "AD", "NC")

train_loader = DataLoader(dataset.ADNI_train("./ADNI/AD_NC/train", "AD", "NC", train_split), batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/train", "AD", "NC", val_split), batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.ADNI_test("./ADNI/AD_NC/test", "AD", "NC"), batch_size=32, shuffle=False)

check_model = modules.ConvNeXt()
check_model.to(device)
check_model.load_state_dict(torch.load('./model_weights/convnext.pth', weights_only=True, map_location=device))

check_model.eval()

for loader in [train_loader, val_loader, test_loader]:
    correct, tp, fp, tn, fn, total = 0, 0, 0, 0, 0, 0
    all_preds, all_labels = 0, 0
    init_tensor = False

    tp_x = []
    tp_y = []
    fp_x = []
    fp_y = []
    tn_x = []
    tn_y = []
    fn_x = []
    fn_y = []

    with torch.no_grad():
        for images, labels in loader:
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

            tp_tensor = torch.logical_and(matching, ad_predicted)
            tn_tensor = torch.logical_and(matching, nc_predicted)
            fp_tensor = torch.logical_and(not_matching, ad_predicted)
            fn_tensor = torch.logical_and(not_matching, nc_predicted)

            tp += tp_tensor.sum().item()
            tn += tn_tensor.sum().item()
            fp += fp_tensor.sum().item()
            fn += fn_tensor.sum().item()

            total += labels.size(0)

            if init_tensor:
                all_preds = torch.cat([all_preds, preds], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)
            else:
                init_tensor = True
                all_preds = preds
                all_labels = labels

            for i in range(outputs.size(dim=0)):
                if tp_tensor[i].item():
                    tp_x.append(outputs[i][0].item())
                    tp_y.append(outputs[i][1].item())
                elif tn_tensor[i].item():
                    tn_x.append(outputs[i][0].item())
                    tn_y.append(outputs[i][1].item())
                elif fp_tensor[i].item():
                    fp_x.append(outputs[i][0].item())
                    fp_y.append(outputs[i][1].item())
                elif fn_tensor[i].item():
                    fn_x.append(outputs[i][0].item())
                    fn_y.append(outputs[i][1].item())

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

    plt.title("True Positive")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer’s Disease")
    plt.scatter(tp_x, tp_y, c="blue", alpha=0.05)
    plt.show()

    plt.title("True Negative")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer’s Disease")
    plt.scatter(tn_x, tn_y, c="green", alpha=0.05)
    plt.show()

    plt.title("False Positive")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer’s Disease")
    plt.scatter(fp_x, fp_y, c="yellow", alpha=0.05)
    plt.show()

    plt.title("False Negative")
    plt.xlabel("Logit for Cognitively Normal")
    plt.ylabel("Logit for Alzheimer’s Disease")
    plt.scatter(fn_x, fn_y, c="red", alpha=0.05)
    plt.show()