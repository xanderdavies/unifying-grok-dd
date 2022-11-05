def add_tags(*args_and_tags):
    tags = []
    name = ""
    for tag, condition in args_and_tags:
        if condition:
            tags.append(tag)
            name += tag
    return name, tags

def print_examples(train_dataloader, val_dataloader, train_dataset, val_dataset, model, train=True):
    for data in train_dataloader if train else val_dataloader:
        X, y = data["text"], data["target"]
        for i in range(10):
            print(train_dataset.tokenizer.decode(y[i]),train_dataset.tokenizer.itos[(model(X)[i].argmax(dim=1)[-2])])
        break
