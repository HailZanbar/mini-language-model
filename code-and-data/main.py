from __future__ import annotations
import torch
import time
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# functions to save losses by epoch in file
def init_result_file(path, params):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        write_params = '\t'.join([f"{key}: {value}" for key, value in params.items()])
        f.write(write_params + '\n')

def add_results(path, loss_values):
    with open(path, 'a') as f:
        for val in loss_values:
            f.write(str(val) + '\n')



if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from transformer import TransformerLM
    import data
    import lm

    seq_len = 128
    batch_size = 64
    data_path = "data/"
    n_layers = 15
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 1e-3  # 5e-4
    gradient_clipping = 1.0
    dropout = True

    params = {  # params to be played with
        'batch_size': batch_size,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'embed_size': embed_size,
        'learning_rate': learning_rate,
        'dropout': dropout
    }

    num_batches_to_train = 50000  # 50000

    tokenizer, tokenized_data = data.load_data(data_path)
    pad_id = tokenizer.pad_id()
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model_path = "transformer_lm.pth"
    losses_path = f"results/{n_layers}_layers_{n_heads}_heads_losses.txt"
    text_path = f'results/{n_layers}_layers_{n_heads}_heads_snippet.txt'

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = True,
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    start_time = time.time()
    init_result_file(losses_path, params)
    losses = []

    model.train()
    
    num_batches = 0
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train: break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            # Move batch_x and batch_y to GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y, pad_id)
            
            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                losses.append(loss.item())

            if num_batches % 1000 == 0:
                for _ in range(1):
                    model.eval()
                    sampled = tokenizer.detokenize(model.better_sample_continuation(tokenizer.tokenize("Hello"), 500, temperature=0.7, topK=5))
                    model.train()
                    print(f"Model sample: '''{sampled}'''")
                    # save snippet to file:
                    with open(text_path, 'w') as file:
                        file.write(sampled)
                print("")

                # print some time details
                curr_time = time.time()
                until_now = curr_time - start_time
                batches_to_sec = num_batches / until_now
                print(f"Time until now: {int(until_now//60):02}:{int(until_now%60):02}. Average batches per second: {round(batches_to_sec, 3)}")

                # save losses to file (append only recent losses)
                add_results(losses_path, losses)
                losses = []

            
            # # Stop the training and save the trained model
            # if loss.item() <= 0.4:
            #     model.save_model(model_path)
            #     print("Model saved to checkpoint.")
            #     break
        break


