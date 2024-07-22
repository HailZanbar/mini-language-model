from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len).to(device)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size).to(device)
        self.layer_norm_1 = nn.LayerNorm(embed_size).to(device)
        self.layer_norm_2 = nn.LayerNorm(embed_size).to(device)
        self.with_residuals = with_residuals

    def forward(self, inputs):
        if self.with_residuals:
            x = inputs

            residual = x
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = x + residual  # adding prev value

            residual = x
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            x = x + residual  # adding prev value

            return x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size).to(device)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size).to(device)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        tok_embeddings = self.token_embeddings(x)
        
        b, n = x.shape
        positions = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        pos_embeddings = self.position_embeddings(positions)

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len).to(device)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals).to(device) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size).to(device)
        self.word_prediction = nn.Linear(embed_size, vocab_size).to(device)
        self.max_context_len = max_context_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.mlp_hidden_size = mlp_hidden_size
        self.with_residuals = with_residuals


        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        # initialize weights
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for pn, p in self.named_parameters():
            if 'bias' in pn:
                torch.nn.init.zeros_(p)  # all biases start with 0
            elif 'layer_norm' in pn:
                torch.nn.init.ones_(p)
            elif 'embed' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(p)  # weights of linear layers

    def sample_continuation(self, prefix: "list[int]", max_tokens_to_generate: int) -> "list[int]":
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated


    def sample_from_topK(self, distribution, k, num_samples):
        top_k_probs, top_k_indices = torch.topk(distribution, k, dim=-1)

        # Sample from the top-k probabilities
        samples = torch.multinomial(top_k_probs, num_samples)

        # Convert the indices of the top-k elements back to the original indices
        final_samples = top_k_indices.gather(dim=-1, index=samples)

        return final_samples
    

    def better_sample_continuation(self, prefix: "list[int]", max_tokens_to_generate: int, temperature: float, topK: int) -> "list[int]":
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.

        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(device))
                logits_for_last_token = logits[0][-1]

                # Add temperature
                distribution_for_last_token = F.softmax(logits_for_last_token / temperature, dim=-1)

                # Sample from topK probs
                sampled_token = self.sample_from_topK(distribution_for_last_token, topK, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated
    

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'embed_size': self.embed_size,
            'max_context_len': self.max_context_len,
            'vocab_size': self.vocab_size,
            'mlp_hidden_size': self.mlp_hidden_size,
            'with_residuals': self.with_residuals
        }, path)

    @classmethod
    def load_model(cls, path, device):
        if device == torch.device('cpu'):
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        model = cls(
            n_layers=checkpoint['n_layers'],
            n_heads=checkpoint['n_heads'],
            embed_size=checkpoint['embed_size'],
            max_context_len=checkpoint['max_context_len'],
            vocab_size=checkpoint['vocab_size'],
            mlp_hidden_size=checkpoint['mlp_hidden_size'],
            with_residuals=checkpoint['with_residuals'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    


