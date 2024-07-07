import torch
import attention

def test_attention_scores():
    print("Attention test")
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]) # a three-dim tensor
    b = torch.tensor([[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]) # a three-dim tensor
    expected_output_1 = torch.full((2, 3, 3), 15.0) # a three-dim tensor

    A = attention.attention_scores(a, b)
    print(A)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output_1)

# Works only without softmax in self_attention in with convertion of -inf to 0.
def test_self_attention():
    v = torch.tensor([[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,3]]], dtype=torch.float32) # a three-dim tensor
    A = torch.tensor([[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]], dtype=torch.float32) # a three-dim tensor
    expected_output = torch.tensor([[[1,2,3],[3,6,9],[6,12,18],[10,20,30]],[[1,2,3],[3,6,9],[6,12,18],[10,20,30]]], dtype=torch.float32)

    mask = attention.create_causal_mask(0, 0, 5)

    sa = attention.self_attention(v, A, mask)
    print(sa)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(sa, expected_output)



#test_attention_scores()
#test_self_attention()
