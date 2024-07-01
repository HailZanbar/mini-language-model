import torch
import attention

def test_attention_scores():
    print("Attention test")
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]) # a three-dim tensor
    b = torch.tensor([[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]) # a three-dim tensor
    expected_output_1 = torch.full((2, 3, 3), 15.0) # a three-dim tensor

    # c = torch.tensor([]) # a three-dim tensor
    # d = torch.tensor([]) # a three-dim tensor
    # expected_output_2 = torch.tensor([]) # a three-dim tensor

    A = attention.attention_scores(a, b)
    print(A)
    #B = attention.attention_scores(c, d)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output_1)
    #assert torch.allclose(B, expected_output_2)

test_attention_scores()
