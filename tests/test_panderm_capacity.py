import torch
from src.models.panderm_capacity import ResidualHead


def test_zero_correction_preserves_linear_predictions_and_training_keeps_base_fixed():
    torch.manual_seed(4)
    coef=torch.randn(7,1024);bias=torch.randn(7)
    model=ResidualHead(coef.numpy(),bias.numpy(),[8,4],.25)
    x=torch.randn(12,1024);y=torch.arange(12)%7
    model.eval()
    torch.testing.assert_close(model(x),x@coef.T+bias)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
    torch.testing.assert_close(model.linear.weight,coef,rtol=0,atol=0)
    torch.testing.assert_close(model.linear.bias,bias,rtol=0,atol=0)
    assert not torch.equal(model(x),x@coef.T+bias)
