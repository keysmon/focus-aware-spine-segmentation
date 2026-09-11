import torch
from experiments.dino_decoder import DenseDecoder
from experiments.dino_dropout import DropoutDecoder


def test_dropout_preserves_initial_parameters_and_deterministic_evaluation():
    torch.set_num_threads(2)
    torch.manual_seed(435);base=DenseDecoder()
    torch.manual_seed(435);model=DropoutDecoder()
    for name,tensor in base.state_dict().items():
        torch.testing.assert_close(tensor,model.state_dict()[name],rtol=0,atol=0)
    features=torch.randn(1,384,3,4);image=torch.rand(1,1,19,23)
    base.eval();model.eval()
    with torch.no_grad():
        torch.testing.assert_close(base(features,image),model(features,image),rtol=0,atol=0)
        torch.testing.assert_close(model(features,image),model(features,image),rtol=0,atol=0)
        model.train()
        a=model(features,image);b=model(features,image)
        assert not torch.equal(a,b)
