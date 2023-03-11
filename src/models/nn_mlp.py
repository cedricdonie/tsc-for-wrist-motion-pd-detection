# Neural network models for analysis
# Author(s): Neha Das (neha.das@tum.de)

import sys
import torch
from torch.distributions import Normal
from attrdict import AttrDict
from torch.utils.data import DataLoader
from os.path import dirname, abspath

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)


class MLPBase(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        act=torch.nn.ReLU(inplace=True),
        use_batch_norm=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.act = act

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                if self.use_batch_norm:
                    bn_layer = torch.nn.BatchNorm1d(num_features=fc.out_channels)
                    if torch.cuda.is_available():
                        bn_layer = bn_layer.cuda()
                    x = bn_layer(x)
                x = self.act(x)

        return x


class MLPModel(MLPBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        output_range,
        act=torch.nn.ReLU(inplace=True),
        use_batch_norm=False,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            act=act,
            use_batch_norm=use_batch_norm,
        )
        self.output_range = output_range

    def forward(self, x):
        x = super().forward(x)
        x = (
            torch.nn.Sigmoid()(x) * (self.output_range[1] - self.output_range[0])
            + self.output_range[0]
        )
        return x


class MLPClassModel(MLPBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        act=torch.nn.ReLU(inplace=True),
        use_batch_norm=False,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            act=act,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.Softplus()(x)
        return x


class MLPOrdinalModel(MLPBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        act=torch.nn.ReLU(inplace=True),
        use_batch_norm=False,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            act=act,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.Sigmoid()(x)
        return x


class GaussianMLPModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        act=torch.nn.ReLU(inplace=True),
        scalar_noise=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.act = act

        dims = [input_dim] + list(hidden_dims)  # + [output_dim]
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.mean = torch.nn.Linear(dims[-1], output_dim)
        if scalar_noise:
            self.std = torch.nn.Linear(dims[-1], 1)
        else:
            self.std = torch.nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            x = self.act(x)

        mean = self.mean(x)
        std = self.std(x)

        return Normal(mean, std)


if __name__ == "__main__":
    # Test Models
    import numpy as np
    from datasets.raw_dataset import RawShimmerPDDataset
    from utils.training import get_model
    from utils.dataset import get_dataset_train_test_files

    train_ids = np.arange(3, 14, 1)
    test_ids = np.arange(14, 20, 1)
    train_files, test_files = get_dataset_train_test_files(
        train_sub_ids=train_ids, test_sub_ids=test_ids, window_sz=30, overlap_sec=29
    )
    dataset = RawShimmerPDDataset(
        signal_source=["leftWrist_Magnitude", "back_Magnitude"],
        output_source=["bradykinesia_LeftLowerLimb"],
        file_list=train_files,
        is_binary=False,
        summarize="catch22",
    )
    # import pdb; pdb.set_trace()
    x, y = dataset.get_a_random_item()

    x = torch.unsqueeze(x, dim=0)

    print(x.shape, y.shape)

    model = get_model(
        network_name="MLPClassModel",
        cfg=AttrDict({"hidden_dims": [128, 128], "use_batch_norm": False}),
        dataset=dataset,
        is_dynamic=False,
        num_classes=5,
        original_target_range=[0, 4],
    )

    op = model(x)  # Reshaped to (batch_sz, -1)
    print(op.shape)
