from data.distributions.lid_base import LIDDistribution

from .lid import LIDDataset

LID_SYNTH_SEED = 42


class LIDSyntheticDataset(LIDDataset):
    """This is a dataset that uses a distribution to generate synthetic data for LID estimation."""

    def __init__(
        self,
        size: int,
        distribution: LIDDistribution,
        seed: int | None = None,
        standardize: bool = False,
        **sampling_kwargs
    ):
        ret = distribution.sample(
            (size,),
            return_dict=True,
            seed=LID_SYNTH_SEED if seed is None else seed,
            **sampling_kwargs
        )
        x = ret["samples"]
        super().__init__(
            x.numel() // x.shape[0],
            x=x,
            lid=ret["lid"],
            idx=ret["idx"],
            standardize=standardize,
        )
