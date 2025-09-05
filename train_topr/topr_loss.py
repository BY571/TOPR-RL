# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import torch
from tensordict import (
    is_tensor_collection,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    composite_lp_aggregate,
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl._utils import _standardize, logger as torchrl_logger, VERBOSE
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _maybe_add_or_extend_key,
    _maybe_get_or_select,
    _reduce,
    _sum_td_features,
    default_value_kwargs,
    ValueEstimators,
)


class TOPRLoss(LossModule):
    """TOPR Loss based on the paper: https://arxiv.org/pdf/2503.14286v2
    
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            sample_log_prob (NestedKey or list of nested keys): The input tensordict key where the
               sample log probability is expected.
               Defaults to ``"sample_log_prob"`` when :func:`~tensordict.nn.composite_lp_aggregate` returns `True`,
                `"action_log_prob"`  otherwise.
            action (NestedKey or list of nested keys): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            reward (NestedKey or list of nested keys): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey or list of nested keys): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey or list of nested keys): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey | list[NestedKey] | None = None
        action: NestedKey | list[NestedKey] = "action"
        reward: NestedKey | list[NestedKey] = "reward"
        done: NestedKey | list[NestedKey] = "done"
        terminated: NestedKey | list[NestedKey] = "terminated"

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    default_value_estimator = ValueEstimators.GAE

    actor_network: ProbabilisticTensorDictModule
    actor_network_params: TensorDictParams


    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        *,
        entropy_bonus: bool = False,
        samples_mc_entropy: int = 1,
        entropy_coeff: float | Mapping[NestedKey, float] | None = None,
        log_explained_variance: bool = True,
        normalize_advantage: bool = False,
        normalize_advantage_exclude_dims: tuple[int] = (),
        advantage_key: str = None,
        ref_logprob_key: str = "ref_logprob",
        reward2go_key: str = "reward2go",
        functional: bool = True,
        actor: ProbabilisticTensorDictSequential = None,
        reduction: str = None,
        neg_clip_value: float = 0,
        pos_clip_value: float = 1,
        device: torch.device | None = None,
        **kwargs,
    ):
        if actor is not None:
            actor_network = actor
            del actor

        if actor_network is None:
            raise TypeError(
                "Missing positional arguments actor_network."
            )
        if reduction is None:
            reduction = "mean"

        self._functional = functional
        self._in_keys = None
        self._out_keys = None
        super().__init__()
        if functional:
            self.convert_to_functional(actor_network, "actor_network")
        else:
            self.actor_network = actor_network
            self.actor_network_params = None
            self.target_actor_network_params = None

        self.log_explained_variance = log_explained_variance
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.reduction = reduction
        self.ref_logprob_key = ref_logprob_key
        self.reward2go_key = reward2go_key

        if device is None:
            try:
                device = next(self.parameters()).device
            except (AttributeError, StopIteration):
                device = getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )()

        # Handle deprecated entropy_coef argument
        if "entropy_coef" in kwargs:
            if entropy_coeff is not None:  # Check if entropy_coeff was explicitly set
                raise ValueError(
                    "Cannot specify both 'entropy_coef' and 'entropy_coeff'"
                )
            warnings.warn(
                "'entropy_coef' is deprecated and will be removed in torchrl v0.11. Please use 'entropy_coeff' instead.",
                DeprecationWarning,
            )
            entropy_coeff = kwargs.pop("entropy_coef")

        # Set default value if None
        if entropy_coeff is None:
            entropy_coeff = 0.01

        if isinstance(entropy_coeff, Mapping):
            # Store the mapping for per-head coefficients
            self._entropy_coeff_map = {k: float(v) for k, v in entropy_coeff.items()}
            # Register an empty buffer for compatibility
            self.register_buffer("entropy_coeff", torch.tensor(0.0))
        elif isinstance(entropy_coeff, (float, int, torch.Tensor)):
            # Register the scalar entropy coefficient
            coeff = (
                float(entropy_coeff)
                if not torch.is_tensor(entropy_coeff)
                else float(entropy_coeff.item())
            )
            self.register_buffer("entropy_coeff", torch.tensor(coeff))
            self._entropy_coeff_map = None
        else:
            raise TypeError("entropy_coeff must be a float or a Mapping[str, float]")

        self.normalize_advantage = normalize_advantage
        self.normalize_advantage_exclude_dims = normalize_advantage_exclude_dims

        self.register_buffer("neg_clip_value", torch.tensor(neg_clip_value))
        self.register_buffer("pos_clip_value", torch.tensor(pos_clip_value))
        try:
            log_prob_keys = self.actor_network.log_prob_keys
            action_keys = self.actor_network.dist_sample_keys
            self.set_keys(sample_log_prob=log_prob_keys[0], action=action_keys[0])
        except AttributeError:
            pass

    @property
    def functional(self):
        return self._functional

    def _set_in_keys(self):
        keys = []
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys)
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.action)
        _maybe_add_or_extend_key(keys, self.tensor_keys.sample_log_prob)
        _maybe_add_or_extend_key(keys, self.tensor_keys.reward, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.done, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.terminated, "next")

        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values


    def reset(self) -> None:
        pass

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor | TensorDict:
        try:
            entropy = dist.entropy()
            if not entropy.isfinite().all():
                del entropy
                if VERBOSE:
                    torchrl_logger.info(
                        "Entropy is not finite. Using Monte Carlo sampling."
                    )
                raise NotImplementedError
        except NotImplementedError:
            if VERBOSE:
                torchrl_logger.warning(
                    f"Entropy not implemented for {type(dist)} or is not finite. Using Monte Carlo sampling."
                )
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))
            else:
                x = dist.sample((self.samples_mc_entropy,))
            with set_composite_lp_aggregate(False) if isinstance(
                dist, CompositeDistribution
            ) else contextlib.nullcontext():
                log_prob = dist.log_prob(x)
                if is_tensor_collection(log_prob):
                    if isinstance(self.tensor_keys.sample_log_prob, NestedKey):
                        log_prob = log_prob.get(self.tensor_keys.sample_log_prob)
                    else:
                        log_prob = log_prob.select(*self.tensor_keys.sample_log_prob)

            entropy = -log_prob.mean(0)
            if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
                entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)

    def _get_cur_log_prob(self, tensordict):
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            # assert tensordict['log_probs'].requires_grad
            # assert tensordict['logits'].requires_grad
            with self.actor_network_params.to_module(
                self.actor_network
            ) if self.functional else contextlib.nullcontext():
                dist = self.actor_network.get_dist(tensordict)
            is_composite = isinstance(dist, CompositeDistribution)

            if is_composite:
                action = tensordict.select(
                    *(
                        (self.tensor_keys.action,)
                        if isinstance(self.tensor_keys.action, NestedKey)
                        else self.tensor_keys.action
                    )
                )
            else:
                action = _maybe_get_or_select(tensordict, self.tensor_keys.action)

            if action.requires_grad:
                raise RuntimeError(
                    f"tensordict stored {self.tensor_keys.action} requires grad."
                )
            log_prob = dist.log_prob(action)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs (to compute "
                "the PPO objective) or the distribution (for the PPO entropy), please augment "
                f"the {type(self).__class__} by implementing your own logic in _get_cur_log_prob."
            )
        return log_prob, dist, is_composite

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, d.Distribution, torch.Tensor]:
        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob is None:
            raise KeyError(
                f"Couldn't find the log-prob {self.tensor_keys.sample_log_prob} in the input data."
            )
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        log_prob, dist, is_composite = self._get_cur_log_prob(tensordict)

        if is_composite:
            with set_composite_lp_aggregate(False):
                if not is_tensor_collection(prev_log_prob):
                    # this isn't great: in general, multi-head actions should have a composite log-prob too
                    warnings.warn(
                        "You are using a composite distribution, yet your log-probability is a tensor. "
                        "Make sure you have called tensordict.nn.set_composite_lp_aggregate(False).set() at "
                        "the beginning of your script to get a proper composite log-prob.",
                        category=UserWarning,
                    )

                    if is_tensor_collection(log_prob):
                        log_prob = _sum_td_features(log_prob)
                        log_prob.view_as(prev_log_prob)
                if log_prob.batch_size != adv_shape:
                    log_prob.batch_size = adv_shape
        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        if is_tensor_collection(log_weight):
            log_weight = _sum_td_features(log_weight)
            log_weight = log_weight.view(adv_shape).unsqueeze(-1)

        kl_approx = (prev_log_prob - log_prob).unsqueeze(-1)
        if is_tensor_collection(kl_approx):
            kl_approx = _sum_td_features(kl_approx)

        return log_weight, dist, kl_approx, log_prob



    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)

        log_weight, dist, kl_approx, log_prob = self._log_weight(
            tensordict
        )

        #prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        reward2go = tensordict.get(("next", "reward2go"))

        
        advantages = reward2go #- reward2go.mean() # subtract baseline
        # TOPR weights (canonical: a^- = 0, a^+ = b^+ = b^- = 1)
        log_weight = log_weight.exp().detach()
        log_weight = torch.where(advantages > 0, torch.ones_like(log_weight), torch.clamp(log_weight, min=0, max=1.0))
        negatives = (advantages < 0).sum()

        neg_loss = log_weight * advantages * log_prob.unsqueeze(-1)
        td_out = TensorDict({"loss_objective": -neg_loss})

        td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
        td_out.set("negatives", negatives)
        
        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantages.shape[:-1])
            if is_tensor_collection(entropy):
                # Reports the entropy of each action head.
                td_out.set("composite_entropy", entropy.detach())
                td_out.set(
                    "entropy", _sum_td_features(entropy).detach().mean()
                )  # for logging
            else:
                td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))
        
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
        )
        return td_out


    def _weighted_loss_entropy(
        self, entropy: torch.Tensor | TensorDictBase
    ) -> torch.Tensor:
        """Compute the weighted entropy loss.

        If `self._entropy_coeff_map` is provided, apply per-head entropy coefficients.
        Otherwise, use the scalar `self.entropy_coeff`.
        The entries in self._entropy_coeff_map require the full nested key to the entropy head.
        """
        if self._entropy_coeff_map is None:
            if is_tensor_collection(entropy):
                entropy = _sum_td_features(entropy)
            return -self.entropy_coeff * entropy

        loss_term = None  # running sum over heads
        coeff = 0
        for head_name, entropy_head in entropy.items(
            include_nested=True, leaves_only=True
        ):
            try:
                coeff = self._entropy_coeff_map[head_name]
            except KeyError as exc:
                raise KeyError(f"Missing entropy coeff for head '{head_name}'") from exc
            coeff_t = torch.as_tensor(
                coeff, dtype=entropy_head.dtype, device=entropy_head.device
            )
            head_loss_term = -coeff_t * entropy_head
            loss_term = (
                head_loss_term if loss_term is None else loss_term + head_loss_term
            )  # accumulate

        return loss_term

class TISLoss(LossModule):
    """TIS Loss based on the paper: https://arxiv.org/pdf/2503.14286v2
    Truncated Importance Sampling Loss
    
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            sample_log_prob (NestedKey or list of nested keys): The input tensordict key where the
               sample log probability is expected.
               Defaults to ``"sample_log_prob"`` when :func:`~tensordict.nn.composite_lp_aggregate` returns `True`,
                `"action_log_prob"`  otherwise.
            action (NestedKey or list of nested keys): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            reward (NestedKey or list of nested keys): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey or list of nested keys): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey or list of nested keys): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey | list[NestedKey] | None = None
        action: NestedKey | list[NestedKey] = "action"
        reward: NestedKey | list[NestedKey] = "reward"
        done: NestedKey | list[NestedKey] = "done"
        terminated: NestedKey | list[NestedKey] = "terminated"

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    default_value_estimator = ValueEstimators.GAE

    actor_network: ProbabilisticTensorDictModule
    actor_network_params: TensorDictParams


    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        *,
        entropy_bonus: bool = False,
        samples_mc_entropy: int = 1,
        entropy_coeff: float | Mapping[NestedKey, float] | None = None,
        log_explained_variance: bool = True,
        normalize_advantage: bool = False,
        normalize_advantage_exclude_dims: tuple[int] = (),
        advantage_key: str = None,
        ref_logprob_key: str = "ref_logprob",
        reward2go_key: str = "reward2go",
        functional: bool = True,
        actor: ProbabilisticTensorDictSequential = None,
        reduction: str = None,
        neg_clip_value: float = 0,
        pos_clip_value: float = 1,
        device: torch.device | None = None,
        **kwargs,
    ):
        if actor is not None:
            actor_network = actor
            del actor

        if actor_network is None:
            raise TypeError(
                "Missing positional arguments actor_network."
            )
        if reduction is None:
            reduction = "mean"

        self._functional = functional
        self._in_keys = None
        self._out_keys = None
        super().__init__()
        if functional:
            self.convert_to_functional(actor_network, "actor_network")
        else:
            self.actor_network = actor_network
            self.actor_network_params = None
            self.target_actor_network_params = None

        self.log_explained_variance = log_explained_variance
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.reduction = reduction
        self.ref_logprob_key = ref_logprob_key
        self.reward2go_key = reward2go_key

        if device is None:
            try:
                device = next(self.parameters()).device
            except (AttributeError, StopIteration):
                device = getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )()

        # Handle deprecated entropy_coef argument
        if "entropy_coef" in kwargs:
            if entropy_coeff is not None:  # Check if entropy_coeff was explicitly set
                raise ValueError(
                    "Cannot specify both 'entropy_coef' and 'entropy_coeff'"
                )
            warnings.warn(
                "'entropy_coef' is deprecated and will be removed in torchrl v0.11. Please use 'entropy_coeff' instead.",
                DeprecationWarning,
            )
            entropy_coeff = kwargs.pop("entropy_coef")

        # Set default value if None
        if entropy_coeff is None:
            entropy_coeff = 0.01

        if isinstance(entropy_coeff, Mapping):
            # Store the mapping for per-head coefficients
            self._entropy_coeff_map = {k: float(v) for k, v in entropy_coeff.items()}
            # Register an empty buffer for compatibility
            self.register_buffer("entropy_coeff", torch.tensor(0.0))
        elif isinstance(entropy_coeff, (float, int, torch.Tensor)):
            # Register the scalar entropy coefficient
            coeff = (
                float(entropy_coeff)
                if not torch.is_tensor(entropy_coeff)
                else float(entropy_coeff.item())
            )
            self.register_buffer("entropy_coeff", torch.tensor(coeff))
            self._entropy_coeff_map = None
        else:
            raise TypeError("entropy_coeff must be a float or a Mapping[str, float]")

        self.normalize_advantage = normalize_advantage
        self.normalize_advantage_exclude_dims = normalize_advantage_exclude_dims

        self.register_buffer("neg_clip_value", torch.tensor(neg_clip_value))
        self.register_buffer("pos_clip_value", torch.tensor(pos_clip_value))
        try:
            log_prob_keys = self.actor_network.log_prob_keys
            action_keys = self.actor_network.dist_sample_keys
            self.set_keys(sample_log_prob=log_prob_keys[0], action=action_keys[0])
        except AttributeError:
            pass

    @property
    def functional(self):
        return self._functional

    def _set_in_keys(self):
        keys = []
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys)
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.action)
        _maybe_add_or_extend_key(keys, self.tensor_keys.sample_log_prob)
        _maybe_add_or_extend_key(keys, self.tensor_keys.reward, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.done, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.terminated, "next")

        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values


    def reset(self) -> None:
        pass

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor | TensorDict:
        try:
            entropy = dist.entropy()
            if not entropy.isfinite().all():
                del entropy
                if VERBOSE:
                    torchrl_logger.info(
                        "Entropy is not finite. Using Monte Carlo sampling."
                    )
                raise NotImplementedError
        except NotImplementedError:
            if VERBOSE:
                torchrl_logger.warning(
                    f"Entropy not implemented for {type(dist)} or is not finite. Using Monte Carlo sampling."
                )
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))
            else:
                x = dist.sample((self.samples_mc_entropy,))
            with set_composite_lp_aggregate(False) if isinstance(
                dist, CompositeDistribution
            ) else contextlib.nullcontext():
                log_prob = dist.log_prob(x)
                if is_tensor_collection(log_prob):
                    if isinstance(self.tensor_keys.sample_log_prob, NestedKey):
                        log_prob = log_prob.get(self.tensor_keys.sample_log_prob)
                    else:
                        log_prob = log_prob.select(*self.tensor_keys.sample_log_prob)

            entropy = -log_prob.mean(0)
            if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
                entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)

    def _get_cur_log_prob(self, tensordict):
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            # assert tensordict['log_probs'].requires_grad
            # assert tensordict['logits'].requires_grad
            with self.actor_network_params.to_module(
                self.actor_network
            ) if self.functional else contextlib.nullcontext():
                dist = self.actor_network.get_dist(tensordict)
            is_composite = isinstance(dist, CompositeDistribution)

            if is_composite:
                action = tensordict.select(
                    *(
                        (self.tensor_keys.action,)
                        if isinstance(self.tensor_keys.action, NestedKey)
                        else self.tensor_keys.action
                    )
                )
            else:
                action = _maybe_get_or_select(tensordict, self.tensor_keys.action)

            if action.requires_grad:
                raise RuntimeError(
                    f"tensordict stored {self.tensor_keys.action} requires grad."
                )
            log_prob = dist.log_prob(action)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs (to compute "
                "the PPO objective) or the distribution (for the PPO entropy), please augment "
                f"the {type(self).__class__} by implementing your own logic in _get_cur_log_prob."
            )
        return log_prob, dist, is_composite

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, d.Distribution, torch.Tensor]:
        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob is None:
            raise KeyError(
                f"Couldn't find the log-prob {self.tensor_keys.sample_log_prob} in the input data."
            )
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        log_prob, dist, is_composite = self._get_cur_log_prob(tensordict)

        if is_composite:
            with set_composite_lp_aggregate(False):
                if not is_tensor_collection(prev_log_prob):
                    # this isn't great: in general, multi-head actions should have a composite log-prob too
                    warnings.warn(
                        "You are using a composite distribution, yet your log-probability is a tensor. "
                        "Make sure you have called tensordict.nn.set_composite_lp_aggregate(False).set() at "
                        "the beginning of your script to get a proper composite log-prob.",
                        category=UserWarning,
                    )

                    if is_tensor_collection(log_prob):
                        log_prob = _sum_td_features(log_prob)
                        log_prob.view_as(prev_log_prob)
                if log_prob.batch_size != adv_shape:
                    log_prob.batch_size = adv_shape
        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        if is_tensor_collection(log_weight):
            log_weight = _sum_td_features(log_weight)
            log_weight = log_weight.view(adv_shape).unsqueeze(-1)

        kl_approx = (prev_log_prob - log_prob).unsqueeze(-1)
        if is_tensor_collection(kl_approx):
            kl_approx = _sum_td_features(kl_approx)

        return log_weight, dist, kl_approx, log_prob



    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)


        log_weight, dist, kl_approx, log_prob = self._log_weight(
            tensordict
        )

        clipped_weight = torch.clamp(log_weight.exp().detach(),
        -self.neg_clip_value.to(log_weight.device),
        self.pos_clip_value.to(log_weight.device))

        reward2go = tensordict.get(("next", "reward2go"))
        #reward2go = reward2go - reward2go.mean() # subtract baseline
        neg_loss = clipped_weight * reward2go * log_prob
        td_out = TensorDict({"loss_objective": -neg_loss})

        td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
        
        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if is_tensor_collection(entropy):
                # Reports the entropy of each action head.
                td_out.set("composite_entropy", entropy.detach())
                td_out.set(
                    "entropy", _sum_td_features(entropy).detach().mean()
                )  # for logging
            else:
                td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))
        
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
        )
        return td_out


    def _weighted_loss_entropy(
        self, entropy: torch.Tensor | TensorDictBase
    ) -> torch.Tensor:
        """Compute the weighted entropy loss.

        If `self._entropy_coeff_map` is provided, apply per-head entropy coefficients.
        Otherwise, use the scalar `self.entropy_coeff`.
        The entries in self._entropy_coeff_map require the full nested key to the entropy head.
        """
        if self._entropy_coeff_map is None:
            if is_tensor_collection(entropy):
                entropy = _sum_td_features(entropy)
            return -self.entropy_coeff * entropy

        loss_term = None  # running sum over heads
        coeff = 0
        for head_name, entropy_head in entropy.items(
            include_nested=True, leaves_only=True
        ):
            try:
                coeff = self._entropy_coeff_map[head_name]
            except KeyError as exc:
                raise KeyError(f"Missing entropy coeff for head '{head_name}'") from exc
            coeff_t = torch.as_tensor(
                coeff, dtype=entropy_head.dtype, device=entropy_head.device
            )
            head_loss_term = -coeff_t * entropy_head
            loss_term = (
                head_loss_term if loss_term is None else loss_term + head_loss_term
            )  # accumulate

        return loss_term

