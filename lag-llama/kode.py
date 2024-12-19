#Implementasi Pengambilan Data
import yfinance as yf

symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'INTC', 'VZ', 'HD', 'T', 'DIS', 'BAC', 'XOM', 'PFE', 'KO', 'CSCO', 'CMCSA', 'PEP', 'BA', 'WFC', 'MCD', 'COST', 'CVX', 'ABT', 'LLY', 'NKE', 'MDT', 'IBM', 'NEE', 'DHR', 'HON', 'ACN', 'TXN', 'GILD', 'LMT', 'BMY', 'SBUX', 'FIS', 'AMGN', 'MO', 'INTU', 'UNP', 'MMM', 'CHTR', 'TMO', 'LOW', 'BKNG', 'UPS', 'ADBE', 'QCOM', 'ORCL', 'MS', 'BLK', 'GS', 'NOW', 'SCHW', 'MDLZ', 'GE', 'C', 'FDX', 'CAT', 'AXP', 'SPGI', 'TGT', 'ISRG', 'DE', 'SYK', 'NSC', 'MU', 'SO', 'GM', 'ADP', 'VRTX', 'ZTS', 'USB', 'CI', 'GD', 'REGN', 'RTX', 'DUK', 'PGR', 'CL', 'CVS', 'D', 'CSX', 'TFC', 'COP', 'SPG', 'EL', 'MMC', 'SHW', 'BIIB']

stock_prices = (
    yf.Tickers(symbols)
    .history(period="max", start="2013-01-01")
    .Close
    .resample('1d')
    .ffill()
)

stock_prices

#Implementasi Data Preprocessing
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset

stock_returns = stock_prices.rolling(5).mean().pct_change().dropna()

def _get_lag_llama_dataset(dataset):
    dataset = dataset.copy()

    for col in dataset.columns:
        if dataset[col].dtype != "object" and not pd.api.types.is_string_dtype(dataset[col]):
            dataset[col] = dataset[col].astype("float32")

    backtest_dataset = PandasDataset(dict(dataset))
    return backtest_dataset

train_dataset = _get_lag_llama_dataset(stock_returns.iloc[:int(0.7*len(stock_returns))])
test_dataset = _get_lag_llama_dataset(stock_returns.iloc[int(0.7*len(stock_returns)):, :10])

prediction_length = 60
num_samples = 1060
device = torch.device("cuda:0")

#Implementasi Persiapan Model
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.torch.distributions import DistributionOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand

from gluon_utils.scalers.robust_scaler import RobustScaler


@dataclass
class LTSMConfig:
    feature_size: int = 3 + 6
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    n_embd_per_head: int = 128
    rope_scaling: Optional[dict] = None
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), use_kv_cache)
        y = x + self.mlp(self.rms_2(x))
        return y


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, device, dtype, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            config.n_embd_per_head * config.n_head,
            bias=False,
        )
        self.kv_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            2 * config.n_embd_per_head * config.n_head,
            bias=False,
        )
        self.c_proj = nn.Linear(
            config.n_embd_per_head * config.n_head,
            config.n_embd_per_head * config.n_head,
            bias=False,
        )
        self.n_head = config.n_head
        self.n_embd_per_head = config.n_embd_per_head
        self.block_size = config.block_size
        self.dropout = config.dropout
        self.rope_scaling = config.rope_scaling
        self._init_rope()
        self.kv_cache = None

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.n_embd_per_head, max_position_embeddings=self.block_size
            )

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        (B, T, C) = x.size()

        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(self.n_embd_per_head * self.n_head, dim=2)

        cache_initialized = self.kv_cache is not None
        if use_kv_cache:
            if cache_initialized:
                k = torch.cat([self.kv_cache[0], k], dim=1)[:, 1:]
                v = torch.cat([self.kv_cache[1], v], dim=1)[:, 1:]
                self.kv_cache = k, v
            else:
                self.kv_cache = k, v

        k = k.view(B, -1, self.n_head, self.n_embd_per_head).transpose(1, 2)
        q = q.view(B, -1, self.n_head, self.n_embd_per_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, self.n_embd_per_head).transpose(1, 2)

        true_seq_len = k.size(2)
        if self.rotary_emb is not None:
            if use_kv_cache and cache_initialized:
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=true_seq_len)
                q, _ = apply_rotary_pos_emb(q, k, cos, sin, position_ids=[-1])
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=true_seq_len)
                _, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)
            else:
                cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=T)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)

        if use_kv_cache and cache_initialized:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd_per_head * config.n_head
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_fc2 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_proj = nn.Linear(
            n_hidden, config.n_embd_per_head * config.n_head, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class LagLlamaModel(nn.Module):
    def __init__(
        self,
        context_length: int,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd_per_head: int,
        n_head: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        rope_scaling=None,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.lags_seq = lags_seq
        if time_feat:
            feature_size = input_size * (len(self.lags_seq)) + 2 * input_size + 6
        else:
            feature_size = input_size * (len(self.lags_seq)) + 2 * input_size

        config = LTSMConfig(
            n_layer=n_layer,
            n_embd_per_head=n_embd_per_head,
            n_head=n_head,
            block_size=max_context_length,
            feature_size=feature_size,
            rope_scaling=rope_scaling,
            dropout=dropout,
        )
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        elif scaling == "robust":
            self.scaler = RobustScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(
            config.n_embd_per_head * config.n_head
        )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(
                    config.feature_size, config.n_embd_per_head * config.n_head
                ),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd_per_head * config.n_head),
            )
        )
        self.y_cache = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        if future_target is not None:
            input = torch.cat(
                (
                    scaled_past_target[..., max(self.lags_seq) :],
                    (future_target[..., :-1] - loc) / scale,
                ),
                dim=-1,
            )
        else:
            input = scaled_past_target[..., max(self.lags_seq) :]
        if (past_time_feat is not None) and (future_time_feat is not None):
            time_feat = (
                torch.cat(
                    (
                        past_time_feat[..., max(self.lags_seq) :, :],
                        future_time_feat[..., :-1, :],
                    ),
                    dim=1,
                )
                if future_time_feat is not None
                else past_time_feat[..., max(self.lags_seq) :, :]
            )

        prior_input = (
            past_target[..., : max(self.lags_seq)] - loc
        ) / scale

        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )

        static_feat = torch.cat(
            (loc.abs().log1p(), scale.log()), dim=-1
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=lags.shape[-2]
        )

        if past_time_feat is not None:
            return (
                torch.cat((lags, expanded_static_feat, time_feat), dim=-1),
                loc,
                scale,
            )
        else:
            return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )
        if use_kv_cache and self.y_cache:
            transformer_input = transformer_input[:, -1:]

        x = self.transformer.wte(
            transformer_input
        )

        for block in self.transformer.h:
            x = block(x, use_kv_cache)
        x = self.transformer.ln_f(
            x
        )
        if use_kv_cache:
            self.y_cache = True
        params = self.param_proj(
            x
        )
        return params, loc, scale

    def reset_cache(self) -> None:
        self.y_cache = False
        for block in self.transformer.h:
            block.attn.kv_cache = None


#Training
ckpt = torch.load("./lag-llama-model/lag-llama.ckpt", map_location=device)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

estimator = LagLlamaEstimator(
        ckpt_path="./lag-llama-model/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=256,
        nonnegative_pred_samples=True,
        aug_prob=0,
        lr=5e-4,

        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        time_feat=estimator_args["time_feat"],

        use_cosine_annealing_lr=True,
        cosine_annealing_lr_args={
            'T_max':50,
            'eta_min':1e-6
        },
        batch_size=64,
        num_parallel_samples=num_samples,
        trainer_kwargs = {"max_epochs": 50}
    )

predictor = estimator.train(
    train_dataset,
    cache_data=True,
    shuffle_buffer_length=1000
)

#Zero-shot learning
ckpt = torch.load("./lag-llama-model/lag-llama.ckpt", map_location="cuda:0") # Uses GPU since in this Colab we use a GPU.
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

def get_lag_llama_predictions(dataset, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100):

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="./lag-llama-model/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,

        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


forecasts_ctx_len_64, tss_ctx_len_64 = get_lag_llama_predictions(test_dataset, prediction_length=prediction_length, device=device, \
                                           context_length=64, use_rope_scaling=False, num_samples=num_samples)
forecasts_ctx_len_64 = list(forecasts_ctx_len_64)
tss_ctx_len_64 = list(tss_ctx_len_64)

agg_metrics_ctx_len_64, ts_metrics_ctx_len_64 = evaluator(iter(tss_ctx_len_64), iter(forecasts_ctx_len_64))
print("CRPS:", agg_metrics_ctx_len_64['mean_wQuantileLoss'])
print('MSE:', agg_metrics_ctx_len_64['MSE'])
print('MAPE:', agg_metrics_ctx_len_64['MAPE'])

#Evaluasi
