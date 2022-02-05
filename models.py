from abc import ABC, abstractmethod

from typing import List, Optional, Union, Tuple

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
    RobertaForMaskedLM,
    BertForMaskedLM,
    AlbertForMaskedLM,
    LogitsProcessorList,
    LogitsProcessor
)

class SelfDebiasingLogitsProcessor(LogitsProcessor):
    """This class represents a logits processor that applies self-debiasing."""

    def __init__(
        self,
        num_debiasing_prefixes: int,
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        :param num_debiasing_prefixes: the number of debiasing prefixes used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param tokenizer: a tokenizer used to print debugging output
        """
        assert (
            not debug or tokenizer
        ), "If debug=True, a tokenizer must be passed to SelfDebiasingLogitsProcessor()"
        self.num_debiasing_prefixes = num_debiasing_prefixes
        self.decay_constant = decay_constant
        self.epsilon = epsilon
        self.debug = debug
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = scores.shape[0] // (1 + self.num_debiasing_prefixes)
        regular_sentence_indices = range(batch_size)
        for regular_sentence_idx in regular_sentence_indices:
            bias_indices = self._get_bias_indices(regular_sentence_idx, batch_size)
            if bias_indices:
                self._debias_scores(scores, regular_sentence_idx, bias_indices)
        return scores

    def _get_bias_indices(
        self, regular_sentence_idx: int, batch_size: int
    ) -> List[int]:
        """Returns the indices of all self-debiasing inputs for a regular input"""
        return [
            regular_sentence_idx + (prefix_idx + 1) * batch_size
            for prefix_idx in range(self.num_debiasing_prefixes)
        ]

    def _debias_scores(
        self, scores: torch.FloatTensor, regular_sent_idx: int, bias_indices: List[int]
    ) -> None:
        """Partially debiases the given scores considering a single sentence and the corresponding self-debiasing inputs"""
        logits_biased = [scores[bias_idx] for bias_idx in bias_indices]

        mask = self._generate_decay_mask(scores[regular_sent_idx], logits_biased)
        scores[regular_sent_idx] = torch.log(
            self._apply_decay_mask(scores[regular_sent_idx], mask)
        )

        for debiasing_sent_idx in bias_indices:
            scores[debiasing_sent_idx] = scores[regular_sent_idx]

    def _apply_decay_mask(
        self, logits: torch.Tensor, decay_mask: torch.Tensor
    ) -> torch.Tensor:
        """Applies exponential decay to a tensor of logits"""
        probabilities = logits.softmax(dim=-1)
        decay_mask = torch.exp(-decay_mask * self.decay_constant)
        decay_mask = torch.max(
            decay_mask, torch.tensor([self.epsilon], device=decay_mask.device)
        )
        probabilities = probabilities * decay_mask
        probabilities = probabilities / probabilities.sum(dim=-1)
        return probabilities

    def _generate_decay_mask(
        self,
        logits_regular: torch.FloatTensor,
        logits_biased_list: List[torch.FloatTensor],
    ) -> torch.Tensor:
        """Computes the alpha values (see paper) for each token and stores them in a mask tensor"""
        p_regular = logits_regular.softmax(dim=-1)
        p_biased = None

        for logits_biased in logits_biased_list:
            if p_biased is None:
                p_biased = logits_biased.softmax(dim=-1)
            else:
                p_biased = torch.max(p_biased, logits_biased.softmax(dim=-1))

        if self.debug:
            print(
                f"== Before Debiasing ==\n"
                f"Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}\n"
                f"Top 5 predictions (biased): {self._get_most_likely_tokens(p_biased, k=5)}"
            )

        mask = torch.max(
            p_biased - p_regular, torch.tensor([0.0], device=p_regular.device)
        )

        if self.debug:
            p_regular = self._apply_decay_mask(logits_regular, mask)
            print(
                f"== After Debiasing ==\n"
                f"Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}"
            )

        return mask

    def _get_most_likely_tokens(
        self, probabilities_tensor: torch.Tensor, k: int
    ) -> List[Tuple[str, float]]:
        """Returns the most likely tokens according to a tensor of probabilities"""
        assert len(probabilities_tensor.shape) == 1
        values, indices = torch.topk(probabilities_tensor, k=k, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(indices)
        return list(zip(tokens, [pv.item() for pv in values]))

def get_top_k_tokens(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, k: int = 5):
    values, indices = torch.topk(logits, k, dim=-1)
    if len(logits.shape) == 2:
        assert logits.shape[0] == 1
        values, indices = values[0], indices[0]
    return tokenizer.convert_ids_to_tokens(indices), values


class MaskedLMWrapper(ABC):
    """
    This class represents a wrapper for a masked language model that provides the ability to perform self-debiasing for sentences with
    a single masked token.
    """

    def __init__(self, model_name: str, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name).to(self._device)

    def get_token_logits(self, input_text: str) -> torch.Tensor:
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        assert (
            sum(1 for id_ in input_ids[0] if id_ == self._tokenizer.mask_token_id) == 1
        ), "Input text must contain exactly one mask token"
        scores = self._model(input_ids)["logits"]
        mask_positions = input_ids == self._tokenizer.mask_token_id
        return scores[mask_positions]

    def get_token_logits_batch(self, input_texts: List[str]) -> torch.Tensor:
        batch = self._tokenizer.batch_encode_plus(
            input_texts, return_tensors="pt", padding=True
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}

        mask_positions = batch["input_ids"] == self._tokenizer.mask_token_id
        assert torch.all(
            mask_positions.sum(axis=-1) == 1
        ), "Each input text must contain exactly one mask token"

        scores = self._model(**batch)["logits"]
        return scores[mask_positions]

    def get_token_logits_self_debiasing(
        self,
        input_ids: torch.Tensor,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
    ) -> torch.Tensor:
        """
        Computes the token logits for the single masked position in the given input ids with self-debiasing.
        :param input_ids: the input ids
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :return: the cross entropy loss
        """
        assert (
            input_ids.shape[0] == 1
        )  # TODO future versions should also work with batches

        logits_processor = SelfDebiasingLogitsProcessor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            tokenizer=self._tokenizer,
        )

        input_prefixes = [self._tokenizer.cls_token] + [
            " ".join([self._tokenizer.cls_token, dp]) for dp in debiasing_prefixes
        ]

        input_prefixes = self._tokenizer.batch_encode_plus(
            input_prefixes, padding=True, return_tensors="pt", add_special_tokens=False
        )
        input_prefixes["attention_mask"] = torch.flip(
            input_prefixes["attention_mask"], dims=[1]
        )

        # remove leading [CLS] tokens
        input_ids = input_ids[:, 1:]

        shifts = input_prefixes["attention_mask"].shape[-1] - input_prefixes[
            "attention_mask"
        ].sum(dim=-1)
        for batch_idx in range(input_prefixes["input_ids"].shape[0]):
            input_prefixes["input_ids"][batch_idx] = input_prefixes["input_ids"][
                batch_idx
            ].roll(shifts[batch_idx].item())

        input_prefixes = {k: v.to(self._device) for k, v in input_prefixes.items()}

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes) + 1, 1)
        attention_mask = torch.ones_like(input_ids_repeated)

        attention_mask = torch.cat(
            [input_prefixes["attention_mask"], attention_mask], dim=-1
        )
        input_ids_repeated = torch.cat(
            [input_prefixes["input_ids"], input_ids_repeated], dim=-1
        )

        mask_positions = input_ids_repeated == self._tokenizer.mask_token_id

        position_ids = attention_mask.long().cumsum(-1)
        if isinstance(self._model, RobertaForMaskedLM):
            position_ids += self._model.base_model.embeddings.padding_idx
        elif isinstance(self._model, BertForMaskedLM):
            position_ids -= 1
        elif isinstance(self._model, AlbertForMaskedLM):
            position_ids -= 1
        else:
            raise ValueError(
                f"Position IDs shift is not implemented for {self._model.__class__}"
            )

        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self._model(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        lm_logits = outputs["logits"]

        for idx in range(lm_logits.shape[1]):
            if torch.any(mask_positions[:, idx]):
                lm_logits[:, idx, :] = logits_processor(
                    input_ids=None, scores=lm_logits[:, idx, :]
                )

        return lm_logits[mask_positions][0]

    def compute_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        outputs = self._model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def compute_loss_self_debiasing(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
    ) -> torch.Tensor:

        relevant_labels = labels[input_ids == self._tokenizer.mask_token_id]
        token_logits = self.get_token_logits_self_debiasing(
            input_ids,
            debiasing_prefixes=debiasing_prefixes,
            decay_constant=decay_constant,
            epsilon=epsilon,
        )
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            token_logits.view(-1, self._model.config.vocab_size),
            relevant_labels.view(-1),
        )
        return masked_lm_loss
