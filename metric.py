

from dataclasses import dataclass
from typing import Optional, List, Any, Callable, Dict, Tuple
import torch


# inspired from https://github.com/google/CommonLoopUtils/tree/master/clu/metric_writers
#
# + declarative programming using `dataclass`
# + nice trick to return inner Subclass for fluent interface
#
# ```python
# @dataclass
# class Metrics(Collection):
#     top5acc: Accuracy.from_output("top5acc")
# ```

class Metric:
    "Interface for computing metrics"
    
    @classmethod
    def from_model_output(cls, *args, **kwargs) -> "Metric":
        raise NotImplementedError("Must override from_model_output()")

    def merge(self, other: "Metric") -> "Metric":
        """Returns `Metric` that is the accumulation of `self` and `other`.
        Args:
          other: A `Metric` whose inermediate values should be accumulated onto the
            values of `self`. 
        Returns:
          A new `Metric` that accumulates the value from both `self` and `other`.
        """
        raise NotImplementedError("Must override merge()")

    def compute(self):
        "Computes final metrics from intermediate values."
        raise NotImplementedError("Must override compute()")
        
    @classmethod
    def from_fun(cls, fun: Callable):  # pylint: disable=g-bare-generic
        """Calls `cls.from_model_output` with the return value from `fun`."""

        class Fun(cls):
            @classmethod
            def from_model_output(cls, **model_output) -> Metric:
                return super().from_model_output(fun(**model_output))

        return Fun

    @classmethod
    def from_output(cls, name: str):  # pylint: disable=g-bare-generic
        """Calls `cls.from_model_output` with model output named `name`."""

        class FromOutput(cls):
            @classmethod
            def from_model_output(cls, **model_output) -> Metric:
                return super().from_model_output(model_output[name])

        return FromOutput


Metric.__doc__ += """
Refer to `Collection` for computing multipel metrics at the same time.

Synopsis:

@dataclass
class Average(Metric):
    total: torch.Tensor
    count: torch.Tensor
    @classmethod
    def from_model_output(cls, value: jnp.array, **_) -> Metric:
        return cls(total=value.sum(), count=jnp.prod(value.shape))
    def merge(self, other: Metric) -> Metric:
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count,
        )
    def compute(self):
        return self.total / self.count

average = None
for value in range(data):
    update = Average.from_model_output(value)
    average = update if average is None else average.merge(update)
print(average.compute())
"""


# ### Average

@dataclass
class Average(Metric):
    """Compute the average of `values`.
    
    Optionally taking a mask to ignore values with mask = 0
    - values : ndim = 0 or ndim = 1
    - masks : shape same as values
    """
    total: torch.Tensor  # accumulation
    count: torch.Tensor  # number of merges

    @classmethod
    def from_model_output(cls, values: torch.Tensor, 
                          mask: Optional[torch.Tensor]=None, **_) -> Metric:
        if values.ndim == 0: 
            values = values[None] # prepend 1
        if mask is None: 
            mask = torch.ones(values.shape).to(values.device)
        return cls(
            total=(mask* values).sum(), 
            count=mask.sum()
        )

    def merge(self, other: "Average") -> "Average":
        # assert total of the same shape
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count
        )
    
    def compute(self) -> Any:
        return self.total / self.count


# ### Accuracy

@dataclass
class Accuracy(Average):
    """Computes the average accuracy from model outputs `logits` and `labels`.
    
    - `labels` {int32} : shape (num_classes)
    - `logits` : shape (batch_size, num_classes)
    """  
    
    @classmethod
    def from_model_output(cls, *, 
                          logits: torch.Tensor, 
                          labels: torch.Tensor, **kwargs) -> Metric:
        return super().from_model_output(
            values=(logits.argmax(axis=-1) == labels).float(), **kwargs
        )


# ### Loss

@dataclass
class Loss(Average):
    "Computes the average `loss`"
    
    @classmethod
    def from_model_output(cls, loss: torch.Tensor, **kwargs) -> Metric:
        return super().from_model_output(values=loss, **kwargs)


# ### Std

@dataclass
class Std(Metric):
    "Computes the standard deviation of a scalar or a batch of scalars."
    total: torch.Tensor
    sum_of_squares: torch.Tensor
    count: torch.Tensor

    @classmethod
    def from_model_output(cls, values: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None,
                          **_) -> Metric:
        if values.ndim == 0:
            values = values[None]
        # utils.check_param(values, ndim=1)
        if mask is None:
            mask = torch.ones(values.shape[0])
        return cls(
            total=values.sum(),
            sum_of_squares=(mask * values**2).sum(),
            count=mask.sum(),
        )

    def merge(self, other: "Std") -> "Std":
        # _assert_same_shape(self.total, other.total)
        return type(self)(
            total=self.total + other.total,
            sum_of_squares=self.sum_of_squares + other.sum_of_squares,
            count=self.count + other.count,
        )

    def compute(self) -> Any:
        # var(X) = 1/N \sum_i (x_i - mean)^2
        #        = 1/N \sum_i (x_i^2 - 2 x_i mean + mean^2)
        #        = 1/N ( \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - 2 mean N mean + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - N * mean^2 )
        #        = \sum_i x_i^2 / N - mean^2
        mean = self.total / self.count
        return (self.sum_of_squares / self.count - mean**2)**.5


# ### Collection

@dataclass
class _ReductionCounter(Metric):
    """Pseudo metric that keeps track of the total number of `.merge()`."""
    value: torch.Tensor

    def merge(self, other: "_ReductionCounter") -> "_ReductionCounter":
        return _ReductionCounter(self.value + other.value)


@dataclass
class Collection:
    "Updates a collection of `Metric` from model outputs."
    _reduction_counter: _ReductionCounter
    
    @classmethod
    def _from_model_output(cls, **kwargs) -> "Collection":
        return cls(
            _reduction_counter=_ReductionCounter(torch.tensor(1)),
            **{
                metric_name: metric.from_model_output(**kwargs)
                for metric_name, metric in cls.__annotations__.items()
            }
        )
    
    @classmethod
    def single_from_model_output(cls, **kwargs) -> "Collection":
        return cls._from_model_output(**kwargs)
    
    def merge(self, other: "Collection") -> "Collection":
        """Returns `Collection` that is the accumulation of `self` and `other`."""
        return type(self)(**{
            metric_name: metric.merge(getattr(other, metric_name))
            for metric_name, metric in vars(self).items()
        })
    
    def reduce(self) -> "Collection":
        """Reduces the collection by calling `Metric.reduce()` on each metric."""
        return type(self)(**{
            metric_name: metric.reduce()
            for metric_name, metric in vars(self).items()
        })
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Computes metrics and returns them as Python numbers/lists."""
        ndim = self._reduction_counter.value.ndim
        if ndim != 0:
          raise ValueError(
              f"Collection is still replicated (ndim={ndim}). Maybe you forgot to "
              f"call a flax.jax_utils.unreplicate() or a Collections.reduce()?")
        return {
            metric_name: metric.compute()
            for metric_name, metric in vars(self).items()
            if metric_name != "_reduction_counter"
        }


Collection.__doc__ +="""
Synopsis:
@dataclass
class Metrics(Collection):
    accuracy: Accuracy

metrics = None
for inputs, labels in data:
    logits = model(inputs)
    update = Metrics.single_from_model_output(logits=logits, labels=labels)
    metrics = update if metrics is None else metrics.merge()
print(metrics.compute())
"""


