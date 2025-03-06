# nn.Module

[pytorchの実装](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/module.py#L402)

参考リンク  
- [pytorchでtensorやnn.Moduleにhookを使ってアクセスする方法](https://gist.github.com/eminamitani/40df0b87f20aaa588cbcee6405f573ad)
- [【PyTorch】model解説](https://zenn.dev/yuto_mo/articles/90976009a5d52e)

`nn.Module`はニューラルネットワークモジュールの基底クラスであり、`__init__`と`forward`関数を実装する必要がある。
```python
# L402
class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
```

`model(x)`の形で推論をすることが出来るが、`__call__`関数は以下のように実装されている。
```python
# L1878
 __call__: Callable[..., Any] = _wrapped_call_impl
```

`_wrapped_call_impl`は以下のように条件分岐する。`_compiled_call_impl`で検索してもほとんど引っかからないため、ほぼ`_call_impl`関数が呼ばれると思ってよさそうだ。
```python
# L1735
def _wrapped_call_impl(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
        else:
            return self._call_impl(*args, **kwargs)
```

`_call_impl`関数内では、基本的に`forward`関数を計算した結果が返される。
```python
 def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
    ~~~
    result = forward_call(*args, **kwargs)

```

`_call_impl`関数内ではモジュールにセットしたhook関数が呼ばれる。hook関数は、例えば中間層の出力にアクセスするために使われることが多い。(参考リンク1) ただ、オプションなので必須ではない。

