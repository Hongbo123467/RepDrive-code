"""
Seg head for bev understanding
"""

import torch
import torch.nn as nn
from einops import rearrange


class BevSegHead(nn.Module):
    def __init__(self, target, input_dim, output_class):
        super(BevSegHead, self).__init__()
        self.target = target
        dynamic_classes, static_classes = self._resolve_output_classes(
            target, output_class
        )

        if self.target == 'dynamic':
            self.dynamic_head = nn.Conv2d(input_dim,
                                          dynamic_classes,
                                          kernel_size=3,
                                          padding=1)
        elif self.target == 'static':
            # segmentation head
            self.static_head = nn.Conv2d(input_dim,
                                         static_classes,
                                         kernel_size=3,
                                         padding=1)
        else:
            self.dynamic_head = nn.Conv2d(input_dim,
                                          dynamic_classes,
                                          kernel_size=3,
                                          padding=1)
            self.static_head = nn.Conv2d(input_dim,
                                         static_classes,
                                         kernel_size=3,
                                         padding=1)

    @staticmethod
    def _resolve_output_classes(target, output_class):
        if isinstance(output_class, dict):
            dynamic_classes = output_class.get('dynamic', output_class.get('dynamic_seg'))
            static_classes = output_class.get('static', output_class.get('static_seg'))
            if dynamic_classes is None or static_classes is None:
                raise ValueError(
                    "output_class dict must define 'dynamic' and 'static' classes"
                )
            return dynamic_classes, static_classes

        if isinstance(output_class, (list, tuple)):
            if len(output_class) != 2:
                raise ValueError(
                    "output_class list/tuple should be [dynamic_classes, static_classes]"
                )
            return output_class[0], output_class[1]

        if target == 'both':
            return output_class, 3
        return output_class, output_class

    def forward(self,  x, b, l):
        if self.target == 'dynamic':
            dynamic_map = self.dynamic_head(x)
            dynamic_map = rearrange(dynamic_map, '(b l) c h w -> b l c h w',
                                    b=b, l=l)
            static_map = torch.zeros_like(dynamic_map,
                                          device=dynamic_map.device)

        elif self.target == 'static':
            static_map = self.static_head(x)
            static_map = rearrange(static_map, '(b l) c h w -> b l c h w',
                                   b=b, l=l)
            dynamic_map = torch.zeros_like(static_map,
                                           device=static_map.device)

        else:
            dynamic_map = self.dynamic_head(x)
            dynamic_map = rearrange(dynamic_map, '(b l) c h w -> b l c h w',
                                    b=b, l=l)
            static_map = self.static_head(x)
            static_map = rearrange(static_map, '(b l) c h w -> b l c h w',
                                   b=b, l=l)

        output_dict = {'static_seg': static_map,
                       'dynamic_seg': dynamic_map}

        return output_dict

