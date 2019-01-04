import torch


class OptimMixin(object):
    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    @staticmethod
    def group_param_tensor(group, name):
        param = group[name] = torch.as_tensor(group[name])
        return param

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons
        """
        for group in self.param_groups:
            self.stabilize_group(group)

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != "params"}
            packed["params"] = [id(p) for p in group["params"]]
            return packed

        param_groups = [pack_group(self._sanitize_group(g)) for g in self.param_groups]
        # Remap state to use ids as keys
        packed_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): self._sanitize_state(v)
            for k, v in self.state.items()
        }
        return {"state": packed_state, "param_groups": param_groups}

    def _sanitize_group(self, group):
        # some stuff may be converted to tensors
        return group

    @staticmethod
    def _sanitize_state(state):
        # do not pickle traced_step
        state.pop("traced_step", None)
        return state
