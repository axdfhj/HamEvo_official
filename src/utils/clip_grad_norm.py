from torch.nn.utils import clip_grad_norm_

from .pylogger import RankedLogger


def get_CGN(args):
    if args is None:
        return None
    return ClipGradNorm(**args)


log = RankedLogger(__name__, rank_zero_only=True)


class ClipGradNorm:
    def __init__(
        self,
        start_iteration=0,
        end_iteration=-1,  # if negative, the norm will always be clipped
        max_norm=0.5,
    ):
        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.max_norm = self.original_max = max_norm

        self.last_epoch = -1
        self.last_norm = None
        self.last20_norm_list = []
        log.info(
            f"clip grad norms works. start_iteration: {start_iteration}, end_iteration: {end_iteration}, max_norm: {max_norm}."
        )

    def adaptive_max_norm(self):
        """Update self.max_norm every 1000 steps."""
        if self.last_epoch % 100 == 0 and self.last20_norm_list:
            if len(self.last20_norm_list) >= 100:
                # Calculate the median and set max_norm to 3 times the median
                sorted_norms = sorted(self.last20_norm_list)
                median_norm = sorted_norms[len(sorted_norms) // 2]
                self.max_norm = min(3 * median_norm, self.original_max)
            log.info(f"max_norm update to {self.max_norm}.")

    def __call__(self, parameters):
        self.last_epoch += 1
        self.adaptive_max_norm()  # Update max_norm if necessary
        clip = False
        if self.last_epoch >= self.start_iteration:
            clip = True
        if self.end_iteration > 0 and self.last_epoch < self.end_iteration:
            clip = True
        if clip:
            total_norm = clip_grad_norm_(parameters, max_norm=min(self.original_max, self.max_norm))
            self.last_norm = total_norm
            # Update the norm history
            self.last20_norm_list.append((total_norm).cpu().item())
            if len(self.last20_norm_list) > 100:
                self.last20_norm_list.pop(0)  # Keep only the last 100 values
            return total_norm.cpu().item()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
