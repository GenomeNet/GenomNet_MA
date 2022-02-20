class BaseSampler:
    def sample(self):
        raise NotImplementedError("Sampler has not been implemented")
    def update_sampler(self, *args, **kwargs):
        pass
