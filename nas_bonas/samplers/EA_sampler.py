import random
from opendomain_utils.mutate import all_mutates
from .base_sampler import BaseSampler
# from samplers.base_sampler import BaseSampler
from opendomain_utils.transform_genotype import geno_to_archs

import copy
import random
# population_size=5, tornament_size=3

class EASampler(BaseSampler):
    def __init__(self, trained_arch_list, population_size=30, tornament_size=10):
        # trained arch list[genostr, acc]
        genotypes = [sample['genotype'] for sample in trained_arch_list]
        ei_scores = [sample['metrics'] for sample in trained_arch_list]
        dataset = geno_to_archs(genotypes, ei_scores)
        self.population_size=population_size
        
        dataset_x = copy.deepcopy(dataset)
        for i, candidate in enumerate(dataset_x):
            candidate.pop("metrics")
        self.dataset_x = dataset_x
        # tornament_size, population_size = 2, 2
        self.tornament_size = tornament_size
        self.population = random.sample(dataset, population_size)
        self.candidates = random.sample(self.population, self.tornament_size)
        # candidates = random.sample(population, tornament_size)
        
    def sample(self, num_samples):
        #parent = max(self.candidates, key=lambda i: i['metrics']) # parent ist das model mit höchstem acc, weil max
        
        # parent = max(candidates, key=lambda i: i['metrics'])
        # for i, candidate in enumerate(self.population):
        #    if str(candidate) == str(parent):
        #        del self.population[i]
          
        def acc_position(dict):
            return dict['metrics']
        self.candidates.sort(reverse=True, key=acc_position) 
        # print('candidates')
        # print(self.candidates)
        
        samples = all_mutates(self.candidates, num_samples, previous_data = self.dataset_x)
        
        return samples
    

    # hier wird jetzt nur self.candidates upgedatet, damit wir später daraus immer parent in sample() bilden können
    def update_sampler(self, trained_arch_list, ifappend=True, *args, **kwargs):
        # TODO: support update sampler for muliple children in supernet case
        # child = trained_datapoints
        
        genotypes = [sample['genotype'] for sample in trained_arch_list]
        ei_scores = [sample['metrics'] for sample in trained_arch_list]
        dataset = geno_to_archs(genotypes, ei_scores)
        print('new_archlist_forsampler')
        print(len(trained_arch_list))
        dataset_x = copy.deepcopy(dataset)
        for i, candidate in enumerate(dataset_x):
            candidate.pop("metrics")
        self.dataset_x = dataset_x
        
        self.population = random.sample(dataset, self.population_size)
        
        print(f"updated sampler, pupulation size:{len(self.population)}")
        self.candidates = random.sample(self.population, self.tornament_size)
        
        
        