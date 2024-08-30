import pandas as pd
from enum import Enum
import random
import os

UCB_const : float = 1.0
Score_upper_limit = 200
Score_lower_limit = 0
Score_insert_standard = 100
Score_delete_standard = 1

class Seed:
    def __init__(self, 
                 context : str,
                 ):
        self.context = context
        self.__visited_num = 0
        self.score = 60
        self.success_attack_num = 0

    def get_visited_num(self) -> int:
        return self.__visited_num

    def visit(self) -> str:
        self.__visited_num += 1
        return self.context
    
    @classmethod
    def takeScore(cls, seed) -> int:
        return seed.score
    

class SeedSelectionPolicy(Enum):
    RandomSelection = 0
    RoundRobinSelection = 1
    UCBSelection = 2
    MCTSExploreSelection = 3    

    @classmethod
    def selection_policy_map(cls, 
                             aim : str):
        aim = aim.lower()
        if aim.find('ucb') != -1:
            return SeedSelectionPolicy.UCBSelection
        elif aim.find('round') != -1 or aim.find('robin') != -1:
            return SeedSelectionPolicy.RoundRobinSelection
        elif aim.find('mcts') != -1:
            return SeedSelectionPolicy.MCTSExploreSelection
        else:
            return SeedSelectionPolicy.RandomSelection

class SeedSelection:
    def __init__(self, 
                 policy : SeedSelectionPolicy):
        self.policy = policy

    # return the index of chosen seed in the list
    def select(self, 
               seed_list : 'list[Seed]') -> int:
        if self.policy == SeedSelectionPolicy.RandomSelection:
            return self.__select_random(seed_list)
        elif self.policy == SeedSelectionPolicy.RoundRobinSelection:
            return self.__select_robin(seed_list)
        elif self.policy == SeedSelectionPolicy.UCBSelection:
            return self.__select_ucb(seed_list)
        elif self.policy == SeedSelectionPolicy.MCTSExploreSelection:
            return self.__select_mcts(seed_list)
        else:
            return None
    
    def __select_random(self, 
                      score_list : 'list[Seed]') -> int:
        return random.randrange(start = 0, stop = len(score_list), step = 1)

    def __select_robin(self, 
                     score_list : 'list[Seed]') -> int:
        i = 0
        min_index = 0
        while i < len(score_list):
            if score_list[i].get_visited_num() < score_list[min_index].get_visited_num():
                min_index = i
            i += 1
        return min_index

    def __select_ucb(self, 
                   score_list : 'list[Seed]') -> int:
        score_list_temp = score_list.copy()
        score_list_temp.sort(key=Seed.takeScore)
        if random.random() <= 0.05:
            return random.randrange(start = 0, stop = len(score_list), step = 1)
        return score_list.index(score_list_temp[-1])

    def __select_mcts(self, 
                    score_list : 'list[Seed]') -> int:
        pass


class SeedPooling:
    def __init__(self, 
                 save_path : str,
                 init_path : str = None,
                 select_policy : SeedSelection = SeedSelection(SeedSelectionPolicy.RandomSelection),
                 reload : bool = False):
        self.__pooling = list()
        self.save_path = save_path
        self.select_policy = select_policy
        if init_path is not None and os.path.exists(init_path):
            if reload == False:
                df = pd.read_csv(init_path)
                df.drop_duplicates(subset=['text'], inplace=True)
                for item in iter(df['text'].tolist()):
                    self.__pooling.append(Seed(context=item))
            else:
                df = pd.read_csv(init_path)
                for index, row in df.iterrows():
                    self.__pooling.append(Seed(context=row['text']))
                    self.__pooling[-1].score = row['score']
                    self.__pooling[-1].success_attack_num = row['attacked']
                    self.__pooling[-1].__visited_num = row['visited']

    def get_length(self):
        return len(self.__pooling)

    def get_all_visited_num(self) -> int:
        result = 0
        for seed in self.__pooling:
            result += seed.get_visited_num()
        return result

    def add_seed(self, 
                 context : str, score=100):
        for item in iter(self.__pooling):
            if context == item.context:
                return
        
        seed = Seed(context)
        seed.success_attack_num = 1
        seed.score = score
        seed.visit()
        self.__pooling.append(seed)

    # def __visit(self, 
    #           index : int) -> str:
    #     if index >= len(self.__pooling) or index < 0:
    #         return None
    #     return self.__pooling[index].visit()
    
    def remove_seed(self, 
                      seed : Seed):
        index = self.__find_seed_index(seed)
        if index >= len(self.__pooling) or index < 0:
            return None
        del self.__pooling[index]

    def find_seed(self, 
                  context : str) -> Seed:
        for item in self.__pooling:
            if item.context == context:
                return item

    def visit_by_index(self, 
                       index : int) -> Seed:
        if index >= len(self.__pooling) or index < 0:
            return None
        return self.__pooling[index]

    def visit_by_context(self, 
                         context : str) -> Seed:
        return self.find_seed(context)
    
    def visit_max_score_seed(self) -> Seed:
        self.sort_by_score()
        return self.__pooling[-1]
    
    def visit_min_score_seed(self) -> Seed:
        self.sort_by_score()
        return self.__pooling[0]

    def get_seed_index(self, 
                       context : str) -> int:
        try:
            return self.__pooling.index(self.find_seed(context))
        except ValueError:
            return -1

    def __find_seed_index(self, 
                       seed : Seed) -> int:
        try:
            return self.__pooling.index(seed)
        except ValueError:
            return -1

    def sort_by_score(self):
        self.__pooling.sort(key=Seed.takeScore)

    def save_current(self):
        result = list()
        score = list()
        visited = list()
        attacked = list()
        asr = list()
        for seed in self.__pooling:
            result.append(seed.context)
            score.append(seed.score)
            attacked.append(seed.success_attack_num)
            visited.append(seed.get_visited_num())
            if seed.get_visited_num() == 0:
                asr.append('/')
            else :
                asr.append(seed.success_attack_num / seed.get_visited_num())

        data = {
            "text" : result,
            "score" : score,
            "attacked" : attacked,
            "visited" : visited,
            "ASR" : asr
        }

        df = pd.DataFrame(data=data, columns=['text', 'score', 'attacked', 'visited', 'ASR'])
        df.to_csv(self.save_path)

    # def test(self):
    #     for item in self.__pooling:
    #         print("seed: " + str(item.context) + " score: " + str(item.score))

    def select(self) -> Seed:
        index = self.select_policy.select(self.__pooling)
        return self.visit_by_index(index)

    def random_visit(self) -> Seed:
        index = random.randrange(start = 0, stop = len(self.__pooling), step = 1)
        return self.visit_by_index(index)



if __name__ == "__main__":
    pass