#!/usr/bin/env python3
from abc import ABC, abstractmethod
import inspect
import random
from typing import Union
from preferences.Item import Item
from preferences.CriterionName import CriterionName
from preferences.Value import Value
import numpy as np
import pandas as pd


class PreferenceModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_value_from_data(self, item: Item, criterion_name: CriterionName) -> Value:
        pass


class IntervalProfileCSV(PreferenceModel):

    def __init__(self,  map_item_criterion: dict[Item, dict[CriterionName, Union[int, float]]]) -> None:
        super().__init__()
        self.map_item_criterion = map_item_criterion
        self.profile_df = self.__get_profile()

    def __get_profile(self, filename: str = 'profiles.csv') -> pd.DataFrame:
        return pd.read_csv(filename, sep=',', index_col='PROFILE')

    def get_value_from_data(self, item: Item, criterion_name: CriterionName) -> Value:

        value_attributes = inspect.getmembers(
            Value, lambda a: not(inspect.isroutine(a)))

        value_list = [a for a in value_attributes if not(
            (a[0].startswith('__') and a[0].endswith('__')) or 'name' in a[0] or 'value' in a[0])]

        value_list = sorted(value_list, key=lambda x: x[1].value)

        profiles = self.profile_df[criterion_name.name].to_numpy()

        real_value = self.map_item_criterion[item.get_name()][criterion_name]

        value_idx = np.argwhere(real_value > profiles)[-1][0]

        value = value_list[value_idx][1]

        return value


class RandomIntervalProfile(PreferenceModel):

    def __init__(self,  map_item_criterion: dict[Item, dict[CriterionName, Union[int, float]]], verbose: bool = True) -> None:
        super().__init__()
        self.map_item_criterion = pd.DataFrame(map_item_criterion)

        value_attributes = inspect.getmembers(
            Value, lambda a: not(inspect.isroutine(a)))

        self.value_list = [a for a in value_attributes if not(
            (a[0].startswith('__') and a[0].endswith('__')) or 'name' in a[0] or 'value' in a[0])]

        self.value_list = sorted(self.value_list, key=lambda x: x[1].value)

        self.__get_profile()

        if verbose:
            print('Generated Random Profiles: ')
            print('---------------------------')
            print(pd.DataFrame(self.criterion_profile)[1:-1])
            print('---------------------------')

    def __get_profile(self):
        self.criterion_profile = {}
        criteria_list = self.map_item_criterion.index

        for criterion in criteria_list:

            max_value = self.map_item_criterion.loc[criterion].max()

            profiles = np.concatenate(
                ([-np.inf], max_value*np.random.random(len(self.value_list)-1), [np.inf]))
            profiles.sort()
            self.criterion_profile[criterion] = profiles

    def get_value_from_data(self, item: Item, criterion_name: CriterionName) -> Value:

        real_value = self.map_item_criterion[item.get_name(
        )][criterion_name.name]

        profiles = self.criterion_profile[criterion_name.name]
        value_idx = np.argwhere(real_value > profiles)[-1][0]

        value = self.value_list[value_idx][1]

        return value
