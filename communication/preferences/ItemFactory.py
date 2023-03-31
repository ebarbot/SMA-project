#!/usr/bin/env python3
from enum import Enum
from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from typing import List, Union
from preferences.Item import Item
from preferences.CriterionName import CriterionName
import pandas as pd


class ItemFactory(ABC):

    @abstractmethod
    def create(self) -> tuple[List[Item], dict[Item, CriterionName]]:
        pass


class ItemCreator_CSV(ItemFactory):
    def __init__(self, path: str = '.', filename: str = 'items.csv', file_separator: str = ','):
        self.items_df = None
        self.filename = Path(path, filename)
        self.item_criterion = {}
        self.file_separator = file_separator

        self.items_df = self.__get_data()
        self.__sanity_check_criteria()

    def create(self) -> tuple[List[Item], dict[Item, CriterionName]]:
        return self.__create_items(), self.__create_item_criterion_map()

    def __create_items(self) -> List[Item]:
        self.items_list = []
        for idx, item in self.items_df.iterrows():
            self.items_list.append(
                Item(idx, item['DESCRIPTION']))

        return self.items_list

    def __create_item_criterion_map(self) -> dict[Item, dict[CriterionName, Union[int, float]]]:

        criteria = set(self.items_df.columns) - \
            set(['ITEM_NAME', 'DESCRIPTION'])
        for item_name, item in self.items_df.iterrows():
            self.item_criterion[item_name] = {}
            for criterion in criteria:
                self.item_criterion[item_name][criterion] = item[criterion]

        return self.item_criterion

    def __get_data(self):
        return pd.read_csv(self.filename, sep=self.file_separator, index_col='ITEM_NAME')

    def __sanity_check_criteria(self) -> None:
        # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
        attributes = inspect.getmembers(
            CriterionName, lambda a: not(inspect.isroutine(a)))

        criteria_names = sorted(set([a[0] for a in attributes if not(
            (a[0].startswith('__') and a[0].endswith('__')) or 'name' in a[0] or 'value' in a[0])]))

        columns = sorted((set(self.items_df.columns) -
                         set(['ITEM_NAME', 'DESCRIPTION'])))

        assert criteria_names == columns, "Criteria names in CSV file do not match the ones in CriterionName class"
