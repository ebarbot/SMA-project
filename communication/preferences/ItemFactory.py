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
    def create_items(self) -> List[Item]:
        pass

    @abstractmethod
    def create_item_criterion_map(self) -> dict[Item, CriterionName]:
        pass

class ItemCreator_CSV(ItemFactory):
    def __init__(self, path: str = '..', filename: str = 'items.csv', file_separator: str = ','):
        self.items_df = None
        self.filename = Path(path, filename)
        self.item_criterion = {}
        self.file_separator = file_separator
    
        self.items_df = self._get_data()
        self._sanity_check_criteria()
        
    def create(self) -> tuple[List[Item], dict[Item, CriterionName]]:
        return self._create_items(), self.create_item_criterion_map()
    
    def _create_items(self) -> List[Item]:
        self.items_list = []
        for item in self.items_df:
            self.items_list.append(Item(item['ITEM_NAME'], item['DESCRIPTION']))

        return self.items_list 
    
    def _create_item_criterion_map(self) -> dict[Item, dict[CriterionName, Union[int, float]]]:

        criteria = self.items_df.columns - 'ITEM_NAME' - 'DESCRIPTION'
        for item in self.items_df:
            self.item_criterion[item['ITEM_NAME']] = {}
            for criterion in criteria:
                self.item_criterion[item['ITEM_NAME']][criterion] = item[criterion]

        return self.item_criterion

    def _get_data(self):
        return pd.read_csv(self.filename, sep=self.file_separator, index_col='ITEM_NAME')
    
    def _sanity_check_criteria(self) -> None:
        # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
        attributes = inspect.getmembers(CriterionName, lambda a: not(inspect.isroutine(a)))
        criteria_names = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]

        assert criteria_names == (self.items_df.columns - 'ITEM_NAME' - 'DESCRIPTION'), "Criteria names in CSV file do not match the ones in CriterionName class"
    