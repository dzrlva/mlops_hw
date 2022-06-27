from typing import List, Union
from insert import insert_vendor, update
from delete import delete_part
from get import get_by_id, get_by_parent_id
from pydantic import BaseModel

class Tree(BaseModel):
    type: str
    name: str
    id: str
    parentId: str = None
    price: int = None
    date: str
    children: List['Tree'] = None
    
class Service:
    def import_test_data(self, type, name, id, price, parentId, date):
        insert_vendor(type, name, id, price, parentId, date)
        return name

    def import_data(self, req):
        for item in req.items:
            item.date = req.updateDate
            row = get_by_id(item.id)
            if item.parentId == "None":
                item.parentId = None
            if row:
                update(item.type, item.name, item.id, item.price, item.parentId, item.date)
            else:
                insert_vendor(item.type, item.name, item.id, item.price, item.parentId, item.date)
        return req.updateDate
    
    def delete_data(self, id):
        row = get_by_id(id)
        if row[0] == "CATEGORY":
            for item in get_by_parent_id(row[2]):
                self.delete_data(item[2])
            delete_part(row[2])
        else:
            delete_part(row[2])
        return None

    def get_info(self, req):
        parent = get_by_id(req)
        if parent == None:
            return None
        tr = Tree(type = parent[0], name = parent[1], id = parent[2], price = parent[3], parentId = parent[4], date = parent[5])
        if tr.type == "CATEGORY":
            children_list = []
            for item in get_by_parent_id(tr.id):
                children_list.append(self.get_info(item[2]))
            tr.children = children_list
        return tr
    
    def has_id(self, id):
        return get_by_id(id)
